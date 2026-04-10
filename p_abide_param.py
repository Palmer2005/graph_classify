import sys
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import numpy as np
import argparse
import copy
import random
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from scipy import stats

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from imports.ABIDEDataset import ABIDEDataset

EPS = 1e-10
device = torch.device("cuda")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ======================= 模型定义（与 p_ds30.py 完全一致）=======================

class LLMFeatureGater(nn.Module):
    def __init__(self, indim, hidden_dim=96, n_layers=1, n_heads=2, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(indim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.gate_head = nn.Linear(hidden_dim, indim)

    def forward(self, x, batch):
        x_seq = self.proj(x).unsqueeze(0)
        h = self.encoder(x_seq).squeeze(0)
        gate = torch.sigmoid(self.gate_head(h))
        scale = 0.5 + 0.5 * gate
        return x * scale, gate


class HCGPool(nn.Module):
    def __init__(self, in_channels, num_clusters, temp=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.cluster_proj = nn.Linear(in_channels, num_clusters)
        self.temp = temp

    def forward(self, x, edge_index, edge_attr, batch):
        device = x.device
        logits = self.cluster_proj(x) / max(self.temp, 1e-6)
        s_matrix = F.softmax(logits, dim=-1)

        B = 1 if batch.numel() == 0 else int(batch.max().item()) + 1

        adj = to_dense_adj(edge_index, batch, edge_attr=edge_attr)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        elif adj.dim() == 4:
            adj = adj.squeeze(-1) if adj.size(-1) == 1 else adj.sum(dim=-1)

        B_adj, Nmax, _ = adj.shape
        counts = torch.bincount(batch, minlength=B).to(device)
        s_batched = s_matrix.new_zeros((B, Nmax, self.num_clusters))
        x_batched = x.new_zeros((B, Nmax, self.in_channels))

        ptr = 0
        for i in range(B):
            ni = int(counts[i].item())
            if ni > 0:
                s_batched[i, :ni, :] = s_matrix[ptr:ptr + ni]
                x_batched[i, :ni, :] = x[ptr:ptr + ni]
                ptr += ni

        s_batched = s_batched.clamp(min=EPS)
        x_super = torch.bmm(s_batched.transpose(1, 2), x_batched).view(-1, self.in_channels)

        adj_super = torch.bmm(torch.bmm(s_batched.transpose(1, 2), adj), s_batched)
        for i in range(adj_super.size(0)):
            Ai = adj_super[i]
            Ai = (Ai + Ai.t()) * 0.5
            row_sum = Ai.sum(dim=-1, keepdim=True).clamp(min=EPS)
            adj_super[i] = Ai / row_sum

        Bc = B * self.num_clusters
        block_adj = adj_super.new_zeros((Bc, Bc))
        for i in range(B):
            s, e = i * self.num_clusters, (i + 1) * self.num_clusters
            block_adj[s:e, s:e] = adj_super[i]

        edge_index_super, edge_attr_super = dense_to_sparse(block_adj)
        batch_super = torch.arange(B, device=device).repeat_interleave(self.num_clusters)
        return x_super, edge_index_super, edge_attr_super, batch_super, s_matrix


class GCN_Gater_Pool_Reg(nn.Module):
    """M3: GCN + Gater + HCGPool + Reg（全组件）"""
    def __init__(self, indim, nclass, num_clusters=8,
                 dropout=0.65, hidden=64, pool_temp=1.0,
                 llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5):
        super().__init__()
        self.num_clusters = num_clusters
        self.dropout_p = dropout
        self.llm_gater = LLMFeatureGater(
            indim=indim, hidden_dim=llm_hidden,
            n_layers=llm_layers, n_heads=llm_heads, dropout=llm_dropout
        )
        self.conv0 = GCNConv(indim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.pool1 = HCGPool(hidden, num_clusters=num_clusters, temp=pool_temp)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc1 = nn.Linear(hidden, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, nclass)

    def forward(self, x, edge_index, batch, edge_attr, pos):
        x_gated, gate = self.llm_gater(x, batch)

        x = F.dropout(F.relu(self.bn0(self.conv0(x_gated, edge_index, edge_attr))),
                      p=self.dropout_p, training=self.training)

        x1 = x
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.bn1(x)) + x1
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x_conv = x

        x_pooled, ei_p, ea_p, batch_p, s_matrix = self.pool1(x, edge_index, edge_attr, batch)

        x_pooled_in = x_pooled
        x_pooled = F.relu(self.bn2(self.conv2(x_pooled, ei_p, ea_p))) + x_pooled_in
        x_pooled = F.dropout(x_pooled, p=self.dropout_p, training=self.training)

        x_readout = global_mean_pool(x_pooled, batch_p)
        x_out = F.dropout(F.relu(self.bn_fc1(self.fc1(x_readout))),
                          p=self.dropout_p, training=self.training)
        x_out = self.fc2(x_out)
        return F.log_softmax(x_out, dim=-1), s_matrix, x_conv, gate


# ======================= 损失函数 =======================

def cluster_reg_loss(x_conv, s_matrix, edge_index, batch, num_clusters, spec_k):
    device = s_matrix.device
    adj = to_dense_adj(edge_index, batch)
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    elif adj.dim() == 4:
        adj = adj.squeeze(-1) if adj.size(-1) == 1 else adj.sum(dim=-1)

    B = adj.size(0)
    Nmax = adj.size(1)
    counts = torch.bincount(batch, minlength=B).to(device)
    s_batched = s_matrix.new_zeros((B, Nmax, num_clusters))

    ptr = 0
    for i in range(B):
        ni = int(counts[i].item())
        if ni > 0:
            s_batched[i, :ni, :] = s_matrix[ptr:ptr + ni]
            ptr += ni

    A_tilde = torch.bmm(s_batched.transpose(1, 2), torch.bmm(adj, s_batched))
    A_tilde_sum = torch.sum(A_tilde, dim=(1, 2), keepdim=True)
    A_tilde_norm = A_tilde / (A_tilde_sum + EPS)
    loss_link = torch.mean(
        torch.sum(A_tilde_norm, dim=(1, 2)) - 2 * torch.einsum('bii->b', A_tilde_norm)
    )
    entropy = -(s_matrix * torch.log(s_matrix + EPS)).sum(dim=1).mean()

    loss_orth = 0.
    loss_balance = 0.
    for i in range(B):
        ni = int(counts[i].item())
        if ni == 0:
            continue
        Si = s_batched[i, :ni, :]
        StS = Si.t() @ Si
        C = StS.shape[0]
        I = torch.eye(C, device=device)
        loss_orth = loss_orth + ((StS - I) ** 2).sum() / (C * C)
        p_vec = Si.sum(dim=0) / (ni + EPS)
        loss_balance = loss_balance + ((p_vec - 1.0 / C) ** 2).sum()
    loss_orth = loss_orth / (B + EPS)
    loss_balance = loss_balance / (B + EPS)

    loss_spec = 0.
    k_requested = max(1, min(spec_k, num_clusters))
    for i in range(B):
        ni = int(counts[i].item())
        if ni <= 1:
            continue
        Ai = adj[i, :ni, :ni]
        deg = Ai.sum(dim=1)
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ Ai @ D_inv_sqrt
        L = torch.eye(ni, device=device) - A_norm
        try:
            e_vals, e_vecs = torch.linalg.eigh(L)
        except RuntimeError:
            e_vals, e_vecs = torch.linalg.eigh(L + EPS * torch.eye(ni, device=device))
        k = min(k_requested, ni - 1)
        if k <= 0:
            continue
        U = e_vecs[:, :k]
        U = U / (U.norm(dim=0, keepdim=True) + EPS)
        Si = s_batched[i, :ni, :]
        StS = Si.t() @ Si
        reg = EPS * torch.eye(StS.size(0), device=device)
        try:
            StS_inv = torch.inverse(StS + reg)
        except RuntimeError:
            StS_inv = torch.pinverse(StS + reg)
        S_pinv = StS_inv @ Si.t()
        U_hat = Si @ (S_pinv @ U)
        loss_spec = loss_spec + ((U - U_hat) ** 2).sum() / (ni * k + EPS)
    loss_spec = loss_spec / (B + EPS)

    return {
        'loss_link': loss_link,
        'entropy': entropy,
        'loss_orth': loss_orth,
        'loss_balance': loss_balance,
        'loss_spec': loss_spec
    }


# ======================= 工具函数 =======================

def drop_edges(edge_index, edge_attr, p):
    if p <= 0 or edge_index is None:
        return edge_index, edge_attr
    E = edge_index.size(1)
    keep_mask = torch.rand(E, device=edge_index.device) > p
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    ei = edge_index[:, keep_mask]
    ea = edge_attr[keep_mask] if edge_attr is not None else None
    return ei, ea


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


# ======================= 单次实验 =======================

def run_single_experiment(run_idx, base_seed, dataset, opt):
    current_seed = base_seed + run_idx
    set_seed(current_seed)

    total = len(dataset)
    test_n  = max(1, int(total * 0.2))
    val_n   = max(1, int(total * 0.2))
    train_n = total - val_n - test_n
    lengths = [train_n, val_n, test_n]
    if sum(lengths) < total:
        lengths[0] += total - sum(lengths)

    train_ds, val_ds, test_ds = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(current_seed)
    )
    train_loader = DataLoader(train_ds, batch_size=opt.batchSize, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=opt.batchSize, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=opt.batchSize, shuffle=False)

    model = GCN_Gater_Pool_Reg(
        opt.indim, opt.nclass,
        num_clusters=opt.num_clusters,
        dropout=opt.dropout,
        hidden=64, pool_temp=1.0,
        llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=2e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def compute_reg_loss(s_matrix, x_conv, ei_orig, b_orig):
        losses_dict = cluster_reg_loss(
            x_conv, s_matrix, ei_orig, b_orig,
            num_clusters=opt.num_clusters, spec_k=3
        )
        base = losses_dict['loss_link'] + 0.5 * losses_dict['entropy']
        return (opt.lamb_cluster * base
                + 0.015 * losses_dict['loss_orth']
                + 0.002 * losses_dict['loss_balance']
                + 0.001 * losses_dict['loss_spec'])

    def fwd(x, ei, b, ea, pos):
        out, s, xc, g = model(x, ei, b, ea, pos)
        return out, s, xc, g

    def total_loss(out, s, xc, g, ei_orig, b_orig, y):
        lc = F.nll_loss(out, y)
        lg = (g ** 2).mean() * 0.0007
        lr_ = compute_reg_loss(s, xc, ei_orig, b_orig)
        return 1.0 * lc + lr_ + lg

    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1e10

    for epoch in range(50):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            ei_a, ea_a = drop_edges(data.edge_index, data.edge_attr, 0.2)
            x_a = data.x.clone()
            mask = torch.rand(x_a.size(0), device=device) < 0.2
            x_a[mask] = 0.0
            out, s, xc, g = fwd(x_a, ei_a, data.batch, ea_a, data.pos)
            loss = total_loss(out, s, xc, g, data.edge_index, data.batch, data.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, s, xc, g = fwd(data.x, data.edge_index, data.batch,
                                    data.edge_attr, data.pos)
                val_loss_sum += total_loss(out, s, xc, g,
                                           data.edge_index, data.batch, data.y).item() * data.num_graphs
        val_loss = val_loss_sum / len(val_ds)
        if val_loss < best_val_loss and epoch > 5:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    model.eval()

    all_probs, all_preds, all_trues = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, _, _, _ = fwd(data.x, data.edge_index, data.batch,
                               data.edge_attr, data.pos)
            prob = F.softmax(out, dim=1)[:, 1]
            pred = out.max(1)[1]
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_trues.append(data.y.cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = 0.0
    return acc, auc


# ======================= 参数验证主循环 =======================

def parse_args():
    parser = argparse.ArgumentParser(description='ABIDE Parameter Sensitivity Validation')
    parser.add_argument('--dataroot', type=str,
                        default='/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/data/ABIDE_pcp/cpac/filt_noglobal')
    parser.add_argument('--indim', type=int, default=182)
    parser.add_argument('--nroi', type=int, default=182)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--batchSize', type=int, default=182)
    parser.add_argument('--num_runs', type=int, default=5,
                        help='每个参数配置的随机实验次数')
    return parser.parse_args()


def main():
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler('abide_param_search.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

    args = parse_args()
    set_seed(123)

    path = args.dataroot
    name = 'ABIDE'
    dataset = ABIDEDataset(path, name)
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0

    logger.info(f"Dataset: {name}, Total size: {len(dataset)}")

    # Default hyperparameters
    DEFAULTS = SimpleNamespace(
        num_clusters=8,
        lr=0.0008,
        dropout=0.65,
        lamb_cluster=0.008,
    )

    param_grid = {
        'num_clusters': [4, 6, 8, 10, 12],
        'lr':           [0.0002, 0.0005, 0.0008, 0.001, 0.002],
        'dropout':      [0.3, 0.5, 0.65, 0.7, 0.8],
        'lamb_cluster': [0.002, 0.005, 0.008, 0.01, 0.02],
    }

    logger.info("\n" + "=" * 60)
    logger.info("ABIDE Parameter Sensitivity — M3 (gcn_gater_pool_reg)")
    logger.info(f"Defaults: {vars(DEFAULTS)}")
    logger.info(f"Runs per config: {args.num_runs}")
    logger.info("=" * 60)

    for param_name, values in param_grid.items():
        logger.info(f"\n--- Parameter: {param_name} ---")
        logger.info(f"{'Value':>15}  {'ACC mean±std':>18}  {'AUC mean±std':>18}")
        logger.info("-" * 55)

        for val in values:
            # Build opt with this parameter varied, others at default
            opt = SimpleNamespace(
                indim=args.indim,
                nroi=args.nroi,
                nclass=args.nclass,
                batchSize=args.batchSize,
                num_clusters=DEFAULTS.num_clusters,
                lr=DEFAULTS.lr,
                dropout=DEFAULTS.dropout,
                lamb_cluster=DEFAULTS.lamb_cluster,
            )
            setattr(opt, param_name, val)

            acc_list, auc_list = [], []
            for r in range(args.num_runs):
                acc, auc = run_single_experiment(r, 42, dataset, opt)
                acc_list.append(acc)
                auc_list.append(auc)

            acc_arr = np.array(acc_list)
            auc_arr = np.array(auc_list)
            acc_mean = acc_arr.mean() * 100
            acc_std  = acc_arr.std()  * 100
            auc_mean = auc_arr.mean() * 100
            auc_std  = auc_arr.std()  * 100

            row = (f"{str(val):>15}  "
                   f"{acc_mean:6.2f} ± {acc_std:5.2f}%  "
                   f"{auc_mean:6.2f} ± {auc_std:5.2f}%")
            logger.info(row)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

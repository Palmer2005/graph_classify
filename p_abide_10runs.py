import sys
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import numpy as np
import argparse
import copy
import random
import logging

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

# ======================= 模型定义 =======================
# 消融顺序:
# M0: baseline           — 纯 GCN
# M1: gcn_gater          — GCN + LLM Feature Gater（无 HCGPool）
# M2: gcn_pool_reg       — GCN + HCGPool + Reg（无 Gater）
# M3: gcn_gater_pool_reg — GCN + Gater + HCGPool + Reg（全组件）


class LLMFeatureGater(nn.Module):
    """Transformer-based 特征门控模块"""
    def __init__(self, indim, hidden_dim=96, n_layers=1, n_heads=2, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(indim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.gate_head = nn.Linear(hidden_dim, indim)

    def forward(self, x, batch):
        x_seq = self.proj(x).unsqueeze(0)           # [1, N, hidden_dim]
        h = self.encoder(x_seq).squeeze(0)           # [N, hidden_dim]
        gate = torch.sigmoid(self.gate_head(h))      # [N, indim]
        scale = 0.5 + 0.5 * gate                     # (0.5, 1.0)
        x_gated = x * scale
        return x_gated, gate


class HCGPool(nn.Module):
    """层次化聚类图池化"""
    def __init__(self, in_channels, num_clusters, temp=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.cluster_proj = nn.Linear(in_channels, num_clusters)
        self.temp = temp

    def forward(self, x, edge_index, edge_attr, batch):
        device = x.device
        logits = self.cluster_proj(x) / max(self.temp, 1e-6)
        s_matrix = F.softmax(logits, dim=-1)   # [N_all, num_clusters]

        if batch.numel() == 0:
            B = 1
        else:
            B = int(batch.max().item()) + 1

        adj = to_dense_adj(edge_index, batch, edge_attr=edge_attr)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        elif adj.dim() == 4:
            if adj.size(-1) == 1:
                adj = adj.squeeze(-1)
            else:
                adj = adj.sum(dim=-1)

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
            start = i * self.num_clusters
            end = start + self.num_clusters
            block_adj[start:end, start:end] = adj_super[i]

        edge_index_super, edge_attr_super = dense_to_sparse(block_adj)
        batch_super = torch.arange(B, device=device).repeat_interleave(self.num_clusters)

        return x_super, edge_index_super, edge_attr_super, batch_super, s_matrix


# ---- M0 ----
class BaselineGCN(nn.Module):
    """M0: 纯 GCN (baseline)"""
    def __init__(self, indim, nclass, hidden=64, dropout=0.65):
        super().__init__()
        self.conv0 = GCNConv(indim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc1 = nn.Linear(hidden, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, nclass)
        self.dropout_p = dropout

    def forward(self, x, edge_index, batch, edge_attr, pos):
        x = self.conv0(x, edge_index, edge_attr)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_readout = global_mean_pool(x, batch)
        x_out = F.relu(self.bn_fc1(self.fc1(x_readout)))
        x_out = F.dropout(x_out, p=self.dropout_p, training=self.training)
        x_out = self.fc2(x_out)
        return F.log_softmax(x_out, dim=-1)   # 返回: output


# ---- M1 ----
class GCN_Gater(nn.Module):
    """M1: GCN + LLM Feature Gater（无 HCGPool，无残差）"""
    def __init__(self, indim, nclass, hidden=64, dropout=0.65,
                 llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5):
        super().__init__()
        self.llm_gater = LLMFeatureGater(
            indim=indim, hidden_dim=llm_hidden,
            n_layers=llm_layers, n_heads=llm_heads, dropout=llm_dropout
        )
        self.conv0 = GCNConv(indim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc1 = nn.Linear(hidden, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, nclass)
        self.dropout_p = dropout

    def forward(self, x, edge_index, batch, edge_attr, pos):
        x_gated, gate = self.llm_gater(x, batch)

        x = self.conv0(x_gated, edge_index, edge_attr)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_readout = global_mean_pool(x, batch)
        x_out = F.relu(self.bn_fc1(self.fc1(x_readout)))
        x_out = F.dropout(x_out, p=self.dropout_p, training=self.training)
        x_out = self.fc2(x_out)
        return F.log_softmax(x_out, dim=-1), gate   # 返回: output, gate


# ---- M2 ----
class GCN_Pool_Reg(nn.Module):
    """M2: GCN + HCGPool + Reg（无 Gater）"""
    def __init__(self, indim, nclass, num_clusters=8,
                 dropout=0.65, hidden=64, pool_temp=1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.dropout_p = dropout

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
        x = self.conv0(x, edge_index, edge_attr)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x1 = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + x1
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_conv = x

        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, s_matrix = \
            self.pool1(x, edge_index, edge_attr, batch)

        x_pooled_in = x_pooled
        x_pooled = self.conv2(x_pooled, edge_index_pooled, edge_attr_pooled)
        x_pooled = self.bn2(x_pooled)
        x_pooled = F.relu(x_pooled)
        x_pooled = x_pooled + x_pooled_in
        x_pooled = F.dropout(x_pooled, p=self.dropout_p, training=self.training)

        x_readout = global_mean_pool(x_pooled, batch_pooled)
        x_out = F.relu(self.bn_fc1(self.fc1(x_readout)))
        x_out = F.dropout(x_out, p=self.dropout_p, training=self.training)
        x_out = self.fc2(x_out)
        return F.log_softmax(x_out, dim=-1), s_matrix, x_conv   # 返回: output, s_matrix, x_conv


# ---- M3 ----
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

        x = self.conv0(x_gated, edge_index, edge_attr)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x1 = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + x1
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_conv = x

        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, s_matrix = \
            self.pool1(x, edge_index, edge_attr, batch)

        x_pooled_in = x_pooled
        x_pooled = self.conv2(x_pooled, edge_index_pooled, edge_attr_pooled)
        x_pooled = self.bn2(x_pooled)
        x_pooled = F.relu(x_pooled)
        x_pooled = x_pooled + x_pooled_in
        x_pooled = F.dropout(x_pooled, p=self.dropout_p, training=self.training)

        x_readout = global_mean_pool(x_pooled, batch_pooled)
        x_out = F.relu(self.bn_fc1(self.fc1(x_readout)))
        x_out = F.dropout(x_out, p=self.dropout_p, training=self.training)
        x_out = self.fc2(x_out)
        return F.log_softmax(x_out, dim=-1), s_matrix, x_conv, gate   # 返回: output, s_matrix, x_conv, gate


# ======================= 损失函数 =======================

def cluster_reg_loss(x_conv, s_matrix, edge_index, batch, num_clusters, spec_k):
    device = s_matrix.device
    adj = to_dense_adj(edge_index, batch)
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    elif adj.dim() == 4:
        if adj.size(-1) == 1:
            adj = adj.squeeze(-1)
        else:
            adj = adj.sum(dim=-1)

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
            L = L + (EPS * torch.eye(ni, device=device))
            e_vals, e_vecs = torch.linalg.eigh(L)
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
        coeff = S_pinv @ U
        U_hat = Si @ coeff
        loss_spec = loss_spec + ((U - U_hat) ** 2).sum() / (ni * k + EPS)
    loss_spec = loss_spec / (B + EPS)

    losses = {
        'loss_link': loss_link,
        'entropy': entropy,
        'loss_orth': loss_orth,
        'loss_balance': loss_balance,
        'loss_spec': loss_spec
    }
    return losses, s_batched, counts


# ======================= 工具函数 =======================

def drop_edges(edge_index, edge_attr, p):
    if p <= 0 or edge_index is None:
        return edge_index, edge_attr
    E = edge_index.size(1)
    keep_mask = torch.rand(E, device=edge_index.device) > p
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    ei = edge_index[:, keep_mask]
    ea = None
    if edge_attr is not None:
        try:
            ea = edge_attr[keep_mask]
        except Exception:
            ea = edge_attr
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


# ======================= 参数解析 =======================

def parse_args():
    parser = argparse.ArgumentParser(description='ABIDE Graph Classification — 10 Random Runs')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batchSize', type=int, default=182)   # ABIDE: 182 ROI
    parser.add_argument('--dataroot', type=str,
                        default='/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/data/ABIDE_pcp/cpac/filt_noglobal')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--stepsize', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weightdecay', type=float, default=2e-5)

    parser.add_argument('--lamb0', type=float, default=1.0)
    parser.add_argument('--lamb_cluster', type=float, default=0.008)
    parser.add_argument('--lamb_spec', type=float, default=0.001)
    parser.add_argument('--spec_k', type=int, default=3)

    parser.add_argument('--indim', type=int, default=182)       # ABIDE: 182 ROI
    parser.add_argument('--nroi', type=int, default=182)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--save_path', type=str, default='./model_abide_randomsplit/')

    parser.add_argument('--p_edge_dropout', type=float, default=0.2)
    parser.add_argument('--p_feat_mask', type=float, default=0.2)
    parser.add_argument('--num_clusters', type=int, default=8)

    parser.add_argument('--model_type', type=str, default='gcn_gater_pool_reg',
                        choices=['baseline', 'gcn_gater', 'gcn_pool_reg', 'gcn_gater_pool_reg'])

    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--num_runs', type=int, default=10,
                        help='重复运行次数，用于统计均值/方差和 p 值')
    return parser.parse_args()


# ======================= 模型构建 =======================

def build_model(opt):
    if opt.model_type == 'baseline':
        model = BaselineGCN(opt.indim, opt.nclass).to(device)

    elif opt.model_type == 'gcn_gater':
        model = GCN_Gater(
            opt.indim, opt.nclass,
            llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5
        ).to(device)

    elif opt.model_type == 'gcn_pool_reg':
        model = GCN_Pool_Reg(
            opt.indim, opt.nclass,
            num_clusters=opt.num_clusters,
            dropout=0.65, hidden=64, pool_temp=1.0,
        ).to(device)

    elif opt.model_type == 'gcn_gater_pool_reg':
        model = GCN_Gater_Pool_Reg(
            opt.indim, opt.nclass,
            num_clusters=opt.num_clusters,
            dropout=0.65, hidden=64, pool_temp=1.0,
            llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5,
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {opt.model_type}")
    return model


# ======================= 训练与评估 =======================

def run_single_experiment(run_idx, base_seed, dataset, opt):
    current_seed = base_seed + run_idx
    set_seed(current_seed)
    logger.info(f"===== Run {run_idx+1}/{opt.num_runs}, seed={current_seed} =====")

    total = len(dataset)
    test_n = max(1, int(total * opt.test_size))
    val_n  = max(1, int(total * opt.val_size))
    train_n = total - val_n - test_n
    lengths = [train_n, val_n, test_n]
    if sum(lengths) < total:
        lengths[0] += total - sum(lengths)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(current_seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=opt.batchSize, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=opt.batchSize, shuffle=False)

    model = build_model(opt)

    if opt.optim == 'Adam':
        optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
    elif opt.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9,
                                    weight_decay=opt.weightdecay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {opt.optim}")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    # ---------- 辅助：pool+reg 损失（M2/M3 共用）----------
    def compute_reg_loss(s_matrix, x_conv, edge_index_orig, batch_orig):
        losses_dict, _, _ = cluster_reg_loss(
            x_conv, s_matrix, edge_index_orig, batch_orig,
            num_clusters=opt.num_clusters, spec_k=opt.spec_k
        )
        loss_cluster_base = losses_dict['loss_link'] + 0.5 * losses_dict['entropy']
        return (opt.lamb_cluster * loss_cluster_base
                + 0.015 * losses_dict['loss_orth']
                + 0.002 * losses_dict['loss_balance']
                + opt.lamb_spec * losses_dict['loss_spec'])

    # ---------- 统一前向 ----------
    def forward_model(x, edge_index, batch, edge_attr, pos):
        mt = opt.model_type
        if mt == 'baseline':
            output = model(x, edge_index, batch, edge_attr, pos)
            return output, None, None, None
        elif mt == 'gcn_gater':
            output, gate = model(x, edge_index, batch, edge_attr, pos)
            return output, None, None, gate
        elif mt == 'gcn_pool_reg':
            output, s_matrix, x_conv = model(x, edge_index, batch, edge_attr, pos)
            return output, s_matrix, x_conv, None
        elif mt == 'gcn_gater_pool_reg':
            output, s_matrix, x_conv, gate = model(x, edge_index, batch, edge_attr, pos)
            return output, s_matrix, x_conv, gate
        else:
            raise ValueError(f"Unknown model_type: {mt}")

    # ---------- 统一 loss 计算 ----------
    def compute_loss(output, s_matrix, x_conv, gate, edge_index_orig, batch_orig, y):
        mt = opt.model_type
        loss_c = F.nll_loss(output, y)

        if mt == 'baseline':
            return opt.lamb0 * loss_c

        elif mt == 'gcn_gater':
            loss_gate = (gate ** 2).mean() * 0.0007
            return opt.lamb0 * loss_c + loss_gate

        elif mt == 'gcn_pool_reg':
            loss_reg = compute_reg_loss(s_matrix, x_conv, edge_index_orig, batch_orig)
            return opt.lamb0 * loss_c + loss_reg

        elif mt == 'gcn_gater_pool_reg':
            loss_gate = (gate ** 2).mean() * 0.0007
            loss_reg = compute_reg_loss(s_matrix, x_conv, edge_index_orig, batch_orig)
            return opt.lamb0 * loss_c + loss_reg + loss_gate

        else:
            raise ValueError(f"Unknown model_type: {mt}")

    # ---------- 训练一个 epoch ----------
    def train_one_epoch(epoch):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            ei_aug, ea_aug = drop_edges(data.edge_index, data.edge_attr, opt.p_edge_dropout)
            x_aug = data.x.clone()
            if opt.p_feat_mask > 0:
                mask_nodes = (torch.rand(x_aug.size(0), device=device) < opt.p_feat_mask)
                x_aug[mask_nodes] = 0.0

            output, s_matrix, x_conv, gate = forward_model(
                x_aug, ei_aug, data.batch, ea_aug, data.pos)
            loss = compute_loss(output, s_matrix, x_conv, gate,
                                data.edge_index, data.batch, data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        scheduler.step()
        return loss_all / len(train_dataset)

    # ---------- 评估准确率 ----------
    def eval_acc(loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                output, _, _, _ = forward_model(
                    data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                pred = output.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    # ---------- 评估 loss（early stopping 用）----------
    def eval_loss(loader):
        model.eval()
        loss_all = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                output, s_matrix, x_conv, gate = forward_model(
                    data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                loss = compute_loss(output, s_matrix, x_conv, gate,
                                    data.edge_index, data.batch, data.y)
                loss_all += loss.item() * data.num_graphs
        return loss_all / len(loader.dataset)

    # ---------- 训练主循环 ----------
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    for epoch in range(opt.epoch, opt.n_epochs):
        tr_loss  = train_one_epoch(epoch)
        tr_acc   = eval_acc(train_loader)
        val_acc  = eval_acc(val_loader)
        val_loss = eval_loss(val_loader)
        logger.info(f"Run {run_idx+1}, Epoch {epoch:03d}, "
                    f"Train Loss {tr_loss:.4f}, Train Acc {tr_acc:.4f}, "
                    f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        if val_loss < best_loss and epoch > 5:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()

    # ---------- 测试集推理 ----------
    all_probs_list, all_preds_list, all_trues_list = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output, _, _, _ = forward_model(
                data.x, data.edge_index, data.batch, data.edge_attr, data.pos)

            prob = F.softmax(output, dim=1)[:, 1]
            pred = output.max(1)[1]

            all_probs_list.append(prob.cpu().numpy())
            all_preds_list.append(pred.cpu().numpy())
            all_trues_list.append(data.y.cpu().numpy())

    probs = np.concatenate(all_probs_list, axis=0)
    preds = np.concatenate(all_preds_list, axis=0)
    trues = np.concatenate(all_trues_list, axis=0)

    acc = accuracy_score(trues, preds)
    cm  = confusion_matrix(trues, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn + EPS)
        spe = tn / (tn + fp + EPS)
    else:
        sen, spe = 0.0, 0.0
    try:
        auc = roc_auc_score(trues, probs)
        f1  = f1_score(trues, preds, average='weighted')
    except Exception:
        auc, f1 = 0.0, 0.0

    logger.info(f"Run {run_idx+1} Test Acc={acc:.4f}, AUC={auc:.4f}, "
                f"Sen={sen:.4f}, Spe={spe:.4f}, F1={f1:.4f}")
    return acc, auc, sen, spe, f1


# ======================= 主函数 =======================

def main():
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler('abide_randomsplit_10runs_stats.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

    opt = parse_args()
    set_seed(123)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path, exist_ok=True)

    path = opt.dataroot
    name = 'ABIDE'
    dataset = ABIDEDataset(path, name)
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0

    logger.info(f"Dataset: {name}, Total size: {len(dataset)}")
    logger.info(f"Model type: {opt.model_type}")

    results = {'acc': [], 'auc': [], 'sen': [], 'spe': [], 'f1': []}
    base_seed = 42

    for i in range(opt.num_runs):
        acc, auc, sen, spe, f1 = run_single_experiment(i, base_seed, dataset, opt)
        results['acc'].append(acc)
        results['auc'].append(auc)
        results['sen'].append(sen)
        results['spe'].append(spe)
        results['f1'].append(f1)

    logger.info("\n========== FINAL STATS ==========")
    logger.info(f"Model: {opt.model_type}")
    for metric in ['acc', 'auc', 'sen', 'spe', 'f1']:
        values = np.array(results[metric])
        mean = values.mean() * 100
        std  = values.std()  * 100
        t_stat, p_val = stats.ttest_1samp(values, 0.5)
        logger.info(f"{metric.upper()}: {mean:.2f} ± {std:.2f}  (vs 0.5, p = {p_val:.2e})")
        logger.info(f"Raw {metric}: {values}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

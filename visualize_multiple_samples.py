import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

sys.path.append(script_dir)
from imports.ABIDEDataset import ABIDEDataset
from p_ab2 import NetworkLLM_GNN, set_seed

def compare_samples():
    """对比多个样本的聚类结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    dataroot = 'data/ABIDE_pcp/cpac/filt_noglobal'
    dataset = ABIDEDataset(dataroot, 'ABIDE')
    dataset._data.y = dataset._data.y.squeeze()
    dataset._data.x[dataset._data.x == float('inf')] = 0
    
    # 加载模型
    model = NetworkLLM_GNN(
        indim=182, nclass=2, num_clusters=8,
        dropout=0.65, hidden=64, pool_temp=1.0,
        llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5,
        use_aux=True
    ).to(device)
    
    model_path = '/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/model_llm_hcg_ablation_randomsplit/full_best.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 找几个正常人和自闭症患者的样本
    normal_indices = [i for i, data in enumerate(dataset) if data.y.item() == 0][:3]
    asd_indices = [i for i, data in enumerate(dataset) if data.y.item() == 1][:3]
    
    print("="*60)
    print("对比分析")
    print("="*60)
    
    for group_name, indices in [("正常人", normal_indices), ("自闭症", asd_indices)]:
        print(f"\n{group_name}组:")
        cluster_counts_list = []
        
        for idx in indices:
            sample = dataset[idx].to(device)
            
            with torch.no_grad():
                x_gated, _ = model.llm_gater(sample.x, sample.batch)
                x = model.conv0(x_gated, sample.edge_index, sample.edge_attr)
                x = model.bn0(x)
                x = F.relu(x)
                x1 = x
                x = model.conv1(x, sample.edge_index, sample.edge_attr)
                x = model.bn1(x)
                x = F.relu(x)
                x = x + x1
                
                logits = model.pool1.cluster_proj(x) / max(model.pool1.temp, 1e-6)
                s_matrix = F.softmax(logits, dim=-1)
                cluster_assignment = s_matrix.argmax(dim=1).cpu().numpy()
            
            # 统计每个簇的大小
            cluster_counts = [np.sum(cluster_assignment == i) for i in range(8)]
            cluster_counts_list.append(cluster_counts)
            
            print(f"  样本{idx}: {cluster_counts}")
        
        # 计算平均
        avg_counts = np.mean(cluster_counts_list, axis=0)
        print(f"  平均: {[f'{c:.1f}' for c in avg_counts]}")
        print(f"  使用的簇数: {np.sum(avg_counts > 0):.1f}")

if __name__ == "__main__":
    set_seed(42)
    compare_samples()

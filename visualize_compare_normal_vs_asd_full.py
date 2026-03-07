import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from imports.ABIDEDataset import ABIDEDataset
from p_ab2 import NetworkLLM_GNN, set_seed

def compare_normal_vs_asd_full():
    """完整版对比：正常人 vs 自闭症患者，多层切面"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据和模型
    print("Loading dataset and model...")
    dataroot = 'data/ABIDE_pcp/cpac/filt_noglobal'
    dataset = ABIDEDataset(dataroot, 'ABIDE')
    dataset._data.y = dataset._data.y.squeeze()
    dataset._data.x[dataset._data.x == float('inf')] = 0
    
    model = NetworkLLM_GNN(
        indim=182, nclass=2, num_clusters=8,
        dropout=0.65, hidden=64, pool_temp=1.0,
        llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5,
        use_aux=True
    ).to(device)
    
    model_path = '/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/model_llm_hcg_ablation_randomsplit/full_best.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载ROI坐标
    roi_coords = np.load('cc200_roi_coords_182.npy')
    print(f"✓ Loaded {len(roi_coords)} ROI coordinates")
    
    # 选择代表性样本
    normal_idx = 10  # 从之前输出选的均衡样本
    asd_idx = 0      # 当前的自闭症样本
    
    print(f"\nAnalyzing samples:")
    print(f"  Normal: Sample {normal_idx}")
    print(f"  ASD:    Sample {asd_idx}")
    
    # 创建大图：2行（正常/自闭症）× 8列（多个切面）
    fig = plt.figure(figsize=(32, 10))
    fig.suptitle('Normal vs ASD: Multi-Slice Clustering Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 高对比度颜色
    colors_high_contrast = [
        '#FF0000',  # 鲜红
        '#00FF00',  # 鲜绿
        '#D3D3D3',  # 浅灰
        '#FF00FF',  # 洋红
        '#FFFF00',  # 黄色
        '#00FFFF',  # 青色
        '#FFA500',  # 橙色
        '#800080',  # 紫色
    ]
    
    # 定义切面范围（显示更多ROI）
    axial_slices = [
        (-30, 0, "Axial Lower"),
        (0, 30, "Axial Upper")
    ]
    
    coronal_slices = [
        (-80, -20, "Coronal Post"),
        (-20, 40, "Coronal Ant")
    ]
    
    sagittal_slices = [
        (-70, 0, "Sagittal L"),
        (0, 70, "Sagittal R")
    ]
    
    all_slices = axial_slices + coronal_slices + sagittal_slices
    
    # 对两个样本分别处理
    for row, (sample_idx, label_name, label_color) in enumerate([
        (normal_idx, 'Normal', 'blue'),
        (asd_idx, 'ASD', 'red')
    ]):
        print(f"\nProcessing {label_name} sample {sample_idx}...")
        
        # 提取聚类信息
        sample = dataset[sample_idx].to(device)
        
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
        
        cluster_sizes = [np.sum(cluster_assignment == i) for i in range(8)]
        dominant_cluster = np.argmax(cluster_sizes)
        
        print(f"  Dominant cluster: {dominant_cluster} with {cluster_sizes[dominant_cluster]} ROIs")
        print(f"  Non-empty clusters: {np.sum(np.array(cluster_sizes) > 0)}/8")
        
        # 绘制6个切面
        for col, slice_info in enumerate(all_slices):
            ax = plt.subplot(2, 8, row*8 + col + 1)
            
            # 解析切面信息
            if col < 2:  # 轴位
                coord_idx, slice_name = 2, slice_info[2]
                x_idx, y_idx = 0, 1
                x_label, y_label = 'X (L-R)', 'Y (P-A)'
                x_lim, y_lim = (-70, 70), (-100, 70)
            elif col < 4:  # 冠状
                coord_idx, slice_name = 1, slice_info[2]
                x_idx, y_idx = 0, 2
                x_label, y_label = 'X (L-R)', 'Z (I-S)'
                x_lim, y_lim = (-70, 70), (-60, 80)
            else:  # 矢状
                coord_idx, slice_name = 0, slice_info[2]
                x_idx, y_idx = 1, 2
                x_label, y_label = 'Y (P-A)', 'Z (I-S)'
                x_lim, y_lim = (-100, 70), (-60, 80)
            
            # 选择当前切面的ROI
            mask_slice = (roi_coords[:, coord_idx] >= slice_info[0]) & \
                        (roi_coords[:, coord_idx] <= slice_info[1])
            n_rois_in_slice = mask_slice.sum()
            
            # 绘制主导簇（灰色背景）
            mask_dominant = (cluster_assignment == dominant_cluster) & mask_slice
            if mask_dominant.sum() > 0:
                ax.scatter(roi_coords[mask_dominant, x_idx], 
                          roi_coords[mask_dominant, y_idx], 
                          c=colors_high_contrast[dominant_cluster], 
                          s=60, alpha=0.25, edgecolors='gray', linewidth=0.5, zorder=1)
            
            # 绘制其他簇（鲜艳颜色）
            for cluster_id in range(8):
                if cluster_id == dominant_cluster:
                    continue
                mask = (cluster_assignment == cluster_id) & mask_slice
                if mask.sum() > 0:
                    ax.scatter(roi_coords[mask, x_idx], roi_coords[mask, y_idx], 
                              c=colors_high_contrast[cluster_id], 
                              s=100, alpha=0.9, edgecolors='black', linewidth=1.5, zorder=3)
            
            # 设置标题和标签
            if row == 0:
                ax.set_title(slice_name, fontsize=11, fontweight='bold')
            
            ax.set_xlabel(x_label, fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{label_name}\n{y_label}', fontsize=9, fontweight='bold', color=label_color)
            else:
                ax.set_ylabel(y_label, fontsize=9)
            
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # 显示ROI数量
            ax.text(0.02, 0.98, f'{n_rois_in_slice}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
            
            ax.set_aspect('equal')
        
        # 第7列：簇大小分布
        ax_bar = plt.subplot(2, 8, row*8 + 7)
        bars = ax_bar.bar(range(8), cluster_sizes, color=colors_high_contrast, 
                          edgecolor='black', linewidth=1.5)
        
        for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
            if size > 0:
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                           f'{size}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        if row == 0:
            ax_bar.set_title('Cluster Sizes', fontsize=11, fontweight='bold')
        ax_bar.set_xlabel('Cluster ID', fontsize=9)
        ax_bar.set_ylabel('ROI Count', fontsize=9)
        ax_bar.set_xticks(range(8))
        ax_bar.set_yscale('symlog')
        ax_bar.grid(axis='y', alpha=0.3)
        
        # 第8列：统计信息
        ax_stats = plt.subplot(2, 8, row*8 + 8)
        ax_stats.axis('off')
        
        # 计算统计指标
        probs = np.array(cluster_sizes) / 182
        entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
        max_entropy = np.log(8)
        norm_entropy = entropy / max_entropy
        gini = 1 - np.sum(probs ** 2)
        
        stats_text = f"{label_name}\n"
        stats_text += f"Sample {sample_idx}\n"
        stats_text += "="*25 + "\n\n"
        
        stats_text += f"Non-empty: {np.sum(np.array(cluster_sizes) > 0)}/8\n"
        stats_text += f"Largest: {max(cluster_sizes)} ({100*max(cluster_sizes)/182:.1f}%)\n"
        stats_text += f"Smallest: {min([s for s in cluster_sizes if s > 0])}\n\n"
        
        stats_text += f"Norm Entropy: {norm_entropy:.3f}\n"
        stats_text += f"(1.0=balanced)\n\n"
        
        stats_text += f"Gini Coeff: {gini:.3f}\n"
        stats_text += f"(0=equal,\n 1=unequal)\n\n"
        
        # 左右半球统计
        n_left = np.sum(roi_coords[cluster_assignment != dominant_cluster, 0] < 0)
        n_right = np.sum(roi_coords[cluster_assignment != dominant_cluster, 0] >= 0)
        stats_text += f"Minor clusters:\n"
        stats_text += f"L:{n_left} R:{n_right}\n"
        
        color = 'lightblue' if row == 0 else 'lightcoral'
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('comparison_normal_vs_asd_full.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 完整对比图已保存: comparison_normal_vs_asd_full.png")
    
    # 打印对比总结
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    
    for sample_idx, label_name in [(normal_idx, 'Normal'), (asd_idx, 'ASD')]:
        sample = dataset[sample_idx].to(device)
        
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
        
        cluster_sizes = [np.sum(cluster_assignment == i) for i in range(8)]
        probs = np.array(cluster_sizes) / 182
        gini = 1 - np.sum(probs ** 2)
        
        print(f"\n{label_name} (Sample {sample_idx}):")
        print(f"  Cluster distribution: {cluster_sizes}")
        print(f"  Gini coefficient: {gini:.3f}")
        print(f"  Dominant cluster占比: {100*max(cluster_sizes)/182:.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    set_seed(42)
    compare_normal_vs_asd_full()

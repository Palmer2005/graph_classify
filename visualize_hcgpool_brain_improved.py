import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

sys.path.append(script_dir)
from imports.ABIDEDataset import ABIDEDataset
from p_ab2 import NetworkLLM_GNN, set_seed

def load_roi_coordinates():
    """加载ROI坐标"""
    coord_file = 'cc200_roi_coords_182.npy'
    if os.path.exists(coord_file):
        coords = np.load(coord_file)
        print(f"✓ 从 {coord_file} 加载了 {len(coords)} 个ROI坐标")
        return coords
    else:
        print(f"⚠ 未找到坐标文件 {coord_file}")
        return None

def load_model_and_data():
    """Load trained model and sample data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataroot = 'data/ABIDE_pcp/cpac/filt_noglobal'
    name = 'ABIDE'
    dataset = ABIDEDataset(dataroot, name)
    dataset._data.y = dataset._data.y.squeeze()
    dataset._data.x[dataset._data.x == float('inf')] = 0
    
    model = NetworkLLM_GNN(
        indim=182, nclass=2, num_clusters=8,
        dropout=0.65, hidden=64, pool_temp=1.0,
        llm_hidden=96, llm_layers=1, llm_heads=2, llm_dropout=0.5,
        use_aux=True
    ).to(device)
    
    model_path = '/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/model_llm_hcg_ablation_randomsplit/full_best.pth'
    
    model_loaded = False
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Successfully loaded trained model")
            model_loaded = True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
    
    model.eval()
    sample = dataset[0].to(device)
    
    return model, sample, device, model_loaded

def extract_clustering_info(model, sample, device):
    """Extract clustering information from the model"""
    with torch.no_grad():
        x_gated, gate = model.llm_gater(sample.x, sample.batch)
        
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
        cluster_probs = s_matrix.cpu().numpy()
        
        adj = to_dense_adj(sample.edge_index, sample.batch, edge_attr=sample.edge_attr)
        if adj.dim() == 4:
            adj = adj.squeeze(0).squeeze(-1)
        elif adj.dim() == 3:
            adj = adj.squeeze(0)
        adj = adj.cpu().numpy()
        
    return cluster_assignment, cluster_probs, adj

def visualize_brain_slices_improved(roi_coords, cluster_assignment, adj, model_loaded, 
                                    save_path='hcgpool_brain_visualization_improved.png'):
    """改进的可视化，突出少数簇"""
    
    n_clusters = 8
    cluster_sizes = [np.sum(cluster_assignment == i) for i in range(n_clusters)]
    
    # 使用对比度更高的颜色
    colors_high_contrast = [
        '#FF0000',  # 鲜红 - Cluster 0
        '#00FF00',  # 鲜绿 - Cluster 1
        '#D3D3D3',  # 浅灰 - Cluster 2 (主导簇用灰色，降低视觉权重)
        '#FF00FF',  # 洋红 - Cluster 3
        '#FFFF00',  # 黄色 - Cluster 4
        '#00FFFF',  # 青色 - Cluster 5
        '#FFA500',  # 橙色 - Cluster 6
        '#800080',  # 紫色 - Cluster 7
    ]
    
    fig = plt.figure(figsize=(24, 14))
    
    title_color = 'green' if model_loaded else 'red'
    title_text = 'HCGPool Clustering - Brain Slice View (TRAINED MODEL - Enhanced Contrast)' if model_loaded else 'RANDOM INIT'
    fig.suptitle(title_text, fontsize=18, fontweight='bold', color=title_color)
    
    # 找出主导簇（最大的簇）
    dominant_cluster = np.argmax(cluster_sizes)
    print(f"\n主导簇: Cluster {dominant_cluster} ({cluster_sizes[dominant_cluster]} ROIs)")
    
    # ===== 子图1: 轴位切面 - 突出显示少数簇 =====
    ax1 = plt.subplot(2, 4, 1)
    ax1.set_title('Axial View (Top View, Z=0±15mm)', fontsize=13, fontweight='bold')
    
    z_range = (-15, 15)
    mask_axial = (roi_coords[:, 2] >= z_range[0]) & (roi_coords[:, 2] <= z_range[1])
    
    edge_threshold = np.percentile(adj[adj > 0], 95) if (adj > 0).any() else 0
    
    # 先绘制主导簇（用灰色，作为背景）
    mask_dominant = (cluster_assignment == dominant_cluster) & mask_axial
    if mask_dominant.sum() > 0:
        ax1.scatter(roi_coords[mask_dominant, 0], roi_coords[mask_dominant, 1], 
                   c=colors_high_contrast[dominant_cluster], s=60, alpha=0.3, 
                   edgecolors='gray', linewidth=0.5, zorder=1,
                   label=f'C{dominant_cluster} (dominant)')
    
    # 再绘制其他簇（用鲜艳颜色，覆盖在上面）
    for cluster_id in range(n_clusters):
        if cluster_id == dominant_cluster:
            continue
        mask = (cluster_assignment == cluster_id) & mask_axial
        if mask.sum() > 0:
            ax1.scatter(roi_coords[mask, 0], roi_coords[mask, 1], 
                       c=colors_high_contrast[cluster_id], s=150, alpha=0.9, 
                       edgecolors='black', linewidth=2, zorder=3,
                       label=f'C{cluster_id} (n={mask.sum()})')
    
    ax1.set_xlabel('X (Left ← �� Right)', fontsize=11)
    ax1.set_ylabel('Y (Posterior ← → Anterior)', fontsize=11)
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-120, 90)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
    ax1.set_aspect('equal')
    
    # ===== 子图2: 冠状切面 =====
    ax2 = plt.subplot(2, 4, 2)
    ax2.set_title('Coronal View (Front View, Y=-10±20mm)', fontsize=13, fontweight='bold')
    
    y_range = (-30, 10)
    mask_coronal = (roi_coords[:, 1] >= y_range[0]) & (roi_coords[:, 1] <= y_range[1])
    
    # 主导簇
    mask_dominant = (cluster_assignment == dominant_cluster) & mask_coronal
    if mask_dominant.sum() > 0:
        ax2.scatter(roi_coords[mask_dominant, 0], roi_coords[mask_dominant, 2], 
                   c=colors_high_contrast[dominant_cluster], s=60, alpha=0.3, 
                   edgecolors='gray', linewidth=0.5, zorder=1)
    
    # 其他簇
    for cluster_id in range(n_clusters):
        if cluster_id == dominant_cluster:
            continue
        mask = (cluster_assignment == cluster_id) & mask_coronal
        if mask.sum() > 0:
            ax2.scatter(roi_coords[mask, 0], roi_coords[mask, 2], 
                       c=colors_high_contrast[cluster_id], s=150, alpha=0.9, 
                       edgecolors='black', linewidth=2, zorder=3)
    
    ax2.set_xlabel('X (Left ← → Right)', fontsize=11)
    ax2.set_ylabel('Z (Inferior ← → Superior)', fontsize=11)
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-80, 80)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_aspect('equal')
    
    # ===== 子图3: 矢状切面 =====
    ax3 = plt.subplot(2, 4, 3)
    ax3.set_title('Sagittal View (Side View, X=±10mm)', fontsize=13, fontweight='bold')
    
    x_range = (-15, 15)
    mask_sagittal = (roi_coords[:, 0] >= x_range[0]) & (roi_coords[:, 0] <= x_range[1])
    
    # 主导簇
    mask_dominant = (cluster_assignment == dominant_cluster) & mask_sagittal
    if mask_dominant.sum() > 0:
        ax3.scatter(roi_coords[mask_dominant, 1], roi_coords[mask_dominant, 2], 
                   c=colors_high_contrast[dominant_cluster], s=60, alpha=0.3, 
                   edgecolors='gray', linewidth=0.5, zorder=1)
    
    # 其他���
    for cluster_id in range(n_clusters):
        if cluster_id == dominant_cluster:
            continue
        mask = (cluster_assignment == cluster_id) & mask_sagittal
        if mask.sum() > 0:
            ax3.scatter(roi_coords[mask, 1], roi_coords[mask, 2], 
                       c=colors_high_contrast[cluster_id], s=150, alpha=0.9, 
                       edgecolors='black', linewidth=2, zorder=3)
    
    ax3.set_xlabel('Y (Posterior ← → Anterior)', fontsize=11)
    ax3.set_ylabel('Z (Inferior ← → Superior)', fontsize=11)
    ax3.set_xlim(-120, 90)
    ax3.set_ylim(-80, 80)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_aspect('equal')
    
    # ===== 子图4: 只显示非主导簇（去除灰色噪音） =====
    ax4 = plt.subplot(2, 4, 4)
    ax4.set_title('Minor Clusters Only (Dominant Removed)', fontsize=13, fontweight='bold')
    
    for cluster_id in range(n_clusters):
        if cluster_id == dominant_cluster:
            continue
        mask = (cluster_assignment == cluster_id) & mask_axial
        if mask.sum() > 0:
            ax4.scatter(roi_coords[mask, 0], roi_coords[mask, 1], 
                       c=colors_high_contrast[cluster_id], s=200, alpha=0.9, 
                       edgecolors='black', linewidth=2,
                       label=f'C{cluster_id} (n={mask.sum()})')
    
    ax4.set_xlabel('X (Left ← → Right)', fontsize=11)
    ax4.set_ylabel('Y (Posterior ← → Anterior)', fontsize=11)
    ax4.set_xlim(-100, 100)
    ax4.set_ylim(-120, 90)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.legend(loc='best', fontsize=10, framealpha=0.9)
    ax4.set_aspect('equal')
    
    # ===== 子图5: 3D视图 =====
    ax5 = plt.subplot(2, 4, 5, projection='3d')
    ax5.set_title('3D View (All Clusters)', fontsize=13, fontweight='bold')
    
    # 主导簇
    mask_dominant = cluster_assignment == dominant_cluster
    ax5.scatter(roi_coords[mask_dominant, 0], roi_coords[mask_dominant, 1], roi_coords[mask_dominant, 2],
               c=colors_high_contrast[dominant_cluster], s=20, alpha=0.2, edgecolors='none')
    
    # 其他簇
    for cluster_id in range(n_clusters):
        if cluster_id == dominant_cluster:
            continue
        mask = cluster_assignment == cluster_id
        if mask.sum() > 0:
            ax5.scatter(roi_coords[mask, 0], roi_coords[mask, 1], roi_coords[mask, 2],
                       c=colors_high_contrast[cluster_id], s=100, alpha=0.9, 
                       edgecolors='black', linewidth=1)
    
    ax5.set_xlabel('X (L-R)')
    ax5.set_ylabel('Y (P-A)')
    ax5.set_zlabel('Z (I-S)')
    ax5.set_xlim(-100, 100)
    ax5.set_ylim(-120, 90)
    ax5.set_zlim(-80, 80)
    
    # ===== 子图6: 簇大小分布（对数刻度） =====
    ax6 = plt.subplot(2, 4, 6)
    ax6.set_title('Cluster Size Distribution (Log Scale)', fontsize=13, fontweight='bold')
    
    bars = ax6.bar(range(n_clusters), cluster_sizes, color=colors_high_contrast, edgecolor='black', linewidth=2)
    
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        if size > 0:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax6.set_xlabel('Cluster ID', fontsize=11)
    ax6.set_ylabel('Number of ROIs (log scale)', fontsize=11)
    ax6.set_xticks(range(n_clusters))
    ax6.set_yscale('symlog')  # 对数刻度，便于看到小数值
    ax6.grid(axis='y', alpha=0.3)
    
    # ===== 子图7: 左右半球分布 =====
    ax7 = plt.subplot(2, 4, 7)
    ax7.set_title('Hemisphere Distribution', fontsize=13, fontweight='bold')
    
    left_counts = []
    right_counts = []
    for cluster_id in range(n_clusters):
        mask = cluster_assignment == cluster_id
        left = np.sum((roi_coords[mask, 0] < 0))
        right = np.sum((roi_coords[mask, 0] >= 0))
        left_counts.append(left)
        right_counts.append(right)
    
    x = np.arange(n_clusters)
    width = 0.35
    ax7.bar(x - width/2, left_counts, width, label='Left Hemisphere', color='steelblue', edgecolor='black')
    ax7.bar(x + width/2, right_counts, width, label='Right Hemisphere', color='coral', edgecolor='black')
    
    ax7.set_xlabel('Cluster ID', fontsize=11)
    ax7.set_ylabel('Number of ROIs', fontsize=11)
    ax7.set_xticks(x)
    ax7.legend(fontsize=10)
    ax7.grid(axis='y', alpha=0.3)
    
    # ===== 子图8: 软分配热图（少数簇） =====
    ax8 = plt.subplot(2, 4, 8)
    ax8.set_title('Soft Assignment (Minor Clusters)', fontsize=13, fontweight='bold')
    
    # 只显示非主导簇的ROI
    non_dominant_mask = cluster_assignment != dominant_cluster
    non_dominant_indices = np.where(non_dominant_mask)[0]
    
    if len(non_dominant_indices) > 0:
        # 取最多30个ROI
        sample_indices = non_dominant_indices[:min(30, len(non_dominant_indices))]
        
        from matplotlib.colors import ListedColormap
        from p_ab2 import cluster_reg_loss
        
        # 重新提取软分配矩阵的相关部分
        # 这里简化显示聚类分配的确定性
        cluster_probs_sample = np.zeros((len(sample_indices), n_clusters))
        for i, idx in enumerate(sample_indices):
            cluster_id = cluster_assignment[idx]
            cluster_probs_sample[i, cluster_id] = 1.0  # 简化版本
        
        im = ax8.imshow(cluster_probs_sample.T, aspect='auto', cmap='hot', vmin=0, vmax=1)
        ax8.set_xlabel('ROI Index', fontsize=11)
        ax8.set_ylabel('Cluster ID', fontsize=11)
        ax8.set_yticks(range(n_clusters))
        
        cbar = plt.colorbar(im, ax=ax8)
        cbar.set_label('Assignment Probability', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 改进的可视化已保存到: {save_path}")
    
    # 打印统计
    print("\n" + "="*60)
    print("聚类统计（按大小排序）")
    print("="*60)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    for i in sorted_indices:
        mask = cluster_assignment == i
        n_rois = mask.sum()
        if n_rois == 0:
            continue
        n_left = np.sum((roi_coords[mask, 0] < 0))
        n_right = np.sum((roi_coords[mask, 0] >= 0))
        percentage = 100 * n_rois / len(cluster_assignment)
        marker = "⚠" if i == dominant_cluster else "✓"
        print(f"{marker} Cluster {i}: {n_rois:3d} ROIs ({percentage:5.1f}%) | Left: {n_left:3d}, Right: {n_right:3d}")
    print("="*60)

def main():
    set_seed(42)
    
    print("Step 1: 加载ROI坐标...")
    roi_coords = load_roi_coordinates()
    if roi_coords is None:
        return
    
    print("\nStep 2: 加载模型和数据...")
    model, sample, device, model_loaded = load_model_and_data()
    
    print("\nStep 3: 提取聚类信息...")
    cluster_assignment, cluster_probs, adj = extract_clustering_info(model, sample, device)
    
    print("\nStep 4: 生成改进的脑切面可视化...")
    visualize_brain_slices_improved(roi_coords, cluster_assignment, adj, model_loaded)
    
    print("\n✓ 完成！")

if __name__ == "__main__":
    main()

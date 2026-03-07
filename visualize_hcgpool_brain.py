import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

# 确保在正确的工作目录
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
        print("  请先运行 extract_roi_coords.py")
        return None

def load_model_and_data():
    """Load trained model and sample data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataroot = 'data/ABIDE_pcp/cpac/filt_noglobal'
    name = 'ABIDE'
    dataset = ABIDEDataset(dataroot, name)
    dataset._data.y = dataset._data.y.squeeze()
    dataset._data.x[dataset._data.x == float('inf')] = 0
    
    # Build model
    model = NetworkLLM_GNN(
        indim=182,
        nclass=2,
        num_clusters=8,
        dropout=0.65,
        hidden=64,
        pool_temp=1.0,
        llm_hidden=96,
        llm_layers=1,
        llm_heads=2,
        llm_dropout=0.5,
        use_aux=True
    ).to(device)
    
    # Load trained model
    model_path = '/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/model_llm_hcg_ablation_randomsplit/full_best.pth'
    
    model_loaded = False
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Successfully loaded trained model from: {model_path}")
            model_loaded = True
        except Exception as e:
            print(f"✗ Failed to load {model_path}: {e}")
    
    if not model_loaded:
        print("✗ No trained model found, using random initialization")
    
    model.eval()
    
    # Get a sample
    sample = dataset[0].to(device)
    
    return model, sample, device, model_loaded

def extract_clustering_info(model, sample, device):
    """Extract clustering information from the model"""
    with torch.no_grad():
        # Forward through LLM gater
        x_gated, gate = model.llm_gater(sample.x, sample.batch)
        
        # Forward through first GCN layers
        x = model.conv0(x_gated, sample.edge_index, sample.edge_attr)
        x = model.bn0(x)
        x = F.relu(x)
        
        x1 = x
        x = model.conv1(x, sample.edge_index, sample.edge_attr)
        x = model.bn1(x)
        x = F.relu(x)
        x = x + x1
        
        # Get clustering from HCGPool
        logits = model.pool1.cluster_proj(x) / max(model.pool1.temp, 1e-6)
        s_matrix = F.softmax(logits, dim=-1)
        
        # Get hard cluster assignment (argmax)
        cluster_assignment = s_matrix.argmax(dim=1).cpu().numpy()
        
        # Get soft probabilities
        cluster_probs = s_matrix.cpu().numpy()
        
        # Get adjacency matrix
        adj = to_dense_adj(sample.edge_index, sample.batch, edge_attr=sample.edge_attr)
        if adj.dim() == 4:
            adj = adj.squeeze(0).squeeze(-1)
        elif adj.dim() == 3:
            adj = adj.squeeze(0)
        adj = adj.cpu().numpy()
        
    return cluster_assignment, cluster_probs, adj

def visualize_brain_slices(roi_coords, cluster_assignment, adj, model_loaded, 
                           save_path='hcgpool_brain_visualization.png'):
    """可视化脑切面图"""
    
    n_clusters = len(np.unique(cluster_assignment))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    fig = plt.figure(figsize=(20, 12))
    
    if model_loaded:
        fig.suptitle('HCGPool Clustering - Brain Slice View (TRAINED MODEL)', 
                     fontsize=16, fontweight='bold', color='green')
    else:
        fig.suptitle('HCGPool Clustering - Brain Slice View (RANDOM INIT)', 
                     fontsize=16, fontweight='bold', color='red')
    
    # ===== 子图1: 轴位切面 (Axial: 从上往下看，Z=常数) =====
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Axial View (Top View, Z=0±15mm)', fontsize=12, fontweight='bold')
    
    # 选择Z坐标在[-15, 15]范围内的ROI
    z_range = (-15, 15)
    mask_axial = (roi_coords[:, 2] >= z_range[0]) & (roi_coords[:, 2] <= z_range[1])
    
    # 绘制连接（边）
    edge_threshold = np.percentile(adj[adj > 0], 95) if (adj > 0).any() else 0
    for i in range(len(roi_coords)):
        if not mask_axial[i]:
            continue
        for j in range(i+1, len(roi_coords)):
            if not mask_axial[j]:
                continue
            if adj[i, j] > edge_threshold:
                ax1.plot([roi_coords[i, 0], roi_coords[j, 0]], 
                        [roi_coords[i, 1], roi_coords[j, 1]], 
                        'gray', alpha=0.15, linewidth=0.3, zorder=1)
    
    # 绘制ROI节点
    for cluster_id in range(n_clusters):
        mask = (cluster_assignment == cluster_id) & mask_axial
        if mask.sum() > 0:
            ax1.scatter(roi_coords[mask, 0], roi_coords[mask, 1], 
                       c=[colors[cluster_id]], s=100, alpha=0.8, 
                       edgecolors='black', linewidth=1, zorder=2,
                       label=f'C{cluster_id}')
    
    ax1.set_xlabel('X (Left ← → Right)', fontsize=10)
    ax1.set_ylabel('Y (Posterior ← → Anterior)', fontsize=10)
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-120, 90)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_aspect('equal')
    
    # ===== 子图2: 冠状切面 (Coronal: 从前往后看，Y=常数) =====
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Coronal View (Front View, Y=-10±20mm)', fontsize=12, fontweight='bold')
    
    y_range = (-30, 10)
    mask_coronal = (roi_coords[:, 1] >= y_range[0]) & (roi_coords[:, 1] <= y_range[1])
    
    # 绘制连接
    for i in range(len(roi_coords)):
        if not mask_coronal[i]:
            continue
        for j in range(i+1, len(roi_coords)):
            if not mask_coronal[j]:
                continue
            if adj[i, j] > edge_threshold:
                ax2.plot([roi_coords[i, 0], roi_coords[j, 0]], 
                        [roi_coords[i, 2], roi_coords[j, 2]], 
                        'gray', alpha=0.15, linewidth=0.3, zorder=1)
    
    # 绘制ROI节点
    for cluster_id in range(n_clusters):
        mask = (cluster_assignment == cluster_id) & mask_coronal
        if mask.sum() > 0:
            ax2.scatter(roi_coords[mask, 0], roi_coords[mask, 2], 
                       c=[colors[cluster_id]], s=100, alpha=0.8, 
                       edgecolors='black', linewidth=1, zorder=2)
    
    ax2.set_xlabel('X (Left ← → Right)', fontsize=10)
    ax2.set_ylabel('Z (Inferior ← → Superior)', fontsize=10)
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-80, 80)
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.set_aspect('equal')
    
    # ===== 子图3: 矢状切面 (Sagittal: 从侧面看，X=常数) =====
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Sagittal View (Side View, X=±5mm)', fontsize=12, fontweight='bold')
    
    # 选择接近中线的ROI
    x_range = (-10, 10)
    mask_sagittal = (roi_coords[:, 0] >= x_range[0]) & (roi_coords[:, 0] <= x_range[1])
    
    # 绘制连接
    for i in range(len(roi_coords)):
        if not mask_sagittal[i]:
            continue
        for j in range(i+1, len(roi_coords)):
            if not mask_sagittal[j]:
                continue
            if adj[i, j] > edge_threshold:
                ax3.plot([roi_coords[i, 1], roi_coords[j, 1]], 
                        [roi_coords[i, 2], roi_coords[j, 2]], 
                        'gray', alpha=0.15, linewidth=0.3, zorder=1)
    
    # 绘制ROI节点
    for cluster_id in range(n_clusters):
        mask = (cluster_assignment == cluster_id) & mask_sagittal
        if mask.sum() > 0:
            ax3.scatter(roi_coords[mask, 1], roi_coords[mask, 2], 
                       c=[colors[cluster_id]], s=100, alpha=0.8, 
                       edgecolors='black', linewidth=1, zorder=2)
    
    ax3.set_xlabel('Y (Posterior ← → Anterior)', fontsize=10)
    ax3.set_ylabel('Z (Inferior ← → Superior)', fontsize=10)
    ax3.set_xlim(-120, 90)
    ax3.set_ylim(-80, 80)
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax3.set_aspect('equal')
    
    # ===== 子图4: 3D投影视图 =====
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    ax4.set_title('3D View', fontsize=12, fontweight='bold')
    
    for cluster_id in range(n_clusters):
        mask = cluster_assignment == cluster_id
        if mask.sum() > 0:
            ax4.scatter(roi_coords[mask, 0], roi_coords[mask, 1], roi_coords[mask, 2],
                       c=[colors[cluster_id]], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax4.set_xlabel('X (L-R)')
    ax4.set_ylabel('Y (P-A)')
    ax4.set_zlabel('Z (I-S)')
    ax4.set_xlim(-100, 100)
    ax4.set_ylim(-120, 90)
    ax4.set_zlim(-80, 80)
    
    # ===== 子图5: 簇大小分布 =====
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    
    cluster_sizes = [np.sum(cluster_assignment == i) for i in range(n_clusters)]
    bars = ax5.bar(range(n_clusters), cluster_sizes, color=colors, edgecolor='black')
    
    for bar, size in zip(bars, cluster_sizes):
        if size > 0:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5.set_xlabel('Cluster ID')
    ax5.set_ylabel('Number of ROIs')
    ax5.set_xticks(range(n_clusters))
    ax5.grid(axis='y', alpha=0.3)
    
    # ===== 子图6: 左右半球分布 =====
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Hemisphere Distribution per Cluster', fontsize=12, fontweight='bold')
    
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
    ax6.bar(x - width/2, left_counts, width, label='Left Hemisphere', color='steelblue', edgecolor='black')
    ax6.bar(x + width/2, right_counts, width, label='Right Hemisphere', color='coral', edgecolor='black')
    
    ax6.set_xlabel('Cluster ID')
    ax6.set_ylabel('Number of ROIs')
    ax6.set_xticks(x)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 脑切面可视化已保存到: {save_path}")
    
    # 打印统计
    print("\n" + "="*60)
    print("聚类统计")
    print("="*60)
    for i in range(n_clusters):
        mask = cluster_assignment == i
        n_rois = mask.sum()
        n_left = np.sum((roi_coords[mask, 0] < 0))
        n_right = np.sum((roi_coords[mask, 0] >= 0))
        print(f"Cluster {i}: {n_rois:3d} ROIs (Left: {n_left:3d}, Right: {n_right:3d})")
    print("="*60)

def main():
    set_seed(42)
    
    # 1. 加载ROI坐标
    print("Step 1: 加载ROI坐标...")
    roi_coords = load_roi_coordinates()
    if roi_coords is None:
        print("请先运行 extract_roi_coords.py 提取坐标")
        return
    
    # 2. 加载模型和数据
    print("\nStep 2: 加载模型和数据...")
    model, sample, device, model_loaded = load_model_and_data()
    
    # 3. 提取聚类信息
    print("\nStep 3: 提取聚类信息...")
    cluster_assignment, cluster_probs, adj = extract_clustering_info(model, sample, device)
    
    # 4. 可视化
    print("\nStep 4: 生成脑切面可视化...")
    visualize_brain_slices(roi_coords, cluster_assignment, adj, model_loaded)
    
    print("\n✓ 完成！")

if __name__ == "__main__":
    main()

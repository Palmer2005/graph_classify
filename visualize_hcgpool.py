import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
from imports.ABIDEDataset import ABIDEDataset
from p_ab2 import NetworkLLM_GNN, set_seed

def load_model_and_data():
    """Load trained model and sample data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataroot = '/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/data/ABIDE_pcp/cpac/filt_noglobal'
    name = 'ABIDE'
    dataset = ABIDEDataset(dataroot, name)
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0
    
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
    
    # Load model weights if exists
    model_path = '/share/home/u24666550/graph_classify/gnn/BrainGNN_Pytorch-wmrc/model_llm_hcg_ablation_randomsplit/full_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found, using random initialization")
    
    model.eval()
    
    # Get a sample
    sample = dataset[0].to(device)
    
    return model, sample, device

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
        s_matrix = F.softmax(logits, dim=-1)  # [N, 8]
        
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

def create_circular_layout(n_nodes):
    """Create circular layout for ROI positions"""
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack([x, y])

def visualize_clustering(cluster_assignment, cluster_probs, adj, save_path='hcgpool_visualization.png'):
    """Visualize the clustering results"""
    n_rois = len(cluster_assignment)
    n_clusters = cluster_probs.shape[1]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Color map for clusters
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    # Layout for ROIs (circular)
    pos = create_circular_layout(n_rois)
    
    # ============ Subplot 1: Original Brain Network ============
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Original Brain Network (182 ROIs)', fontsize=14, fontweight='bold')
    
    # Draw edges (only top 5% strongest connections for clarity)
    edge_threshold = np.percentile(adj[adj > 0], 95)
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            if adj[i, j] > edge_threshold:
                ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                        'gray', alpha=0.3, linewidth=0.5)
    
    # Draw nodes
    ax1.scatter(pos[:, 0], pos[:, 1], c='lightblue', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add some ROI labels
    for i in [0, 45, 90, 135, 181]:
        ax1.annotate(f'ROI {i}', xy=(pos[i, 0], pos[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.axis('off')
    
    # ============ Subplot 2: Clustered Network (Hard Assignment) ============
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Hard Clustering (8 Clusters)', fontsize=14, fontweight='bold')
    
    # Draw edges
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            if adj[i, j] > edge_threshold:
                ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                        'gray', alpha=0.2, linewidth=0.5)
    
    # Draw nodes colored by cluster
    for cluster_id in range(n_clusters):
        mask = cluster_assignment == cluster_id
        ax2.scatter(pos[mask, 0], pos[mask, 1], 
                   c=[colors[cluster_id]], s=50, alpha=0.9, 
                   edgecolors='black', linewidth=0.8,
                   label=f'Cluster {cluster_id}')
    
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
    ax2.axis('off')
    
    # ============ Subplot 3: Cluster Size Distribution ============
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    
    cluster_sizes = [np.sum(cluster_assignment == i) for i in range(n_clusters)]
    bars = ax3.bar(range(n_clusters), cluster_sizes, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Cluster ID', fontsize=12)
    ax3.set_ylabel('Number of ROIs', fontsize=12)
    ax3.set_xticks(range(n_clusters))
    ax3.grid(axis='y', alpha=0.3)
    
    # ============ Subplot 4: Soft Assignment Heatmap ============
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Soft Clustering Probabilities (Sample ROIs)', fontsize=14, fontweight='bold')
    
    # Show a subset of ROIs for clarity
    sample_rois = np.linspace(0, n_rois-1, 30, dtype=int)
    soft_probs_subset = cluster_probs[sample_rois, :]
    
    im = ax4.imshow(soft_probs_subset.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax4.set_xlabel('ROI Index', fontsize=12)
    ax4.set_ylabel('Cluster ID', fontsize=12)
    ax4.set_xticks(range(0, len(sample_rois), 5))
    ax4.set_xticklabels([str(sample_rois[i]) for i in range(0, len(sample_rois), 5)], rotation=45)
    ax4.set_yticks(range(n_clusters))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Assignment Probability', fontsize=10)
    
    # ============ Subplot 5: Inter-cluster Connectivity ============
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Inter-Cluster Connectivity', fontsize=14, fontweight='bold')
    
    # Compute connectivity between clusters
    inter_cluster_conn = np.zeros((n_clusters, n_clusters))
    for i in range(n_rois):
        for j in range(n_rois):
            if adj[i, j] > 0:
                ci, cj = cluster_assignment[i], cluster_assignment[j]
                inter_cluster_conn[ci, cj] += adj[i, j]
    
    # Normalize by cluster sizes
    for i in range(n_clusters):
        for j in range(n_clusters):
            size_i = np.sum(cluster_assignment == i)
            size_j = np.sum(cluster_assignment == j)
            if size_i > 0 and size_j > 0:
                inter_cluster_conn[i, j] /= (size_i * size_j)
    
    im2 = ax5.imshow(inter_cluster_conn, cmap='Blues', aspect='auto')
    ax5.set_xlabel('Cluster ID', fontsize=12)
    ax5.set_ylabel('Cluster ID', fontsize=12)
    ax5.set_xticks(range(n_clusters))
    ax5.set_yticks(range(n_clusters))
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax5)
    cbar2.set_label('Avg Connection Strength', fontsize=10)
    
    # Add values in cells
    for i in range(n_clusters):
        for j in range(n_clusters):
            text = ax5.text(j, i, f'{inter_cluster_conn[i, j]:.2f}',
                          ha="center", va="center", color="black" if inter_cluster_conn[i, j] < 0.5 else "white",
                          fontsize=8)
    
    # ============ Subplot 6: Cluster Statistics ============
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Cluster Assignment Confidence', fontsize=14, fontweight='bold')
    
    # Compute assignment confidence (max probability for each ROI)
    confidence = cluster_probs.max(axis=1)
    
    # Histogram
    ax6.hist(confidence, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax6.axvline(confidence.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidence.mean():.3f}')
    ax6.axvline(np.median(confidence), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(confidence):.3f}')
    
    ax6.set_xlabel('Max Cluster Probability', fontsize=12)
    ax6.set_ylabel('Number of ROIs', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add text with statistics
    stats_text = f"Min: {confidence.min():.3f}\nMax: {confidence.max():.3f}\nStd: {confidence.std():.3f}"
    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("HCGPool Clustering Statistics")
    print("="*60)
    print(f"Total ROIs: {n_rois}")
    print(f"Number of clusters: {n_clusters}")
    print("\nCluster sizes:")
    for i in range(n_clusters):
        count = np.sum(cluster_assignment == i)
        percentage = 100 * count / n_rois
        print(f"  Cluster {i}: {count} ROIs ({percentage:.1f}%)")
    
    print(f"\nAssignment confidence:")
    print(f"  Mean: {confidence.mean():.3f}")
    print(f"  Median: {np.median(confidence):.3f}")
    print(f"  Min: {confidence.min():.3f}")
    print(f"  Max: {confidence.max():.3f}")
    
    print(f"\nMost uncertain ROIs (lowest confidence):")
    uncertain_rois = np.argsort(confidence)[:10]
    for idx in uncertain_rois:
        print(f"  ROI {idx}: confidence={confidence[idx]:.3f}, assigned to cluster {cluster_assignment[idx]}")
    
    print("="*60 + "\n")

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Load model and data
    print("Loading model and data...")
    model, sample, device = load_model_and_data()
    
    # Extract clustering information
    print("Extracting clustering information...")
    cluster_assignment, cluster_probs, adj = extract_clustering_info(model, sample, device)
    
    # Create visualization
    print("Creating visualization...")
    visualize_clustering(cluster_assignment, cluster_probs, adj)
    
    print("\nDone! Check 'hcgpool_visualization.png' for the results.")

if __name__ == "__main__":
    main()
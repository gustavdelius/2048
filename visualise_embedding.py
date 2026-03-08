import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from agent import DeepEmbeddingDQN

def visualize_embeddings(model_path="best_model.pth", save_path="embedding_viz.png"):
    # Load model
    device = torch.device("cpu")
    model = DeepEmbeddingDQN(board_size=(3, 3), num_actions=4)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Extract embedding weights
    embeddings = model.embedding.weight.detach().cpu().numpy()
    
    # Tile values for labels corresponding to indices 0..8
    labels = ["0 (Empty)", "2", "4", "8", "16", "32", "64", "128", "256+"]
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Heatmap of the raw embedding values
    sns.heatmap(embeddings, annot=True, cmap="coolwarm", yticklabels=labels, ax=ax1, fmt=".3f")
    ax1.set_title("Embedding Weights Heatmap")
    ax1.set_xlabel("Embedding Dimension")
    ax1.set_ylabel("Tile Value")
    
    # 2. PCA 2D scatter plot
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Use a color palette to show progression
    colors = sns.color_palette("viridis", n_colors=len(labels))
    
    for i, (x, y) in enumerate(embeddings_2d):
        ax2.scatter(x, y, color=colors[i], s=150, zorder=5)
        ax2.annotate(labels[i], (x, y), xytext=(8, 8), textcoords='offset points', 
                     fontsize=12, fontweight='bold')
                     
    ax2.set_title(f"PCA of Embeddings (2D Projection)\nVariance Explained: {sum(pca.explained_variance_ratio_):.2%}")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Draw arrows to show sequence (optional progression)
    for i in range(1, len(embeddings_2d)):
        ax2.annotate("", xy=embeddings_2d[i], xytext=embeddings_2d[i-1],
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'best_model.pth'
    visualize_embeddings(model_file)

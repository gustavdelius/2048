import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from agent import DeepEmbeddingDQN
import sys

def analyze_embeddings(model_path):
    device = torch.device("cpu")
    model = DeepEmbeddingDQN(board_size=(3, 3), num_actions=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    embeddings = model.embedding.weight.detach().cpu().numpy()
    labels = ["0 (Empty)", "2", "4", "8", "16", "32", "64", "128", "256+"]
    
    print("=== EMBEDDING MAGNITUDES ===")
    magnitudes = np.linalg.norm(embeddings, axis=1)
    for i, mag in enumerate(magnitudes):
        print(f"{labels[i]:>10}: {mag:.4f}")
        
    print("\n=== COSINE SIMILARITY BETWEEN CONSECUTIVE TILES ===")
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0, 0]
        print(f"{labels[i]:>10} -> {labels[i+1]:<10}: {sim:.4f}")
        
    print("\n=== DIFFERENCE VECTORS (Direction of doubling) ===")
    diffs = []
    for i in range(1, len(embeddings) - 1):
        diffs.append(embeddings[i+1] - embeddings[i])
        
    # Check if difference vectors are similar (is there a consistent 'increase' direction?)
    print("Cosine similarities between (Tile_{i+1} - Tile_i) and (Tile_{i+2} - Tile_{i+1}):")
    for i in range(len(diffs) - 1):
        sim = cosine_similarity([diffs[i]], [diffs[i+1]])[0, 0]
        print(f"({labels[i+1]}->{labels[i+2]}) vs ({labels[i+2]}->{labels[i+3]}): {sim:.4f}")

    print("\n=== RAW VALUES ===")
    np.set_printoptions(precision=4, suppress=True)
    for i, v in enumerate(embeddings):
        print(f"{labels[i]:>10}: {v}")

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'best_model.pth'
    analyze_embeddings(model_file)

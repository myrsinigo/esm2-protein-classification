import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# def plot_2d(X_2d, y, title, output_path):
#     plt.figure(figsize=(7, 6))

#     for label, name in [(0, "Non-membrane"), (1, "Membrane")]:
#         mask = y == label
#         plt.scatter(
#             X_2d[mask, 0],
#             X_2d[mask, 1],
#             alpha=0.7,
#             label=name,
#             s=30,
#         )

#     plt.title(title)
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()


def plot_2d(X_2d, y, title, output_path):
    plt.figure(figsize=(8, 6))

    colors = {
        0: "#1f77b4",
        1: "#ff7f0e"
    }

    labels = {
        0: "Non-membrane",
        1: "Membrane"
    }

    for label in [0, 1]:
        mask = y == label

        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=colors[label],
            label=labels[label],
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.3,
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)

    plt.legend(frameon=True)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    Path("results").mkdir(exist_ok=True)

    X = np.load("results/esm2_embeddings.npy")
    y = np.load("results/labels.npy")

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)

    plot_2d(
        X_pca,
        y,
        "PCA of ESM-2 Protein Embeddings",
        "results/pca_embeddings.png",
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    X_tsne = tsne.fit_transform(X_scaled)

    plot_2d(
        X_tsne,
        y,
        "t-SNE of ESM-2 Protein Embeddings",
        "results/tsne_embeddings.png",
    )

    print("Saved plots:")
    print("results/pca_embeddings.png")
    print("results/tsne_embeddings.png")


if __name__ == "__main__":
    main()
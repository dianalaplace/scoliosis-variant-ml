#!/usr/bin/env python3
"""
Step 05: UMAP dimensionality reduction + KMeans clustering.

Performs clustering on gene-level features:
1. StandardScaler normalization
2. UMAP (n_neighbors=10, min_dist=0.1)
3. KMeans k=2..6, best k by silhouette score

Output:
  results/gene_clustering.csv
  results/figures/umap_genes.png (two panels: by subtype & by cluster)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color scheme
COLORS = {
    "ECM": "#E74C3C",
    "VERTEBRAL_DEV": "#3498DB",
    "NEUROMUSCULAR": "#2ECC71",
    "CILIARY": "#9B59B6",
    "CONTROL": "#95A5A6",
}

# Features used for clustering
FEATURE_COLS = [
    "n_variants", "n_pathogenic", "pct_rare", "pct_lof",
    "pct_missense", "mean_gnomad_af_global_log",
    "mean_gnomad_af_nfe_log", "mean_review_stars",
]


def main():
    print("=" * 60)
    print("Step 05: UMAP + KMeans Clustering")
    print("=" * 60)

    # Load gene-level features
    df = pd.read_csv(DATA_DIR / "gene_level_features.csv")
    print(f"\nLoaded {len(df)} genes")
    print(f"Subtypes: {df['subtype'].value_counts().to_dict()}")

    # Prepare feature matrix
    X = df[FEATURE_COLS].fillna(0).values
    print(f"Feature matrix shape: {X.shape}")

    # 1. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. UMAP
    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=min(10, len(X) - 1),
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric="euclidean",
    )
    embedding = reducer.fit_transform(X_scaled)
    df["umap_1"] = embedding[:, 0]
    df["umap_2"] = embedding[:, 1]
    print(f"UMAP embedding shape: {embedding.shape}")

    # 3. KMeans — only on scoliosis genes
    scoliosis_mask = df["is_scoliosis"] == 1
    X_scol = X_scaled[scoliosis_mask]

    if len(X_scol) >= 4:
        print("\nFinding optimal k (KMeans on scoliosis genes)...")
        best_k = 2
        best_score = -1
        scores = {}

        for k in range(2, min(7, len(X_scol))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scol)
            score = silhouette_score(X_scol, labels)
            scores[k] = score
            print(f"  k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

        print(f"\nBest k={best_k} (silhouette={best_score:.3f})")

        # Final clustering with best k
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        scol_labels = km_final.fit_predict(X_scol)
        df.loc[scoliosis_mask, "kmeans_cluster"] = scol_labels
        df.loc[~scoliosis_mask, "kmeans_cluster"] = -1
        df["kmeans_cluster"] = df["kmeans_cluster"].astype(int)
    else:
        print("\nToo few scoliosis genes for KMeans, skipping clustering")
        df["kmeans_cluster"] = -1

    # Save results
    output_path = RESULTS_DIR / "gene_clustering.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved clustering results to {output_path}")

    # 4. Plot UMAP figures
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Fig A: UMAP colored by subtype
    ax = axes[0]
    # Plot controls first (gray background)
    ctrl = df[df["subtype"] == "CONTROL"]
    if len(ctrl) > 0:
        ax.scatter(ctrl["umap_1"], ctrl["umap_2"],
                   c=COLORS["CONTROL"], alpha=0.3, s=30,
                   label="Control", zorder=1)

    # Plot scoliosis genes by subtype
    for subtype in ["ECM", "VERTEBRAL_DEV", "NEUROMUSCULAR", "CILIARY"]:
        subset = df[df["subtype"] == subtype]
        if len(subset) == 0:
            continue
        ax.scatter(subset["umap_1"], subset["umap_2"],
                   c=COLORS[subtype], s=80, edgecolors="black",
                   linewidth=0.5, label=subtype, zorder=2)
        # Annotate gene names
        for _, row in subset.iterrows():
            ax.annotate(row["gene"], (row["umap_1"], row["umap_2"]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("A. Gene UMAP — by Pathway Subtype", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")

    # Fig B: UMAP colored by KMeans cluster
    ax = axes[1]
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, max(df["kmeans_cluster"].max() + 1, 2)))

    # Controls
    if len(ctrl) > 0:
        ax.scatter(ctrl["umap_1"], ctrl["umap_2"],
                   c=COLORS["CONTROL"], alpha=0.3, s=30,
                   label="Control", zorder=1)

    # Scoliosis genes by cluster
    scol_df = df[df["is_scoliosis"] == 1]
    for cluster_id in sorted(scol_df["kmeans_cluster"].unique()):
        if cluster_id < 0:
            continue
        subset = scol_df[scol_df["kmeans_cluster"] == cluster_id]
        ax.scatter(subset["umap_1"], subset["umap_2"],
                   c=[cluster_colors[cluster_id]], s=80,
                   edgecolors="black", linewidth=0.5,
                   label=f"Cluster {cluster_id}", zorder=2)
        for _, row in subset.iterrows():
            ax.annotate(row["gene"], (row["umap_1"], row["umap_2"]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("B. Gene UMAP — by KMeans Cluster", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig_path = FIG_DIR / "umap_genes.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP figure to {fig_path}")

    # Print cluster composition
    if df["kmeans_cluster"].max() >= 0:
        print("\nCluster composition:")
        for cluster_id in sorted(scol_df["kmeans_cluster"].unique()):
            if cluster_id < 0:
                continue
            genes_in_cluster = scol_df[scol_df["kmeans_cluster"] == cluster_id]
            subtypes = genes_in_cluster["subtype"].value_counts().to_dict()
            gene_list = genes_in_cluster["gene"].tolist()
            print(f"  Cluster {cluster_id}: {subtypes} → {gene_list}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 04: Gene-level feature aggregation.

Aggregates variant-level features to gene-level for UMAP clustering.
Gene-level analysis is more biologically interpretable.

Output: data/gene_level_features.csv
        data/control_gene_features_all.csv (for permutation test)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"


def load_gene_subtypes():
    """Load gene → subtype mapping from config."""
    with open(DATA_DIR / "genes_config.json") as f:
        config = json.load(f)
    gene_subtype = {}
    for subtype, info in config["subtypes"].items():
        for gene in info["genes"]:
            gene_subtype[gene] = subtype
    return gene_subtype


def aggregate_gene_features(df, gene_subtype_map=None, is_scoliosis=True):
    """Aggregate variant-level features to gene level."""
    if len(df) == 0:
        return pd.DataFrame()

    gene_groups = df.groupby("gene")
    records = []

    for gene, group in gene_groups:
        rec = {
            "gene": gene,
            "n_variants": len(group),
            "n_pathogenic": int(group["is_pathogenic"].sum()) if "is_pathogenic" in group.columns else 0,
            "pct_rare": float(group["is_rare"].mean() * 100) if "is_rare" in group.columns else 100.0,
            "pct_lof": float(group["is_lof"].mean() * 100) if "is_lof" in group.columns else 0.0,
            "mean_review_stars": float(group["review_stars"].mean()) if "review_stars" in group.columns else 0.0,
        }

        # Missense percentage
        if "cons_missense" in group.columns:
            rec["pct_missense"] = float(group["cons_missense"].mean() * 100)
        elif "consequence_cat" in group.columns:
            rec["pct_missense"] = float(
                (group["consequence_cat"] == "missense").mean() * 100
            )
        else:
            rec["pct_missense"] = 0.0

        # gnomAD AF means
        for col in ["gnomad_af_global_log", "gnomad_af_nfe_log"]:
            if col in group.columns:
                rec[f"mean_{col}"] = float(group[col].mean())
            else:
                rec[f"mean_{col}"] = np.log10(1e-7)

        # Subtype
        if is_scoliosis and gene_subtype_map:
            rec["subtype"] = gene_subtype_map.get(gene, "UNKNOWN")
            rec["is_scoliosis"] = 1
        else:
            rec["subtype"] = "CONTROL"
            rec["is_scoliosis"] = 0

        records.append(rec)

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Step 04: Gene-Level Feature Aggregation")
    print("=" * 60)

    gene_subtype_map = load_gene_subtypes()

    # 1. Scoliosis genes
    annot_path = DATA_DIR / "variants_annotated.csv"
    if not annot_path.exists():
        print(f"Error: {annot_path} not found. Run 03_feature_engineering.py first.")
        return

    df_scol = pd.read_csv(annot_path)
    print(f"\nScoliosis variants loaded: {len(df_scol)}")

    df_genes_scol = aggregate_gene_features(df_scol, gene_subtype_map, is_scoliosis=True)
    print(f"Scoliosis genes with data: {len(df_genes_scol)}")

    # 2. Control sets — aggregate each, then compute median per gene across sets
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    print(f"\nProcessing {len(control_files)} control sets...")

    all_control_genes = []
    for cf in control_files:
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0:
            continue
        df_ctrl_genes = aggregate_gene_features(df_ctrl, is_scoliosis=False)
        set_idx = int(cf.stem.split("_")[1])
        df_ctrl_genes["control_set"] = set_idx
        all_control_genes.append(df_ctrl_genes)

    if all_control_genes:
        df_all_controls = pd.concat(all_control_genes, ignore_index=True)
        print(f"Total control gene entries across all sets: {len(df_all_controls)}")

        # For UMAP: take median features per unique control gene
        numeric_cols = [
            "n_variants", "n_pathogenic", "pct_rare", "pct_lof",
            "pct_missense", "mean_gnomad_af_global_log",
            "mean_gnomad_af_nfe_log", "mean_review_stars",
        ]
        df_control_summary = (
            df_all_controls
            .groupby("gene")[numeric_cols]
            .median()
            .reset_index()
        )
        df_control_summary["subtype"] = "CONTROL"
        df_control_summary["is_scoliosis"] = 0
    else:
        df_control_summary = pd.DataFrame()

    # 3. Combine scoliosis + control genes
    df_combined = pd.concat(
        [df_genes_scol, df_control_summary], ignore_index=True
    )

    output_path = DATA_DIR / "gene_level_features.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_combined)} genes to {output_path}")
    print(f"  Scoliosis: {len(df_genes_scol)}, Control: {len(df_control_summary)}")

    # Summary
    print("\nGenes per subtype:")
    print(df_combined["subtype"].value_counts().to_string())

    # Save raw control data for permutation test in step 06
    if all_control_genes:
        ctrl_output = DATA_DIR / "control_gene_features_all.csv"
        df_all_controls.to_csv(ctrl_output, index=False)
        print(f"\nSaved all control gene features to {ctrl_output.name}")


if __name__ == "__main__":
    main()

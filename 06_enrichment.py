#!/usr/bin/env python3
"""
Step 06: Enrichment analysis.

Part A: Fisher's exact test (rare pathogenic variant enrichment per subtype)
Part B: Permutation test (key result — subtype AF vs random gene sets)
Part C: LoF fraction comparison across subtypes

Output:
  results/enrichment_results.csv
  results/figures/permutation_test.png
  results/figures/lof_comparison.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ECM": "#E74C3C",
    "VERTEBRAL_DEV": "#3498DB",
    "NEUROMUSCULAR": "#2ECC71",
    "CILIARY": "#9B59B6",
    "CONTROL": "#95A5A6",
}

SUBTYPES = ["ECM", "VERTEBRAL_DEV", "NEUROMUSCULAR", "CILIARY"]


def load_gene_subtypes():
    with open(DATA_DIR / "genes_config.json") as f:
        config = json.load(f)
    gene_subtype = {}
    for subtype, info in config["subtypes"].items():
        for gene in info["genes"]:
            gene_subtype[gene] = subtype
    return gene_subtype


def part_a_fisher(df_annotated):
    """Fisher's exact test: rare variant enrichment per subtype."""
    print("\n--- Part A: Fisher's Exact Test ---")
    results = []

    for subtype in SUBTYPES:
        subtype_mask = df_annotated["subtype"] == subtype
        other_mask = (df_annotated["subtype"] != subtype) & (df_annotated["subtype"] != "CONTROL")

        sub_df = df_annotated[subtype_mask]
        oth_df = df_annotated[other_mask]

        if len(sub_df) == 0 or len(oth_df) == 0:
            print(f"  {subtype}: insufficient data, skipping")
            continue

        rare_sub = int(sub_df["is_rare"].sum())
        common_sub = len(sub_df) - rare_sub
        rare_oth = int(oth_df["is_rare"].sum())
        common_oth = len(oth_df) - rare_oth

        table = [[rare_sub, common_sub], [rare_oth, common_oth]]
        odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

        # Bonferroni correction (n=4 tests)
        p_corrected = min(p_value * 4, 1.0)

        # Confidence interval for OR (Woolf's method)
        if all(x > 0 for row in table for x in row):
            log_or = np.log(odds_ratio)
            se = np.sqrt(sum(1.0 / x for row in table for x in row))
            ci_lower = np.exp(log_or - 1.96 * se)
            ci_upper = np.exp(log_or + 1.96 * se)
        else:
            ci_lower, ci_upper = 0, float("inf")

        results.append({
            "test": "fisher_rare_enrichment",
            "subtype": subtype,
            "odds_ratio": odds_ratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "p_corrected": p_corrected,
            "n_rare_subtype": rare_sub,
            "n_total_subtype": len(sub_df),
            "n_rare_others": rare_oth,
            "n_total_others": len(oth_df),
            "significant": p_corrected < 0.05,
        })

        sig = "*" if p_corrected < 0.05 else ""
        print(f"  {subtype}: OR={odds_ratio:.2f} [{ci_lower:.2f}-{ci_upper:.2f}], "
              f"p={p_value:.4f}, p_corr={p_corrected:.4f} {sig}")

    return results


def part_b_permutation(df_annotated):
    """Permutation test: compare subtype AF to random gene sets."""
    print("\n--- Part B: Permutation Test ---")

    # Calculate mean gnomAD AF (NFE) for each scoliosis subtype
    subtype_af = {}
    for subtype in SUBTYPES:
        sub_df = df_annotated[df_annotated["subtype"] == subtype]
        if len(sub_df) > 0 and "gnomad_af_nfe_log" in sub_df.columns:
            subtype_af[subtype] = sub_df["gnomad_af_nfe_log"].mean()
        else:
            subtype_af[subtype] = None
        print(f"  {subtype}: mean AF_NFE_log = {subtype_af[subtype]}")

    # Calculate mean AF for each of 100 random control sets
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    control_af_means = []

    for cf in control_files:
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0 or "gnomad_af_nfe_log" not in df_ctrl.columns:
            continue
        mean_af = df_ctrl["gnomad_af_nfe_log"].mean()
        control_af_means.append(mean_af)

    control_af_means = np.array(control_af_means)
    print(f"\n  Control sets with data: {len(control_af_means)}")
    if len(control_af_means) > 0:
        print(f"  Control AF_NFE_log: mean={control_af_means.mean():.4f}, "
              f"std={control_af_means.std():.4f}")

    # Empirical p-values
    results = []
    for subtype in SUBTYPES:
        if subtype_af[subtype] is None or len(control_af_means) == 0:
            continue
        # p-value = fraction of random sets with same or lower mean AF
        p_empirical = (control_af_means <= subtype_af[subtype]).sum() / len(control_af_means)
        results.append({
            "test": "permutation_af_nfe",
            "subtype": subtype,
            "subtype_mean_af_nfe_log": subtype_af[subtype],
            "control_mean": control_af_means.mean(),
            "control_std": control_af_means.std(),
            "p_empirical": p_empirical,
            "n_permutations": len(control_af_means),
            "significant": p_empirical < 0.05,
        })
        sig = "*" if p_empirical < 0.05 else ""
        print(f"  {subtype}: p_empirical={p_empirical:.3f} {sig}")

    # --- KEY FIGURE: Permutation test visualization ---
    if len(control_af_means) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(control_af_means, bins=20, color=COLORS["CONTROL"],
                alpha=0.6, edgecolor="black", linewidth=0.5,
                label="Random gene sets (n=100)")

        for subtype in SUBTYPES:
            if subtype_af[subtype] is not None:
                ax.axvline(subtype_af[subtype], color=COLORS[subtype],
                           linewidth=2.5, linestyle="--", label=subtype)

        ax.set_xlabel("Mean log₁₀(gnomAD AF NFE) of pathogenic variants", fontsize=11)
        ax.set_ylabel("Number of random gene sets", fontsize=11)
        ax.set_title("Permutation Test: Pathogenic Variant Allele Frequencies\n"
                     "Scoliosis subtypes vs. matched random gene sets",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")

        plt.tight_layout()
        fig_path = FIG_DIR / "permutation_test.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved permutation test figure to {fig_path}")

    return results


def part_c_lof(df_annotated):
    """LoF fraction comparison across subtypes."""
    print("\n--- Part C: LoF Fraction Comparison ---")

    results = []
    lof_data = {}

    for subtype in SUBTYPES:
        sub_df = df_annotated[df_annotated["subtype"] == subtype]
        if len(sub_df) == 0:
            continue
        lof_pct = sub_df["is_lof"].mean() * 100
        lof_data[subtype] = lof_pct
        results.append({
            "test": "lof_fraction",
            "subtype": subtype,
            "lof_pct": lof_pct,
            "n_lof": int(sub_df["is_lof"].sum()),
            "n_total": len(sub_df),
        })
        print(f"  {subtype}: {lof_pct:.1f}% LoF ({int(sub_df['is_lof'].sum())}/{len(sub_df)})")

    # Control LoF — compute per-set median
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    control_lof = []
    for cf in control_files:
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0 or "is_lof" not in df_ctrl.columns:
            continue
        control_lof.append(df_ctrl["is_lof"].mean() * 100)

    control_median = np.median(control_lof) if control_lof else 0
    control_std = np.std(control_lof) if control_lof else 0
    print(f"  CONTROL (median of {len(control_lof)} sets): "
          f"{control_median:.1f}% ± {control_std:.1f}%")

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))

    bar_subtypes = SUBTYPES + ["CONTROL"]
    bar_values = [lof_data.get(s, 0) for s in SUBTYPES] + [control_median]
    bar_errors = [0] * len(SUBTYPES) + [control_std]
    bar_colors = [COLORS[s] for s in bar_subtypes]

    bars = ax.bar(bar_subtypes, bar_values, color=bar_colors,
                  edgecolor="black", linewidth=0.5, yerr=bar_errors,
                  capsize=5, error_kw={"linewidth": 1.5})

    ax.set_ylabel("Loss-of-Function variants (%)", fontsize=11)
    ax.set_xlabel("Pathway Subtype", fontsize=11)
    ax.set_title("LoF Variant Enrichment by Scoliosis Subtype",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    # Add value labels on bars
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_path = FIG_DIR / "lof_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved LoF comparison figure to {fig_path}")

    return results


def main():
    print("=" * 60)
    print("Step 06: Enrichment Analysis")
    print("=" * 60)

    # Load annotated variants
    annot_path = DATA_DIR / "variants_annotated.csv"
    if not annot_path.exists():
        print(f"Error: {annot_path} not found. Run 03_feature_engineering.py first.")
        return

    df = pd.read_csv(annot_path)
    print(f"Loaded {len(df)} annotated variants")

    all_results = []

    # Part A: Fisher's exact test
    fisher_results = part_a_fisher(df)
    all_results.extend(fisher_results)

    # Part B: Permutation test
    perm_results = part_b_permutation(df)
    all_results.extend(perm_results)

    # Part C: LoF comparison
    lof_results = part_c_lof(df)
    all_results.extend(lof_results)

    # Save all results
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_path = RESULTS_DIR / "enrichment_results.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\nSaved enrichment results to {output_path}")

    print("\nDone! Enrichment analysis complete.")


if __name__ == "__main__":
    main()

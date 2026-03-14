#!/usr/bin/env python3
"""
Step 07: Generate all poster-ready figures.

Produces 5 publication-quality figures (300 dpi, 8x5 inches):
  fig1_variant_distribution.png — Stacked bar: variant types per subtype
  fig2_umap_genes.png           — UMAP colored by pathway subtype
  fig3_permutation_test.png     — Permutation distribution + subtype lines
  fig4_af_comparison.png        — Violin plot: gnomAD AF NFE by subtype
  fig5_lof_enrichment.png       — Bar + Fisher OR with CI

All figures use consistent color scheme.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("  Note: pip install adjustText for better gene labels on UMAP")

matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.linewidth"] = 0.8

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
SHORT_LABELS = {"ECM": "ECM", "VERTEBRAL_DEV": "Vert.Dev",
                "NEUROMUSCULAR": "Neuromusc.", "CILIARY": "Ciliary",
                "CONTROL": "Control"}
FIGSIZE = (8, 5)
DPI = 300


def fmt_pvalue(p):
    """Format p-value: p<0.001 instead of p=0.000."""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def fig1_variant_distribution(df):
    """Two-panel stacked bar: A) absolute counts, B) normalized proportions."""
    print("  Creating Fig 1: Variant Distribution...")

    cons_types = ["missense", "nonsense", "frameshift", "splice",
                  "deletion", "duplication", "other"]
    cons_colors = {
        "missense": "#3498DB",
        "nonsense": "#E74C3C",
        "frameshift": "#E67E22",
        "splice": "#9B59B6",
        "deletion": "#1ABC9C",
        "duplication": "#F39C12",
        "other": "#BDC3C7",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Count variants per subtype and consequence
    plot_data = {}
    totals = {}
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        counts = sub_df["consequence_cat"].value_counts()
        plot_data[subtype] = {ct: counts.get(ct, 0) for ct in cons_types}
        totals[subtype] = len(sub_df)

    x = np.arange(len(SUBTYPES))
    short = [SHORT_LABELS[s] for s in SUBTYPES]
    width = 0.6

    # Panel A: Absolute counts
    bottoms_a = np.zeros(len(SUBTYPES))
    for cons_type in cons_types:
        values = [plot_data[s].get(cons_type, 0) for s in SUBTYPES]
        if sum(values) == 0:
            continue
        ax1.bar(x, values, width, bottom=bottoms_a,
                color=cons_colors.get(cons_type, "#BDC3C7"),
                edgecolor="white", linewidth=0.5,
                label=cons_type.capitalize())
        bottoms_a += values

    # Add n= annotations
    for i, subtype in enumerate(SUBTYPES):
        ax1.text(i, bottoms_a[i] + 10, f"n={totals[subtype]}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=10)
    ax1.set_ylabel("Number of Pathogenic Variants", fontsize=11)
    ax1.set_title("A. Variant Counts", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",
               frameon=True, fancybox=True)

    # Panel B: Normalized 100%
    bottoms_b = np.zeros(len(SUBTYPES))
    for cons_type in cons_types:
        values_raw = [plot_data[s].get(cons_type, 0) for s in SUBTYPES]
        values_pct = [v / totals[s] * 100 if totals[s] > 0 else 0
                      for v, s in zip(values_raw, SUBTYPES)]
        if sum(values_raw) == 0:
            continue
        ax2.bar(x, values_pct, width, bottom=bottoms_b,
                color=cons_colors.get(cons_type, "#BDC3C7"),
                edgecolor="white", linewidth=0.5)
        bottoms_b += values_pct

    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=10)
    ax2.set_ylabel("Proportion (%)", fontsize=11)
    ax2.set_title("B. Variant Type Proportions", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_variant_distribution.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


def fig2_umap(df_genes):
    """UMAP plot colored by pathway subtype."""
    print("  Creating Fig 2: UMAP Genes...")

    if "umap_1" not in df_genes.columns:
        print("    Warning: UMAP coordinates not found. Run 05_clustering.py first.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Controls (background)
    ctrl = df_genes[df_genes["subtype"] == "CONTROL"]
    if len(ctrl) > 0:
        ax.scatter(ctrl["umap_1"], ctrl["umap_2"],
                   c=COLORS["CONTROL"], alpha=0.25, s=25,
                   label="Control genes", zorder=1)

    # Scoliosis genes
    texts = []
    for subtype in SUBTYPES:
        subset = df_genes[df_genes["subtype"] == subtype]
        if len(subset) == 0:
            continue
        ax.scatter(subset["umap_1"], subset["umap_2"],
                   c=COLORS[subtype], s=90, edgecolors="black",
                   linewidth=0.6, label=SHORT_LABELS.get(subtype, subtype),
                   zorder=2)
        for _, row in subset.iterrows():
            t = ax.text(row["umap_1"], row["umap_2"], row["gene"],
                        fontsize=7, fontweight="bold", ha="center", va="bottom")
            texts.append(t)

    # Spread overlapping labels
    if HAS_ADJUST_TEXT and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    force_text=(0.8, 0.8), force_points=(0.3, 0.3),
                    expand=(1.5, 1.5))

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("Gene-Level UMAP — Scoliosis Pathway Subtypes",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best", frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_umap_genes.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig3_permutation(df):
    """Permutation test distribution with subtype vertical lines."""
    print("  Creating Fig 3: Permutation Test...")

    # Scoliosis subtype mean AF
    subtype_af = {}
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        if len(sub_df) > 0 and "gnomad_af_nfe_log" in sub_df.columns:
            subtype_af[subtype] = sub_df["gnomad_af_nfe_log"].mean()

    # Control set means
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    control_af_means = []
    for cf in control_files:
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0 or "gnomad_af_nfe_log" not in df_ctrl.columns:
            continue
        control_af_means.append(df_ctrl["gnomad_af_nfe_log"].mean())

    if not control_af_means:
        print("    No control data available, skipping figure.")
        return

    control_af_means = np.array(control_af_means)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.hist(control_af_means, bins=20, color=COLORS["CONTROL"],
            alpha=0.6, edgecolor="black", linewidth=0.5,
            label=f"Random gene sets (n={len(control_af_means)})")

    for subtype in SUBTYPES:
        if subtype in subtype_af:
            p_emp = (control_af_means <= subtype_af[subtype]).sum() / len(control_af_means)
            label = f"{SHORT_LABELS.get(subtype, subtype)} ({fmt_pvalue(p_emp)})"
            ax.axvline(subtype_af[subtype], color=COLORS[subtype],
                       linewidth=2.5, linestyle="--", label=label)

    ax.set_xlabel("Mean log₁₀(gnomAD AF NFE) of pathogenic variants", fontsize=11)
    ax.set_ylabel("Number of random gene sets", fontsize=11)
    ax.set_title("Permutation Test: Pathogenic Variant Allele Frequencies\n"
                 "Scoliosis subtypes vs. transcript-length-matched random genes",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best", frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_permutation_test.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


def fig4_af_violin(df):
    """Violin plot: gnomAD AF NFE by subtype vs control."""
    print("  Creating Fig 4: AF Comparison (Violin)...")

    if "gnomad_af_nfe_log" not in df.columns:
        print("    Warning: gnomad_af_nfe_log not found, skipping.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Collect data for each subtype
    plot_subtypes = SUBTYPES + ["CONTROL"]

    # For control, aggregate from control files
    control_afs = []
    for cf in sorted(CONTROLS_DIR.glob("set_*.csv")):
        df_ctrl = pd.read_csv(cf)
        if "gnomad_af_nfe_log" in df_ctrl.columns:
            control_afs.extend(df_ctrl["gnomad_af_nfe_log"].dropna().tolist())

    violin_data = []
    violin_labels = []
    violin_colors = []

    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        vals = sub_df["gnomad_af_nfe_log"].dropna()
        if len(vals) >= 2:
            violin_data.append(vals.values)
            violin_labels.append(subtype)
            violin_colors.append(COLORS[subtype])

    if control_afs:
        violin_data.append(np.array(control_afs))
        violin_labels.append("CONTROL")
        violin_colors.append(COLORS["CONTROL"])

    if not violin_data:
        print("    Insufficient data for violin plot.")
        return

    parts = ax.violinplot(violin_data, positions=range(len(violin_data)),
                          showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.6)
    for partname in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(1)

    ax.set_xticks(range(len(violin_labels)))
    ax.set_xticklabels(violin_labels, fontsize=10, rotation=15)
    ax.set_ylabel("log₁₀(gnomAD AF NFE)", fontsize=11)
    ax.set_title("Population Allele Frequency Distribution\n"
                 "Pathogenic variants by scoliosis subtype",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_af_comparison.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


def fig5_lof_enrichment(df):
    """Bar plot: LoF fraction + Fisher OR with CI per subtype."""
    print("  Creating Fig 5: LoF Enrichment...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: LoF % bar chart
    lof_pcts = {}
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        if len(sub_df) > 0:
            lof_pcts[subtype] = sub_df["is_lof"].mean() * 100

    # Control median
    ctrl_lof = []
    for cf in sorted(CONTROLS_DIR.glob("set_*.csv")):
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) > 0 and "is_lof" in df_ctrl.columns:
            ctrl_lof.append(df_ctrl["is_lof"].mean() * 100)
    ctrl_median = np.median(ctrl_lof) if ctrl_lof else 0
    ctrl_std = np.std(ctrl_lof) if ctrl_lof else 0

    bar_labels_raw = SUBTYPES + ["CONTROL"]
    n_ctrl = len(ctrl_lof)
    bar_labels = [SHORT_LABELS[s] for s in SUBTYPES] + [f"Control\n(n={n_ctrl} sets)"]
    bar_vals = [lof_pcts.get(s, 0) for s in SUBTYPES] + [ctrl_median]
    bar_errs = [0] * len(SUBTYPES) + [ctrl_std]
    bar_cols = [COLORS[s] for s in bar_labels_raw]

    bars = ax1.bar(range(len(bar_vals)), bar_vals, color=bar_cols,
                   edgecolor="black", linewidth=0.5,
                   yerr=bar_errs, capsize=5)
    for bar, val in zip(bars, bar_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(range(len(bar_labels)))
    ax1.set_xticklabels(bar_labels, fontsize=10)
    ax1.set_ylabel("Loss-of-Function variants (%)", fontsize=11)
    ax1.set_title("A. LoF Variant Fraction", fontsize=12, fontweight="bold")

    # Right panel: Fisher OR with CI
    fisher_data = []
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        oth_df = df[(df["subtype"] != subtype) & (df["subtype"] != "CONTROL")]
        if len(sub_df) == 0 or len(oth_df) == 0:
            continue

        lof_sub = int(sub_df["is_lof"].sum())
        non_lof_sub = len(sub_df) - lof_sub
        lof_oth = int(oth_df["is_lof"].sum())
        non_lof_oth = len(oth_df) - lof_oth

        table = [[lof_sub, non_lof_sub], [lof_oth, non_lof_oth]]

        # Add 0.5 continuity correction if any cell is 0
        if any(x == 0 for row in table for x in row):
            table = [[x + 0.5 for x in row] for row in table]

        odds_ratio, p_value = stats.fisher_exact(
            [[int(x) for x in row] for row in table],
            alternative="two-sided"
        )

        # CI (Woolf's method)
        log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
        se = np.sqrt(sum(1.0 / x for row in table for x in row))
        ci_lower = np.exp(log_or - 1.96 * se)
        ci_upper = np.exp(log_or + 1.96 * se)

        fisher_data.append({
            "subtype": subtype,
            "or": odds_ratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p": p_value,
        })

    if fisher_data:
        y_pos = range(len(fisher_data))
        ors = [d["or"] for d in fisher_data]
        ci_lows = [d["or"] - d["ci_lower"] for d in fisher_data]
        ci_highs = [d["ci_upper"] - d["or"] for d in fisher_data]
        labels = [d["subtype"] for d in fisher_data]
        colors = [COLORS[d["subtype"]] for d in fisher_data]

        ax2.barh(y_pos, ors, color=colors, edgecolor="black",
                 linewidth=0.5, height=0.5)
        ax2.errorbar(ors, y_pos, xerr=[ci_lows, ci_highs],
                     fmt="none", color="black", capsize=5, linewidth=1.5)
        ax2.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([SHORT_LABELS.get(l, l) for l in labels], fontsize=10)
        ax2.set_xlabel("Odds Ratio (LoF enrichment)", fontsize=11)
        ax2.set_title("B. Fisher's Exact Test — LoF OR",
                      fontsize=12, fontweight="bold")

        # Add p-value annotations
        for i, d in enumerate(fisher_data):
            p_text = fmt_pvalue(d["p"])
            ax2.text(d["ci_upper"] + 0.1, i, p_text, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_lof_enrichment.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("Step 07: Generating Poster-Ready Figures")
    print("=" * 60)

    # Load data
    df_variants = pd.read_csv(DATA_DIR / "variants_annotated.csv")
    print(f"Loaded {len(df_variants)} annotated variants")

    clustering_path = RESULTS_DIR / "gene_clustering.csv"
    df_genes = pd.read_csv(clustering_path) if clustering_path.exists() else None

    # Generate all figures
    print("\nGenerating figures...")

    fig1_variant_distribution(df_variants)
    print("    ✓ fig1_variant_distribution.png")

    if df_genes is not None:
        fig2_umap(df_genes)
        print("    ✓ fig2_umap_genes.png")
    else:
        print("    ⚠ Skipping fig2 (no clustering data)")

    fig3_permutation(df_variants)
    print("    ✓ fig3_permutation_test.png")

    fig4_af_violin(df_variants)
    print("    ✓ fig4_af_comparison.png")

    fig5_lof_enrichment(df_variants)
    print("    ✓ fig5_lof_enrichment.png")

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()

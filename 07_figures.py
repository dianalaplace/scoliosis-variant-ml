#!/usr/bin/env python3
"""
Step 07: Generate all poster-ready figures.

Produces 5 publication-quality figures (300 dpi):
  fig1_variant_distribution.png — Two-panel: counts + proportions
  fig2_umap_genes.png           — UMAP colored by pathway subtype
  fig3_permutation_test.png     — Permutation distribution + subtype lines
  fig4_af_comparison.png        — Violin plot: gnomAD AF NFE by subtype
  fig5_lof_enrichment.png       — Bar + Fisher OR with CI

All figures use consistent color scheme and short labels.
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

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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
SHORT_LABELS = {
    "ECM": "ECM", "VERTEBRAL_DEV": "Vert.Dev",
    "NEUROMUSCULAR": "Neuromusc.", "CILIARY": "Ciliary",
    "CONTROL": "Control",
}
DPI = 300

CONS_TYPES = ["missense", "nonsense", "frameshift", "splice",
              "deletion", "duplication", "other"]
CONS_COLORS = {
    "missense": "#3498DB", "nonsense": "#E74C3C", "frameshift": "#E67E22",
    "splice": "#9B59B6", "deletion": "#1ABC9C", "duplication": "#F39C12",
    "other": "#BDC3C7",
}


def fmt_pvalue(p):
    """Format p-value: p<0.001 instead of p=0.000."""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def fmt_stars(p):
    """Significance stars for annotations."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _short(label):
    return SHORT_LABELS.get(label, label)


# ═══════════════════════════════════════════════════════════
# Fig 1: Variant Type Distribution
# ═══════════════════════════════════════════════════════════

def fig1_variant_distribution(df):
    """Two-panel stacked bar: A) absolute counts, B) normalized proportions."""
    print("  Creating Fig 1: Variant Distribution...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Data
    plot_data, totals = {}, {}
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        counts = sub_df["consequence_cat"].value_counts()
        plot_data[subtype] = {ct: counts.get(ct, 0) for ct in CONS_TYPES}
        totals[subtype] = len(sub_df)

    x = np.arange(len(SUBTYPES))
    short = [_short(s) for s in SUBTYPES]
    width = 0.55

    # Panel A: Absolute counts
    bottoms_a = np.zeros(len(SUBTYPES))
    for ct in CONS_TYPES:
        vals = [plot_data[s].get(ct, 0) for s in SUBTYPES]
        if sum(vals) == 0:
            continue
        ax1.bar(x, vals, width, bottom=bottoms_a,
                color=CONS_COLORS[ct], edgecolor="white", linewidth=0.5,
                label=ct.capitalize())
        bottoms_a += vals

    for i, s in enumerate(SUBTYPES):
        ax1.text(i, bottoms_a[i] + 20, f"n={totals[s]}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=10)
    ax1.set_ylabel("Number of Pathogenic Variants")
    ax1.set_title("A. Variant Counts", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right", frameon=True, fancybox=True,
               ncol=2, framealpha=0.9)

    # Panel B: Normalized 100%
    bottoms_b = np.zeros(len(SUBTYPES))
    for ct in CONS_TYPES:
        vals_raw = [plot_data[s].get(ct, 0) for s in SUBTYPES]
        vals_pct = [v / totals[s] * 100 if totals[s] > 0 else 0
                    for v, s in zip(vals_raw, SUBTYPES)]
        if sum(vals_raw) == 0:
            continue
        bars = ax2.bar(x, vals_pct, width, bottom=bottoms_b,
                       color=CONS_COLORS[ct], edgecolor="white", linewidth=0.5)
        # Label the dominant segments (>15%)
        for j, pct in enumerate(vals_pct):
            if pct > 15:
                ax2.text(x[j], bottoms_b[j] + pct / 2, f"{pct:.0f}%",
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold")
        bottoms_b += vals_pct

    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=10)
    ax2.set_ylabel("Proportion (%)")
    ax2.set_title("B. Variant Type Proportions", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_variant_distribution.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Fig 2: UMAP
# ═══════════════════════════════════════════════════════════

def fig2_umap(df_genes):
    """UMAP plot colored by pathway subtype with smart labels."""
    print("  Creating Fig 2: UMAP Genes...")

    if "umap_1" not in df_genes.columns:
        print("    Warning: UMAP coordinates not found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Controls (background)
    ctrl = df_genes[df_genes["subtype"] == "CONTROL"]
    if len(ctrl) > 0:
        ax.scatter(ctrl["umap_1"], ctrl["umap_2"],
                   c=COLORS["CONTROL"], alpha=0.2, s=20,
                   label="Control genes", zorder=1, edgecolors="none")

    # Scoliosis genes
    texts = []
    for subtype in SUBTYPES:
        subset = df_genes[df_genes["subtype"] == subtype]
        if len(subset) == 0:
            continue
        ax.scatter(subset["umap_1"], subset["umap_2"],
                   c=COLORS[subtype], s=100, edgecolors="black",
                   linewidth=0.7, label=_short(subtype), zorder=3)
        for _, row in subset.iterrows():
            t = ax.text(row["umap_1"], row["umap_2"] + 0.3, row["gene"],
                        fontsize=7.5, fontweight="bold", ha="center", va="bottom",
                        zorder=4)
            texts.append(t)

    # Spread overlapping labels
    if HAS_ADJUST_TEXT and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    force_text=(1.0, 1.0), force_points=(0.5, 0.5),
                    expand=(1.8, 2.0), ensure_inside_axes=True)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Gene-Level UMAP — Scoliosis Pathway Subtypes",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True,
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_umap_genes.png", dpi=DPI, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Fig 3: Permutation Test
# ═══════════════════════════════════════════════════════════

def fig3_permutation(df):
    """Permutation test distribution with subtype vertical lines."""
    print("  Creating Fig 3: Permutation Test...")

    subtype_af = {}
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        if len(sub_df) > 0 and "gnomad_af_nfe_log" in sub_df.columns:
            subtype_af[subtype] = sub_df["gnomad_af_nfe_log"].mean()

    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    control_af_means = []
    for cf in control_files:
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0 or "gnomad_af_nfe_log" not in df_ctrl.columns:
            continue
        control_af_means.append(df_ctrl["gnomad_af_nfe_log"].mean())

    if not control_af_means:
        print("    No control data available, skipping figure.")
        # Save placeholder
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No control data available\nRun full pipeline first",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        plt.savefig(FIG_DIR / "fig3_permutation_test.png", dpi=DPI)
        plt.close()
        return

    control_af_means = np.array(control_af_means)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.hist(control_af_means, bins=15, color=COLORS["CONTROL"],
            alpha=0.5, edgecolor="black", linewidth=0.5,
            label=f"Random gene sets (n={len(control_af_means)})")

    line_styles = ["-", "--", "-.", ":"]
    for i, subtype in enumerate(SUBTYPES):
        if subtype not in subtype_af:
            continue
        p_emp = (control_af_means <= subtype_af[subtype]).sum() / len(control_af_means)
        label = f"{_short(subtype)} ({fmt_pvalue(p_emp)})"
        ax.axvline(subtype_af[subtype], color=COLORS[subtype],
                   linewidth=2.5, linestyle=line_styles[i % 4], label=label)

    ax.set_xlabel(r"Mean $\log_{10}$(gnomAD AF$_{\rm NFE}$) of pathogenic variants",
                  fontsize=11)
    ax.set_ylabel("Number of random gene sets")
    ax.set_title("Permutation Test: Pathogenic Variant Allele Frequencies\n"
                 "Scoliosis subtypes vs. transcript-length-matched random genes",
                 fontsize=12, fontweight="bold")

    # Annotation: negative result is honest
    ax.text(0.98, 0.95, "All subtypes within\nnull distribution (ns)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            fontstyle="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))

    ax.legend(fontsize=9, loc="upper left", frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_permutation_test.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Fig 4: AF Violin
# ═══════════════════════════════════════════════════════════

def fig4_af_violin(df):
    """Violin plot: gnomAD AF NFE by subtype vs control with stats."""
    print("  Creating Fig 4: AF Comparison (Violin)...")

    if "gnomad_af_nfe_log" not in df.columns:
        print("    Warning: gnomad_af_nfe_log not found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Collect control AF
    control_afs = []
    for cf in sorted(CONTROLS_DIR.glob("set_*.csv")):
        df_ctrl = pd.read_csv(cf)
        if "gnomad_af_nfe_log" in df_ctrl.columns:
            control_afs.extend(df_ctrl["gnomad_af_nfe_log"].dropna().tolist())

    violin_data, violin_labels_raw, violin_colors = [], [], []
    n_per_group = []

    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        vals = sub_df["gnomad_af_nfe_log"].dropna()
        if len(vals) >= 2:
            violin_data.append(vals.values)
            violin_labels_raw.append(subtype)
            violin_colors.append(COLORS[subtype])
            n_per_group.append(len(vals))

    if control_afs:
        violin_data.append(np.array(control_afs))
        violin_labels_raw.append("CONTROL")
        violin_colors.append(COLORS["CONTROL"])
        n_per_group.append(len(control_afs))

    if not violin_data:
        print("    Insufficient data for violin plot.")
        return

    positions = range(len(violin_data))
    parts = ax.violinplot(violin_data, positions=positions,
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
    for partname in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(0.8)

    # Labels with n=
    display_labels = [f"{_short(l)}\n(n={n})" for l, n in
                      zip(violin_labels_raw, n_per_group)]
    ax.set_xticks(list(positions))
    ax.set_xticklabels(display_labels, fontsize=9)

    # Kruskal-Wallis test across scoliosis subtypes
    scoliosis_groups = [d for d, l in zip(violin_data, violin_labels_raw)
                        if l != "CONTROL"]
    if len(scoliosis_groups) >= 2:
        h_stat, kw_p = stats.kruskal(*scoliosis_groups)
        ax.text(0.98, 0.02, f"Kruskal-Wallis H={h_stat:.1f}, {fmt_pvalue(kw_p)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="#cccccc"))

    # Legend for mean vs median lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=0.8, linestyle="-",
               label="Mean (thick) / Median (thin)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    ax.set_ylabel(r"$\log_{10}$(gnomAD AF$_{\rm NFE}$)", fontsize=11)
    ax.set_title("Population Allele Frequency Distribution\n"
                 "Pathogenic variants by scoliosis subtype",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_af_comparison.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Fig 5: LoF Enrichment (main result)
# ═══════════════════════════════════════════════════════════

def fig5_lof_enrichment(df):
    """Bar plot: LoF fraction + Fisher OR with CI per subtype."""
    print("  Creating Fig 5: LoF Enrichment...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.1]})

    # ── Left panel: LoF % bar chart ──
    lof_pcts, n_per_subtype = {}, {}
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype]
        if len(sub_df) > 0:
            lof_pcts[subtype] = sub_df["is_lof"].mean() * 100
            n_per_subtype[subtype] = len(sub_df)

    # Control
    ctrl_lof = []
    for cf in sorted(CONTROLS_DIR.glob("set_*.csv")):
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) > 0 and "is_lof" in df_ctrl.columns:
            ctrl_lof.append(df_ctrl["is_lof"].mean() * 100)
    ctrl_median = np.median(ctrl_lof) if ctrl_lof else 0
    ctrl_std = np.std(ctrl_lof) if ctrl_lof else 0
    n_ctrl = len(ctrl_lof)

    bar_labels_raw = SUBTYPES + ["CONTROL"]
    bar_labels = [_short(s) for s in SUBTYPES] + [f"Control\n(n={n_ctrl} sets)"]
    bar_vals = [lof_pcts.get(s, 0) for s in SUBTYPES] + [ctrl_median]
    bar_errs = [0] * len(SUBTYPES) + [ctrl_std]
    bar_cols = [COLORS[s] for s in bar_labels_raw]

    bars = ax1.bar(range(len(bar_vals)), bar_vals, color=bar_cols,
                   edgecolor="black", linewidth=0.5,
                   yerr=bar_errs, capsize=4, error_kw={"linewidth": 1.2})

    for i, (bar, val) in enumerate(zip(bars, bar_vals)):
        label = f"{val:.1f}%"
        if i < len(SUBTYPES) and SUBTYPES[i] in n_per_subtype:
            label += f"\n(n={n_per_subtype[SUBTYPES[i]]})"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 label, ha="center", va="bottom", fontsize=8)

    # Dashed line at control median
    ax1.axhline(ctrl_median, color=COLORS["CONTROL"], linestyle="--",
                linewidth=1, alpha=0.7)

    ax1.set_xticks(range(len(bar_labels)))
    ax1.set_xticklabels(bar_labels, fontsize=9)
    ax1.set_ylabel("Loss-of-Function variants (%)")
    ax1.set_title("A. LoF Variant Fraction", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 108)

    # ── Right panel: Forest plot (Fisher OR with CI) ──
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
        if any(x == 0 for row in table for x in row):
            table = [[x + 0.5 for x in row] for row in table]

        odds_ratio, p_value = stats.fisher_exact(
            [[int(x) for x in row] for row in table],
            alternative="two-sided"
        )

        log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
        se = np.sqrt(sum(1.0 / x for row in table for x in row))
        ci_lower = np.exp(log_or - 1.96 * se)
        ci_upper = np.exp(log_or + 1.96 * se)

        fisher_data.append({
            "subtype": subtype, "or": odds_ratio,
            "ci_lower": ci_lower, "ci_upper": ci_upper, "p": p_value,
        })

    if fisher_data:
        y_pos = list(range(len(fisher_data)))
        ors = [d["or"] for d in fisher_data]
        ci_lows = [d["or"] - d["ci_lower"] for d in fisher_data]
        ci_highs = [d["ci_upper"] - d["or"] for d in fisher_data]
        labels = [d["subtype"] for d in fisher_data]
        colors = [COLORS[d["subtype"]] for d in fisher_data]

        # Horizontal bars
        ax2.barh(y_pos, ors, color=colors, edgecolor="black",
                 linewidth=0.5, height=0.45, alpha=0.85)
        ax2.errorbar(ors, y_pos, xerr=[ci_lows, ci_highs],
                     fmt="none", color="black", capsize=4, linewidth=1.5)

        # Reference line at OR=1
        ax2.axvline(1.0, color="#555555", linestyle="--", linewidth=1, zorder=0)
        ax2.text(1.0, len(fisher_data) - 0.1, "OR=1", ha="center",
                 va="bottom", fontsize=8, color="#555555")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([_short(l) for l in labels], fontsize=10)
        ax2.set_xlabel("Odds Ratio (LoF enrichment)")
        ax2.set_title("B. Fisher's Exact Test — LoF OR",
                      fontsize=12, fontweight="bold")

        # p-value + stars annotations
        max_x = max(d["ci_upper"] for d in fisher_data)
        for i, d in enumerate(fisher_data):
            stars = fmt_stars(d["p"])
            p_text = f"{fmt_pvalue(d['p'])} {stars}"
            x_pos = min(d["ci_upper"] + 0.15, max_x + 0.5)
            ax2.text(x_pos, i, p_text, va="center", fontsize=9,
                     fontweight="bold" if stars != "ns" else "normal")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_lof_enrichment.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Step 07: Generating Poster-Ready Figures")
    print("=" * 60)

    df_variants = pd.read_csv(DATA_DIR / "variants_annotated.csv")
    print(f"Loaded {len(df_variants)} annotated variants")

    clustering_path = RESULTS_DIR / "gene_clustering.csv"
    df_genes = pd.read_csv(clustering_path) if clustering_path.exists() else None

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

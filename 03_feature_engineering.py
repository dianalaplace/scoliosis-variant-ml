#!/usr/bin/env python3
"""
Step 03: Feature engineering for ML analysis.

Encodes variant-level features:
- consequence_cat → one-hot
- review_stars (0-4)
- is_pathogenic binary
- gnomAD AF log-transforms
- is_rare, is_lof binary

Output: data/variants_annotated.csv
        Updates control set CSVs in data/clinvar_controls/
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"

# Review status → stars mapping
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
}


def classify_consequence(row):
    """Classify molecular consequence into categories."""
    mol_cons = str(row.get("molecular_consequence", "")).lower()
    var_type = str(row.get("variant_type", "")).lower()
    title = str(row.get("title", "")).lower()

    # Check molecular consequence first
    if "missense" in mol_cons:
        return "missense"
    if "nonsense" in mol_cons or "stop_gained" in mol_cons:
        return "nonsense"
    if "frameshift" in mol_cons:
        return "frameshift"
    if "splice" in mol_cons:
        return "splice"

    # Check variant type
    if "deletion" in var_type:
        return "deletion"
    if "duplication" in var_type:
        return "duplication"
    if "insertion" in var_type:
        return "insertion"

    # Check title as fallback
    if "missense" in title:
        return "missense"
    if any(x in title for x in ["nonsense", "stop", "ter"]):
        return "nonsense"
    if "frameshift" in title:
        return "frameshift"
    if "splice" in title or "intron" in title:
        return "splice"
    if "del" in title:
        return "deletion"
    if "dup" in title:
        return "duplication"

    return "other"


def map_review_stars(review_status):
    """Map review status string to 0-4 star rating."""
    if pd.isna(review_status):
        return 0
    review_lower = str(review_status).lower().strip()
    for pattern, stars in REVIEW_STARS.items():
        if pattern in review_lower:
            return stars
    return 0


def engineer_features(df):
    """Add engineered features to a variants DataFrame."""
    if len(df) == 0:
        return df

    # 1. Consequence category
    df["consequence_cat"] = df.apply(classify_consequence, axis=1)

    # 2. Review stars
    df["review_stars"] = df["review_status"].apply(map_review_stars)

    # 3. is_pathogenic (1=Pathogenic, 0=Likely pathogenic)
    df["is_pathogenic"] = (
        df["clinical_significance"]
        .fillna("")
        .str.lower()
        .str.contains("^pathogenic$|^pathogenic/", regex=True)
        .astype(int)
    )

    # 4. gnomAD AF log transforms
    AF_FLOOR = 1e-7
    for col in ["gnomad_af_global", "gnomad_af_nfe"]:
        log_col = f"{col}_log"
        if col in df.columns:
            df[log_col] = df[col].apply(
                lambda x: np.log10(max(float(x), AF_FLOOR))
                if pd.notna(x) else np.log10(AF_FLOOR)
            )
        else:
            df[log_col] = np.log10(AF_FLOOR)

    # 5. is_rare (AF_global < 0.01)
    if "gnomad_af_global" in df.columns:
        df["is_rare"] = (
            df["gnomad_af_global"].apply(
                lambda x: 1 if pd.isna(x) or float(x) < 0.01 else 0
            )
        )
    else:
        df["is_rare"] = 1

    # 6. is_lof (nonsense OR frameshift OR splice)
    lof_types = {"nonsense", "frameshift", "splice"}
    df["is_lof"] = df["consequence_cat"].isin(lof_types).astype(int)

    # 7. One-hot encode consequence categories
    consequence_dummies = pd.get_dummies(
        df["consequence_cat"], prefix="cons"
    ).astype(int)
    df = pd.concat([df, consequence_dummies], axis=1)

    return df


def main():
    print("=" * 60)
    print("Step 03: Feature Engineering")
    print("=" * 60)

    # 1. Process scoliosis variants
    input_path = DATA_DIR / "variants_with_gnomad.csv"
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run 02_fetch_gnomad.py first.")
        return

    print(f"\nProcessing scoliosis variants...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} variants")

    df = engineer_features(df)

    output_path = DATA_DIR / "variants_annotated.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"  Consequence categories:")
    print(f"    {df['consequence_cat'].value_counts().to_dict()}")
    print(f"  Review stars distribution:")
    print(f"    {df['review_stars'].value_counts().sort_index().to_dict()}")
    print(f"  Pathogenic: {df['is_pathogenic'].sum()}, "
          f"Likely pathogenic: {(1 - df['is_pathogenic']).sum()}")
    print(f"  Rare variants: {df['is_rare'].sum()} ({df['is_rare'].mean()*100:.1f}%)")
    print(f"  LoF variants: {df['is_lof'].sum()} ({df['is_lof'].mean()*100:.1f}%)")

    # 2. Process control sets
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    print(f"\nProcessing {len(control_files)} control sets...")

    for cf in control_files:
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0:
            continue
        df_ctrl = engineer_features(df_ctrl)
        df_ctrl.to_csv(cf, index=False)

    print("Done! Feature engineering complete.")


if __name__ == "__main__":
    main()

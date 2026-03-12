#!/usr/bin/env python3
"""
Step 02: Fetch population allele frequencies from gnomAD v4.

Uses gnomAD GraphQL API to look up variants by rsID.
Adds AF columns to scoliosis and control datasets.

Output: data/variants_with_gnomad.csv (scoliosis genes with gnomAD AF)
        Updates control set CSVs in data/clinvar_controls/
"""

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
RATE_LIMIT_DELAY = 0.5


def query_gnomad_by_rsid(rsid):
    """Query gnomAD v4 for a variant by rsID."""
    query = """
    query VariantByRsid($rsid: String!, $dataset: DatasetId!) {
      variant(rsid: $rsid, dataset: $dataset) {
        variant_id
        rsids
        genome {
          ac
          an
          af
          populations {
            id
            ac
            an
            af
          }
        }
        exome {
          ac
          an
          af
          populations {
            id
            ac
            an
            af
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            GNOMAD_API,
            json={
                "query": query,
                "variables": {"rsid": rsid, "dataset": "gnomad_r4"},
            },
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        if "errors" in data or not data.get("data", {}).get("variant"):
            return None

        return data["data"]["variant"]
    except Exception as e:
        print(f"    gnomAD error for {rsid}: {e}")
        return None


def extract_af(variant_data):
    """Extract allele frequencies from gnomAD response."""
    result = {
        "gnomad_af_global": None,
        "gnomad_af_nfe": None,
        "gnomad_af_fin": None,
        "gnomad_af_eas": None,
        "gnomad_af_afr": None,
    }

    if not variant_data:
        return result

    # Prefer exome data, fallback to genome
    freq_data = variant_data.get("exome") or variant_data.get("genome")
    if not freq_data:
        return result

    result["gnomad_af_global"] = freq_data.get("af")

    pop_map = {
        "nfe": "gnomad_af_nfe",
        "fin": "gnomad_af_fin",
        "eas": "gnomad_af_eas",
        "afr": "gnomad_af_afr",
    }

    for pop in freq_data.get("populations", []):
        pop_id = pop.get("id", "").lower()
        if pop_id in pop_map:
            result[pop_map[pop_id]] = pop.get("af")

    return result


def annotate_with_gnomad(df):
    """Add gnomAD allele frequencies to a DataFrame of ClinVar variants."""
    af_columns = [
        "gnomad_af_global", "gnomad_af_nfe", "gnomad_af_fin",
        "gnomad_af_eas", "gnomad_af_afr",
    ]
    for col in af_columns:
        if col not in df.columns:
            df[col] = None

    # Find variants with valid rsIDs
    valid_mask = df["rsid"].notna() & df["rsid"].astype(str).str.startswith("rs")
    n_with_rsid = valid_mask.sum()
    print(f"  {n_with_rsid} variants with rsID out of {len(df)}")

    fetched = 0
    found = 0
    for idx in df.index[valid_mask]:
        rsid = str(df.at[idx, "rsid"]).strip()
        variant_data = query_gnomad_by_rsid(rsid)
        af_data = extract_af(variant_data)

        for col, val in af_data.items():
            df.at[idx, col] = val

        fetched += 1
        if af_data["gnomad_af_global"] is not None:
            found += 1

        if fetched % 50 == 0:
            print(f"    Fetched {fetched}/{n_with_rsid}... (found: {found})")

        time.sleep(RATE_LIMIT_DELAY)

    print(f"  Found gnomAD data for {found}/{fetched} queried variants")
    return df


def main():
    print("=" * 60)
    print("Step 02: Fetching gnomAD Allele Frequencies")
    print("=" * 60)

    # 1. Process scoliosis variants
    scoliosis_path = DATA_DIR / "clinvar_scoliosis.csv"
    if not scoliosis_path.exists():
        print(f"Error: {scoliosis_path} not found. Run 01_fetch_clinvar.py first.")
        return

    print(f"\nProcessing scoliosis variants...")
    df_scol = pd.read_csv(scoliosis_path)
    print(f"  Loaded {len(df_scol)} variants")
    df_scol = annotate_with_gnomad(df_scol)

    output_path = DATA_DIR / "variants_with_gnomad.csv"
    df_scol.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    # 2. Process control sets
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    print(f"\nProcessing {len(control_files)} control sets...")

    for i, cf in enumerate(control_files):
        print(f"\n--- Control set {i} ---")
        df_ctrl = pd.read_csv(cf)
        if len(df_ctrl) == 0:
            print("  Empty set, skipping")
            continue
        if "rsid" not in df_ctrl.columns:
            print("  No rsid column, skipping")
            continue
        df_ctrl = annotate_with_gnomad(df_ctrl)
        df_ctrl.to_csv(cf, index=False)
        print(f"  Updated {cf.name}")

    print("\n" + "=" * 60)
    print("Done! gnomAD annotation complete.")


if __name__ == "__main__":
    main()

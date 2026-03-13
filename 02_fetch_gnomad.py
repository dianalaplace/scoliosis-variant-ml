#!/usr/bin/env python3
"""
Step 02: Fetch population allele frequencies from gnomAD v4.

Uses gnomAD GraphQL API to look up variants by rsID.
Falls back to estimated AF if API is unreachable (marks data as SIMULATED).

IMPORTANT: For publication/poster, run this locally with real internet
access to get genuine gnomAD allele frequencies.

Output: data/variants_with_gnomad.csv
        Updates control set CSVs in data/clinvar_controls/
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

np.random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
RATE_LIMIT_DELAY = 2.0  # gnomAD is strict on rate limiting
MAX_RETRIES = 3

GNOMAD_QUERY = """
query VariantByRsid($rsid: String!, $dataset: DatasetId!) {
  variant(rsid: $rsid, dataset: $dataset) {
    variant_id
    genome { ac an af populations { id ac an } }
    exome  { ac an af populations { id ac an } }
  }
}
"""


# ──────────────── Real gnomAD API ────────────────

def check_gnomad_available():
    """Check if gnomAD API is reachable."""
    try:
        # Use a minimal valid introspection query
        test_query = '{ __typename }'
        resp = requests.post(
            GNOMAD_API,
            json={"query": test_query},
            timeout=15,
            headers={"Content-Type": "application/json"},
        )
        # 200 = valid query, 400 = invalid query but API is up
        # Both mean the API is reachable
        return resp.status_code in (200, 400)
    except Exception:
        return False


def query_gnomad(rsid):
    """Query gnomAD v4 for a variant by rsID with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                GNOMAD_API,
                json={
                    "query": GNOMAD_QUERY,
                    "variables": {"rsid": rsid, "dataset": "gnomad_r4"},
                },
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                print(f"    429 rate limit, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if "errors" in data or not data.get("data", {}).get("variant"):
                return None
            return data["data"]["variant"]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                print(f"    gnomAD error {rsid}: {e}", flush=True)
    return None


def extract_af(variant_data):
    """Extract allele frequencies from gnomAD response."""
    result = {
        "gnomad_af_global": None, "gnomad_af_nfe": None,
        "gnomad_af_fin": None, "gnomad_af_eas": None, "gnomad_af_afr": None,
    }
    if not variant_data:
        return result

    freq_data = variant_data.get("exome") or variant_data.get("genome")
    if not freq_data:
        return result

    result["gnomad_af_global"] = freq_data.get("af")

    pop_map = {"nfe": "gnomad_af_nfe", "fin": "gnomad_af_fin",
               "eas": "gnomad_af_eas", "afr": "gnomad_af_afr"}
    for pop in freq_data.get("populations", []):
        pid = pop.get("id", "").lower()
        if pid in pop_map and pop.get("an", 0) > 0:
            result[pop_map[pid]] = pop["ac"] / pop["an"]
    return result


def annotate_real_gnomad(df):
    """Add real gnomAD allele frequencies via API."""
    af_columns = ["gnomad_af_global", "gnomad_af_nfe", "gnomad_af_fin",
                   "gnomad_af_eas", "gnomad_af_afr"]
    for col in af_columns:
        if col not in df.columns:
            df[col] = None

    valid_mask = df["rsid"].notna() & df["rsid"].astype(str).str.startswith("rs")
    n_with_rsid = valid_mask.sum()
    print(f"  {n_with_rsid} variants with rsID out of {len(df)}", flush=True)

    fetched = 0
    found = 0
    for idx in df.index[valid_mask]:
        rsid = str(df.at[idx, "rsid"]).strip()
        variant_data = query_gnomad(rsid)
        af_data = extract_af(variant_data)

        for col, val in af_data.items():
            df.at[idx, col] = val

        fetched += 1
        if af_data["gnomad_af_global"] is not None:
            found += 1
        if fetched % 50 == 0:
            print(f"    Fetched {fetched}/{n_with_rsid}... (found: {found})", flush=True)
        time.sleep(RATE_LIMIT_DELAY)

    print(f"  Found gnomAD data for {found}/{fetched} queried variants", flush=True)
    df["gnomad_source"] = "gnomad_v4_api"
    return df


# ──────────────── Fallback: Simulated AF ────────────────

def annotate_simulated_af(df):
    """
    Estimate realistic AF for pathogenic variants when gnomAD is unreachable.

    ⚠️  SIMULATED DATA — NOT FOR PUBLICATION ⚠️
    Run locally with real gnomAD API before using in poster/thesis.

    Model: log-normal distribution based on published pathogenic variant AF:
    - Most pathogenic: AF ~ 1e-6 to 1e-4
    - LoF: typically AF < 1e-5
    - Missense: AF ~ 1e-5 to 5e-4
    """
    n = len(df)
    af_columns = ["gnomad_af_global", "gnomad_af_nfe", "gnomad_af_fin",
                   "gnomad_af_eas", "gnomad_af_afr"]
    for col in af_columns:
        df[col] = np.nan

    base_log_af = np.random.normal(-5.5, 1.0, n)

    for i, (idx, row) in enumerate(df.iterrows()):
        mol_cons = str(row.get("molecular_consequence", "")).lower()
        var_type = str(row.get("variant_type", "")).lower()

        if any(x in mol_cons for x in ["frameshift", "nonsense", "stop_gained"]):
            base_log_af[i] -= 0.5
        elif "splice" in mol_cons:
            base_log_af[i] -= 0.3
        elif "missense" in mol_cons:
            base_log_af[i] += 0.3
        if "deletion" in var_type or "duplication" in var_type:
            base_log_af[i] -= 0.2

    base_log_af = np.clip(base_log_af, -7.5, -2.5)
    global_af = 10 ** base_log_af

    # ~20% absent from gnomAD
    absent_mask = np.random.random(n) < 0.20
    global_af[absent_mask] = np.nan
    df["gnomad_af_global"] = global_af

    valid = ~absent_mask
    df.loc[valid, "gnomad_af_nfe"] = global_af[valid] * np.random.lognormal(0, 0.3, n)[valid]
    df.loc[valid, "gnomad_af_fin"] = global_af[valid] * np.random.lognormal(0, 0.5, n)[valid]
    df.loc[valid, "gnomad_af_eas"] = global_af[valid] * np.random.lognormal(-0.3, 0.5, n)[valid]
    df.loc[valid, "gnomad_af_afr"] = global_af[valid] * np.random.lognormal(-0.2, 0.3, n)[valid]

    for col in af_columns:
        df[col] = df[col].clip(upper=1.0)

    present = df["gnomad_af_global"].notna().sum()
    print(f"  ⚠️  SIMULATED AF: {present}/{n} variants", flush=True)
    print(f"  Median global AF: {df['gnomad_af_global'].median():.2e}", flush=True)
    df["gnomad_source"] = "SIMULATED"
    return df


# ──────────────── Main ────────────────

def process_dataframe(df, use_real_api):
    """Annotate a DataFrame with gnomAD AF (real or simulated)."""
    if use_real_api:
        return annotate_real_gnomad(df)
    else:
        return annotate_simulated_af(df)


def main():
    print("=" * 60, flush=True)
    print("Step 02: gnomAD Allele Frequency Annotation", flush=True)
    print("=" * 60, flush=True)

    use_real = check_gnomad_available()
    if use_real:
        print("\n✓ gnomAD API reachable — fetching real data", flush=True)
        print("  (This will take ~15-20 min for scoliosis variants)", flush=True)
    else:
        print("\n⚠️  gnomAD API unreachable — using SIMULATED AF", flush=True)
        print("  Run locally with internet for real data before publication!", flush=True)

    # 1. Scoliosis variants
    scoliosis_path = DATA_DIR / "clinvar_scoliosis.csv"
    if not scoliosis_path.exists():
        print(f"Error: {scoliosis_path} not found. Run 01_fetch_clinvar.py first.")
        return

    print(f"\nProcessing scoliosis variants...", flush=True)
    df_scol = pd.read_csv(scoliosis_path)
    print(f"  Loaded {len(df_scol)} variants", flush=True)
    df_scol = process_dataframe(df_scol, use_real)

    output_path = DATA_DIR / "variants_with_gnomad.csv"
    df_scol.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}", flush=True)

    # 2. Control sets
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    control_files = sorted(CONTROLS_DIR.glob("set_*.csv"))
    print(f"\nProcessing {len(control_files)} control sets...", flush=True)

    if len(control_files) == 0:
        print("  ⚠️  No control sets found in data/clinvar_controls/", flush=True)
        print("  Run 01b_fetch_controls.py first to generate them.", flush=True)
        print("  (Or run ./run_local.sh for the full pipeline)", flush=True)
    else:
        for i, cf in enumerate(control_files):
            df_ctrl = pd.read_csv(cf)
            if len(df_ctrl) == 0:
                continue
            df_ctrl = process_dataframe(df_ctrl, use_real)
            df_ctrl.to_csv(cf, index=False)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(control_files)} sets", flush=True)

    source_label = "gnomAD v4 API" if use_real else "SIMULATED (run locally for real data)"
    print(f"\nDone! Source: {source_label}", flush=True)


if __name__ == "__main__":
    main()

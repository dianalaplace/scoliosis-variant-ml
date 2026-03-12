#!/usr/bin/env python3
"""
Step 01: Fetch pathogenic/likely-pathogenic variants from ClinVar.

Uses NCBI Entrez esearch + esummary (JSON) for reliability.
Processes both scoliosis genes and 100 random control sets.

Output:
  data/clinvar_scoliosis.csv        — variants for 20 scoliosis genes
  data/clinvar_controls/set_{i}.csv — variants for each of 100 random sets
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"
CONTROLS_DIR.mkdir(parents=True, exist_ok=True)

EMAIL = "diana.lysenko@students.jku.at"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_LIMIT_DELAY = 0.35


def load_scoliosis_genes():
    """Load gene list and subtype mapping from config."""
    with open(DATA_DIR / "genes_config.json") as f:
        config = json.load(f)
    gene_subtype = {}
    for subtype, info in config["subtypes"].items():
        for gene in info["genes"]:
            gene_subtype[gene] = subtype
    return gene_subtype


def esearch_clinvar(gene_name, retmax=1000):
    """Search ClinVar for pathogenic/likely-pathogenic variants."""
    query = (
        f'{gene_name}[Gene Name] AND '
        f'("Pathogenic"[Clinical significance] OR '
        f'"Likely pathogenic"[Clinical significance])'
    )
    params = {
        "db": "clinvar",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "email": EMAIL,
    }
    try:
        resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        return ids
    except Exception as e:
        print(f"    esearch error for {gene_name}: {e}", flush=True)
        return []


def esummary_clinvar(id_list):
    """Fetch ClinVar summaries for a list of IDs via esummary (JSON)."""
    if not id_list:
        return []

    records = []
    batch_size = 200
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start:start + batch_size]
        params = {
            "db": "clinvar",
            "id": ",".join(batch),
            "retmode": "json",
            "email": EMAIL,
        }
        try:
            resp = requests.get(
                f"{NCBI_BASE}/esummary.fcgi", params=params, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            result = data.get("result", {})
            uids = result.get("uids", [])
            for uid in uids:
                doc = result.get(str(uid), {})
                if not doc or "error" in doc:
                    continue
                parsed = parse_esummary_doc(uid, doc)
                if parsed:
                    records.append(parsed)

            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"    esummary error batch {start}: {e}", flush=True)
            time.sleep(1)

    return records


def parse_esummary_doc(uid, doc):
    """Parse a ClinVar esummary JSON document (2024+ format)."""
    record = {
        "clinvar_id": str(uid),
        "title": doc.get("title", ""),
        "variant_type": doc.get("obj_type", ""),
        "clinical_significance": "",
        "review_status": "",
        "gene": "",
        "chromosome": "",
        "start": "",
        "stop": "",
        "rsid": "",
        "molecular_consequence": "",
        "conditions": "",
    }

    # --- Clinical significance & review status ---
    # New format: germline_classification
    germ = doc.get("germline_classification", {})
    if isinstance(germ, dict) and germ.get("description"):
        record["clinical_significance"] = germ.get("description", "")
        record["review_status"] = germ.get("review_status", "")
        # Traits from germline_classification
        trait_set = germ.get("trait_set", [])
        traits = []
        for ts in trait_set:
            if isinstance(ts, dict):
                tn = ts.get("trait_name", "")
                if tn:
                    traits.append(tn)
        record["conditions"] = "; ".join(traits)
    else:
        # Fallback: old format
        cs = doc.get("clinical_significance", {})
        if isinstance(cs, dict):
            record["clinical_significance"] = cs.get("description", "")
            record["review_status"] = cs.get("review_status", "")
        elif isinstance(cs, str):
            record["clinical_significance"] = cs

    # --- Gene ---
    genes = doc.get("genes", [])
    if genes and isinstance(genes[0], dict):
        record["gene"] = genes[0].get("symbol", "")

    # --- Molecular consequence ---
    mc_list = doc.get("molecular_consequence_list", [])
    if mc_list:
        record["molecular_consequence"] = mc_list[0] if isinstance(mc_list[0], str) else ""

    # --- Protein change ---
    protein_change = doc.get("protein_change", "")
    if protein_change:
        record["title"] = record["title"] or protein_change

    # --- Variation set (coordinates, rsID) ---
    variation_set = doc.get("variation_set", [])
    if variation_set and isinstance(variation_set[0], dict):
        vs = variation_set[0]

        # Coordinates
        var_locs = vs.get("variation_loc", [])
        for loc in var_locs:
            if isinstance(loc, dict):
                assembly = loc.get("assembly_name", "")
                if "GRCh38" in assembly:
                    record["chromosome"] = loc.get("chr", "")
                    record["start"] = loc.get("start", "")
                    record["stop"] = loc.get("stop", "")
                    break
                elif not record["chromosome"]:
                    record["chromosome"] = loc.get("chr", "")
                    record["start"] = loc.get("start", "")
                    record["stop"] = loc.get("stop", "")

        # rsID
        xrefs = vs.get("variation_xrefs", [])
        for xref in xrefs:
            if isinstance(xref, dict) and xref.get("db_source") == "dbSNP":
                db_id = xref.get("db_id", "")
                if db_id:
                    record["rsid"] = f"rs{db_id}" if not db_id.startswith("rs") else db_id
                break

    return record


def fetch_gene_variants(gene_name, subtype="CONTROL"):
    """Fetch all pathogenic variants for a single gene."""
    ids = esearch_clinvar(gene_name)
    time.sleep(RATE_LIMIT_DELAY)

    if not ids:
        return []

    records = esummary_clinvar(ids)

    # Filter: keep only Pathogenic / Likely pathogenic consensus classification
    filtered = []
    for r in records:
        cs = r.get("clinical_significance", "").lower()
        if "pathogenic" in cs and "conflicting" not in cs and "uncertain" not in cs:
            r["subtype"] = subtype
            if not r.get("gene"):
                r["gene"] = gene_name
            filtered.append(r)

    return filtered


def main():
    print("=" * 60, flush=True)
    print("Step 01: Fetching ClinVar Pathogenic Variants", flush=True)
    print("=" * 60, flush=True)

    # 1. Fetch scoliosis genes
    gene_subtype = load_scoliosis_genes()
    scoliosis_genes = list(gene_subtype.keys())
    print(f"\nScoliosis genes ({len(scoliosis_genes)}): {scoliosis_genes}", flush=True)

    all_variants = []
    for gene in scoliosis_genes:
        subtype = gene_subtype[gene]
        print(f"  Fetching {gene} ({subtype})...", end=" ", flush=True)
        variants = fetch_gene_variants(gene, subtype)
        print(f"{len(variants)} variants", flush=True)
        all_variants.extend(variants)
        time.sleep(RATE_LIMIT_DELAY)

    df_scoliosis = pd.DataFrame(all_variants)
    output_path = DATA_DIR / "clinvar_scoliosis.csv"
    df_scoliosis.to_csv(output_path, index=False)
    print(f"\nTotal scoliosis variants: {len(df_scoliosis)}", flush=True)
    print(f"Saved to {output_path}", flush=True)

    if not df_scoliosis.empty:
        print("\nVariants per gene:", flush=True)
        print(df_scoliosis["gene"].value_counts().to_string(), flush=True)
        print(f"\nClinical significance distribution:", flush=True)
        print(df_scoliosis["clinical_significance"].value_counts().head(10).to_string(), flush=True)

    # 2. Fetch control gene sets
    random_sets_path = DATA_DIR / "random_gene_sets.json"
    if not random_sets_path.exists():
        print("\nWarning: random_gene_sets.json not found. Run 00_random_controls.py first.")
        return

    with open(random_sets_path) as f:
        random_sets = json.load(f)

    print(f"\nFetching variants for {len(random_sets)} control sets...", flush=True)

    for i, gene_set in enumerate(random_sets):
        set_variants = []
        for gene in gene_set:
            variants = fetch_gene_variants(gene, "CONTROL")
            set_variants.extend(variants)
            time.sleep(RATE_LIMIT_DELAY)

        df_set = pd.DataFrame(set_variants) if set_variants else pd.DataFrame()
        set_path = CONTROLS_DIR / f"set_{i}.csv"
        df_set.to_csv(set_path, index=False)
        print(f"  Set {i}: {len(df_set)} variants", flush=True)

    print("\nDone! ClinVar fetch complete.", flush=True)


if __name__ == "__main__":
    main()

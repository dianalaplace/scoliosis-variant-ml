#!/usr/bin/env python3
"""
Step 01b: Fetch ClinVar variants for control gene sets.
Separate script for controls to allow parallel/resumed execution.

Uses 30 control sets (sufficient for permutation test).
Filters results to only include the target gene per query.
"""

import json
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
N_CONTROL_SETS = 30  # Sufficient for permutation test


def esearch_clinvar(gene_name, retmax=500):
    query = (
        f'{gene_name}[Gene Name] AND '
        f'("Pathogenic"[Clinical significance] OR '
        f'"Likely pathogenic"[Clinical significance])'
    )
    params = {
        "db": "clinvar", "term": query, "retmax": retmax,
        "retmode": "json", "email": EMAIL,
    }
    for attempt in range(3):
        try:
            resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"    esearch fail {gene_name}: {e}", flush=True)
                return []


def esummary_clinvar(id_list):
    if not id_list:
        return []
    records = []
    batch_size = 200
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start:start + batch_size]
        params = {
            "db": "clinvar", "id": ",".join(batch),
            "retmode": "json", "email": EMAIL,
        }
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{NCBI_BASE}/esummary.fcgi", params=params, timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                result = data.get("result", {})
                for uid in result.get("uids", []):
                    doc = result.get(str(uid), {})
                    if doc and "error" not in doc:
                        records.append(parse_doc(uid, doc))
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    esummary fail batch {start}: {e}", flush=True)
        time.sleep(RATE_LIMIT_DELAY)
    return records


def parse_doc(uid, doc):
    rec = {
        "clinvar_id": str(uid),
        "title": doc.get("title", ""),
        "variant_type": doc.get("obj_type", ""),
        "clinical_significance": "",
        "review_status": "",
        "gene": "",
        "chromosome": "", "start": "", "stop": "",
        "rsid": "",
        "molecular_consequence": "",
        "conditions": "",
    }
    germ = doc.get("germline_classification", {})
    if isinstance(germ, dict) and germ.get("description"):
        rec["clinical_significance"] = germ.get("description", "")
        rec["review_status"] = germ.get("review_status", "")
        traits = [t.get("trait_name", "") for t in germ.get("trait_set", []) if isinstance(t, dict)]
        rec["conditions"] = "; ".join(t for t in traits if t)

    genes = doc.get("genes", [])
    if genes and isinstance(genes[0], dict):
        rec["gene"] = genes[0].get("symbol", "")

    mc_list = doc.get("molecular_consequence_list", [])
    if mc_list:
        rec["molecular_consequence"] = mc_list[0] if isinstance(mc_list[0], str) else ""

    vs_list = doc.get("variation_set", [])
    if vs_list and isinstance(vs_list[0], dict):
        vs = vs_list[0]
        for loc in vs.get("variation_loc", []):
            if isinstance(loc, dict):
                if "GRCh38" in loc.get("assembly_name", ""):
                    rec["chromosome"] = loc.get("chr", "")
                    rec["start"] = loc.get("start", "")
                    rec["stop"] = loc.get("stop", "")
                    break
                elif not rec["chromosome"]:
                    rec["chromosome"] = loc.get("chr", "")
                    rec["start"] = loc.get("start", "")
                    rec["stop"] = loc.get("stop", "")
        for xref in vs.get("variation_xrefs", []):
            if isinstance(xref, dict) and xref.get("db_source") == "dbSNP":
                db_id = xref.get("db_id", "")
                if db_id:
                    rec["rsid"] = f"rs{db_id}" if not db_id.startswith("rs") else db_id
                break
    return rec


def fetch_gene_variants(gene_name):
    ids = esearch_clinvar(gene_name)
    time.sleep(RATE_LIMIT_DELAY)
    if not ids:
        return []
    records = esummary_clinvar(ids)
    # Filter: only target gene + pathogenic
    filtered = []
    for r in records:
        cs = r.get("clinical_significance", "").lower()
        gene_match = r.get("gene", "").upper() == gene_name.upper()
        if gene_match and "pathogenic" in cs and "conflicting" not in cs and "uncertain" not in cs:
            r["subtype"] = "CONTROL"
            filtered.append(r)
    return filtered


def main():
    print(f"Fetching ClinVar for {N_CONTROL_SETS} control sets...", flush=True)

    with open(DATA_DIR / "random_gene_sets.json") as f:
        random_sets = json.load(f)

    for i in range(N_CONTROL_SETS):
        set_path = CONTROLS_DIR / f"set_{i}.csv"
        # Skip if already completed
        if set_path.exists():
            df_existing = pd.read_csv(set_path)
            if len(df_existing) > 0:
                print(f"  Set {i}: already exists ({len(df_existing)} variants), skipping", flush=True)
                continue

        gene_set = random_sets[i]
        set_variants = []
        for gene in gene_set:
            variants = fetch_gene_variants(gene)
            set_variants.extend(variants)
            time.sleep(RATE_LIMIT_DELAY)

        df_set = pd.DataFrame(set_variants) if set_variants else pd.DataFrame()
        df_set.to_csv(set_path, index=False)
        print(f"  Set {i}: {len(df_set)} variants", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()

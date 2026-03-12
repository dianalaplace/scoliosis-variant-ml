#!/usr/bin/env python3
"""
Step 01: Fetch pathogenic/likely-pathogenic variants from ClinVar.

Uses NCBI Entrez (Biopython) to query ClinVar for each gene.
Processes both scoliosis genes and 100 random control sets.

Output:
  data/clinvar_scoliosis.csv        — variants for 20 scoliosis genes
  data/clinvar_controls/set_{i}.csv — variants for each of 100 random sets
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from Bio import Entrez

DATA_DIR = Path(__file__).parent / "data"
CONTROLS_DIR = DATA_DIR / "clinvar_controls"
CONTROLS_DIR.mkdir(parents=True, exist_ok=True)

# NCBI Entrez config
Entrez.email = "diana.lysenko@students.jku.at"
Entrez.api_key = None  # Optional: add NCBI API key for higher rate limits

RATE_LIMIT_DELAY = 0.34  # 3 requests/sec without API key


def load_scoliosis_genes():
    """Load gene list and subtype mapping from config."""
    with open(DATA_DIR / "genes_config.json") as f:
        config = json.load(f)
    gene_subtype = {}
    for subtype, info in config["subtypes"].items():
        for gene in info["genes"]:
            gene_subtype[gene] = subtype
    return gene_subtype


def search_clinvar(gene_name):
    """Search ClinVar for pathogenic/likely-pathogenic variants of a gene."""
    query = (
        f'{gene_name}[Gene Name] AND '
        f'("Pathogenic"[Clinical significance] OR '
        f'"Likely pathogenic"[Clinical significance])'
    )
    try:
        handle = Entrez.esearch(db="clinvar", term=query, retmax=1000)
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        return ids
    except Exception as e:
        print(f"    Search error for {gene_name}: {e}")
        return []


def fetch_clinvar_records(id_list):
    """Fetch full ClinVar records for a list of IDs."""
    if not id_list:
        return []

    records = []
    batch_size = 50
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start:start + batch_size]
        try:
            handle = Entrez.efetch(
                db="clinvar",
                id=",".join(batch),
                rettype="clinvarset",
                retmode="xml",
            )
            xml_data = handle.read()
            handle.close()

            # Parse XML
            root = ET.fromstring(xml_data)
            for clinvar_set in root.findall(".//ClinVarSet"):
                record = parse_clinvar_set(clinvar_set)
                if record:
                    records.append(record)

            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"    Fetch error for batch starting {start}: {e}")
            time.sleep(1)

    return records


def parse_clinvar_set(clinvar_set):
    """Parse a single ClinVarSet XML element into a dict."""
    try:
        record = {}

        # ClinVar accession
        rcv = clinvar_set.find(".//ReferenceClinVarAssertion")
        if rcv is None:
            return None

        # ClinVar ID
        cv_acc = rcv.find(".//ClinVarAccession")
        record["clinvar_id"] = cv_acc.get("Acc", "") if cv_acc is not None else ""

        # Title
        title_elem = rcv.find(".//Title")
        record["title"] = title_elem.text if title_elem is not None else ""

        # Clinical significance
        clin_sig = rcv.find(".//ClinicalSignificance/Description")
        record["clinical_significance"] = clin_sig.text if clin_sig is not None else ""

        # Review status
        review = rcv.find(".//ClinicalSignificance/ReviewStatus")
        record["review_status"] = review.text if review is not None else ""

        # Gene
        gene_elem = rcv.find(".//MeasureSet//Gene")
        if gene_elem is not None:
            record["gene"] = gene_elem.get("Symbol", "")
        else:
            # Try alternative path
            gene_elem = rcv.find(".//GenotypeSet//Gene")
            record["gene"] = gene_elem.get("Symbol", "") if gene_elem is not None else ""

        # Variant type
        measure = rcv.find(".//MeasureSet/Measure")
        if measure is not None:
            record["variant_type"] = measure.get("Type", "")
        else:
            record["variant_type"] = ""

        # Chromosome, start, stop
        seq_loc = rcv.find(".//MeasureSet/Measure/SequenceLocation[@Assembly='GRCh38']")
        if seq_loc is None:
            seq_loc = rcv.find(".//MeasureSet/Measure/SequenceLocation[@Assembly='GRCh37']")
        if seq_loc is None:
            seq_loc = rcv.find(".//MeasureSet/Measure/SequenceLocation")

        if seq_loc is not None:
            record["chromosome"] = seq_loc.get("Chr", "")
            record["start"] = seq_loc.get("start", "")
            record["stop"] = seq_loc.get("stop", "")
        else:
            record["chromosome"] = ""
            record["start"] = ""
            record["stop"] = ""

        # rsID from XRef
        record["rsid"] = ""
        for xref in rcv.findall(".//MeasureSet/Measure/XRef"):
            if xref.get("DB") == "dbSNP":
                record["rsid"] = f"rs{xref.get('ID', '')}"
                break

        # Molecular consequence
        record["molecular_consequence"] = ""
        for attr in rcv.findall(".//MeasureSet/Measure/AttributeSet/Attribute"):
            if attr.get("Type") == "MolecularConsequence":
                record["molecular_consequence"] = attr.text or ""
                break

        # Conditions/traits
        traits = []
        for trait in rcv.findall(".//TraitSet/Trait/Name/ElementValue"):
            if trait.text:
                traits.append(trait.text)
        record["conditions"] = "; ".join(traits)

        return record

    except Exception as e:
        print(f"    Parse error: {e}")
        return None


def fetch_gene_variants(gene_name, subtype="CONTROL"):
    """Fetch all pathogenic variants for a single gene."""
    ids = search_clinvar(gene_name)
    time.sleep(RATE_LIMIT_DELAY)

    if not ids:
        return []

    records = fetch_clinvar_records(ids)

    # Add subtype
    for r in records:
        r["subtype"] = subtype
        # Ensure gene field is set even if parsing failed
        if not r.get("gene"):
            r["gene"] = gene_name

    return records


def main():
    print("=" * 60)
    print("Step 01: Fetching ClinVar Pathogenic Variants")
    print("=" * 60)

    # 1. Fetch scoliosis genes
    gene_subtype = load_scoliosis_genes()
    scoliosis_genes = list(gene_subtype.keys())
    print(f"\nScoliosis genes ({len(scoliosis_genes)}): {scoliosis_genes}")

    all_variants = []
    for gene in scoliosis_genes:
        subtype = gene_subtype[gene]
        print(f"\n  Fetching {gene} ({subtype})...")
        variants = fetch_gene_variants(gene, subtype)
        print(f"    Found {len(variants)} variants")
        all_variants.extend(variants)
        time.sleep(RATE_LIMIT_DELAY)

    df_scoliosis = pd.DataFrame(all_variants)
    output_path = DATA_DIR / "clinvar_scoliosis.csv"
    df_scoliosis.to_csv(output_path, index=False)
    print(f"\nTotal scoliosis variants: {len(df_scoliosis)}")
    print(f"Saved to {output_path}")

    if not df_scoliosis.empty:
        print("\nVariants per gene:")
        print(df_scoliosis["gene"].value_counts().to_string())

    # 2. Fetch control gene sets
    random_sets_path = DATA_DIR / "random_gene_sets.json"
    if not random_sets_path.exists():
        print("\nWarning: random_gene_sets.json not found. Run 00_random_controls.py first.")
        return

    with open(random_sets_path) as f:
        random_sets = json.load(f)

    print(f"\nFetching variants for {len(random_sets)} control sets...")

    for i, gene_set in enumerate(random_sets):
        print(f"\n--- Control set {i} ({len(gene_set)} genes) ---")
        set_variants = []
        for gene in gene_set:
            variants = fetch_gene_variants(gene, "CONTROL")
            set_variants.extend(variants)
            time.sleep(RATE_LIMIT_DELAY)

        df_set = pd.DataFrame(set_variants) if set_variants else pd.DataFrame()
        set_path = CONTROLS_DIR / f"set_{i}.csv"
        df_set.to_csv(set_path, index=False)
        print(f"  Set {i}: {len(df_set)} variants, saved to {set_path}")

    print("\nDone! ClinVar fetch complete.")


if __name__ == "__main__":
    main()

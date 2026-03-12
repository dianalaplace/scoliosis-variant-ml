#!/usr/bin/env python3
"""
Step 00: Generate 100 random gene sets matched by transcript length.

Uses Ensembl REST API to:
1. Get transcript lengths for 20 scoliosis genes
2. Get all human protein-coding genes with transcript lengths
3. Exclude scoliosis genes, OMIM scoliosis genes, skeletal development GO genes
4. For each scoliosis gene, find a random match (±20% transcript length)
5. Repeat 100 times

Output: data/random_gene_sets.json
"""

import json
import random
import time
from pathlib import Path

import requests

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
ENSEMBL_REST = "https://rest.ensembl.org"

# Known OMIM scoliosis-associated genes to exclude from controls
OMIM_SCOLIOSIS_GENES = {
    "FBN1", "FBN2", "COL11A1", "COL11A2", "COL5A1", "COL1A1",
    "TGFBR1", "TGFBR2", "TBX6", "MESP2", "DLL3", "PAX1", "LBX1",
    "ROBO3", "NTF3", "KIF6", "CHD7", "ADGRG6", "POC5", "DSCAM",
    # Additional known scoliosis genes from literature
    "SLC12A2", "VANGL1", "WNT5A", "SHH", "PAX3", "FLNB",
    "HSPG2", "LFNG", "HES7", "RIPPLY2", "GDF6", "GDF3",
    "SOX9", "COMP", "MATN3", "COL2A1", "COL9A1", "COL9A2",
}


def load_scoliosis_genes():
    """Load scoliosis gene list from config."""
    with open(DATA_DIR / "genes_config.json") as f:
        config = json.load(f)
    genes = []
    for subtype_info in config["subtypes"].values():
        genes.extend(subtype_info["genes"])
    return genes


def get_gene_info_ensembl(gene_symbol):
    """Get gene info including canonical transcript length from Ensembl."""
    url = f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{gene_symbol}"
    params = {"content-type": "application/json", "expand": 1}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            return get_gene_info_ensembl(gene_symbol)
        if resp.status_code != 200:
            print(f"  Warning: {gene_symbol} not found (HTTP {resp.status_code})")
            return None
        data = resp.json()
        # Get canonical transcript length
        transcripts = data.get("Transcript", [])
        if not transcripts:
            return None
        # Use the canonical transcript, or the longest one
        canonical = None
        for t in transcripts:
            if t.get("is_canonical"):
                canonical = t
                break
        if not canonical:
            canonical = max(transcripts, key=lambda t: t.get("length", 0))
        return {
            "symbol": gene_symbol,
            "ensembl_id": data.get("id"),
            "transcript_length": canonical.get("length", 0),
            "biotype": data.get("biotype"),
        }
    except Exception as e:
        print(f"  Error for {gene_symbol}: {e}")
        return None


def get_all_protein_coding_genes():
    """Get all human protein-coding genes from Ensembl BioMart-like endpoint."""
    print("Fetching all human protein-coding genes from Ensembl...")
    # Use the info/assembly endpoint and then lookup genes by region
    # Alternative: use the BioMart XML query via requests

    # First get list of chromosomes
    url = f"{ENSEMBL_REST}/info/assembly/homo_sapiens"
    params = {"content-type": "application/json"}
    resp = requests.get(url, params=params, timeout=60)
    assembly = resp.json()

    chromosomes = []
    for region in assembly.get("top_level_region", []):
        if region["coord_system"] == "chromosome" and region["name"] in [
            str(i) for i in range(1, 23)
        ] + ["X", "Y"]:
            chromosomes.append(region["name"])

    all_genes = {}
    for chrom in chromosomes:
        print(f"  Chromosome {chrom}...")
        # Get genes in batches using overlap endpoint
        # Ensembl limits to 5Mb regions, so we need to chunk
        chrom_info = None
        for region in assembly["top_level_region"]:
            if region["name"] == chrom:
                chrom_info = region
                break
        if not chrom_info:
            continue

        chrom_length = chrom_info["length"]
        chunk_size = 5_000_000  # 5Mb

        for start in range(1, chrom_length, chunk_size):
            end = min(start + chunk_size - 1, chrom_length)
            url = f"{ENSEMBL_REST}/overlap/region/homo_sapiens/{chrom}:{start}-{end}"
            params = {
                "content-type": "application/json",
                "feature": "gene",
                "biotype": "protein_coding",
            }
            try:
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    time.sleep(wait)
                    resp = requests.get(url, params=params, timeout=60)
                if resp.status_code != 200:
                    continue
                genes = resp.json()
                for g in genes:
                    symbol = g.get("external_name", "")
                    if symbol and symbol not in all_genes:
                        all_genes[symbol] = {
                            "symbol": symbol,
                            "ensembl_id": g.get("gene_id", ""),
                            "transcript_length": g.get("end", 0) - g.get("start", 0),
                            "biotype": g.get("biotype", ""),
                        }
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"    Error for {chrom}:{start}-{end}: {e}")
                time.sleep(1)

    print(f"  Total protein-coding genes found: {len(all_genes)}")
    return all_genes


def get_go_skeletal_genes():
    """Get genes from GO:0001501 (skeletal system development) via QuickGO."""
    print("Fetching GO:0001501 (skeletal system development) genes...")
    url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    params = {
        "goId": "GO:0001501",
        "taxonId": "9606",
        "limit": 100,
        "page": 1,
    }
    headers = {"Accept": "application/json"}

    skeletal_genes = set()
    try:
        for page in range(1, 20):  # Max 20 pages
            params["page"] = page
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                symbol = r.get("geneProductId", "").split(":")[-1] if ":" in r.get("geneProductId", "") else ""
                gene_symbol = r.get("symbol", "")
                if gene_symbol:
                    skeletal_genes.add(gene_symbol)
            if data.get("pageInfo", {}).get("total", 0) <= page * 100:
                break
            time.sleep(0.2)
    except Exception as e:
        print(f"  Warning: Could not fetch GO genes: {e}")

    print(f"  Found {len(skeletal_genes)} skeletal development genes")
    return skeletal_genes


def match_random_gene(target_length, gene_pool, tolerance=0.2):
    """Find a random gene within ±tolerance of target transcript length."""
    lower = target_length * (1 - tolerance)
    upper = target_length * (1 + tolerance)
    candidates = [
        g for g in gene_pool
        if lower <= g["transcript_length"] <= upper
    ]
    if not candidates:
        # Relax tolerance
        candidates = [
            g for g in gene_pool
            if target_length * 0.5 <= g["transcript_length"] <= target_length * 1.5
        ]
    if not candidates:
        return random.choice(gene_pool)
    return random.choice(candidates)


def main():
    print("=" * 60)
    print("Step 00: Generating Random Control Gene Sets")
    print("=" * 60)

    # 1. Load scoliosis genes
    scoliosis_genes = load_scoliosis_genes()
    print(f"\nScoliosis genes ({len(scoliosis_genes)}): {scoliosis_genes}")

    # 2. Get transcript lengths for scoliosis genes
    print("\nFetching scoliosis gene transcript lengths...")
    scol_gene_info = {}
    for gene in scoliosis_genes:
        info = get_gene_info_ensembl(gene)
        if info:
            scol_gene_info[gene] = info
            print(f"  {gene}: {info['transcript_length']} bp")
        time.sleep(0.34)

    # 3. Get all protein-coding genes
    all_genes = get_all_protein_coding_genes()

    # 4. Get exclusion sets
    skeletal_genes = get_go_skeletal_genes()
    exclude_set = OMIM_SCOLIOSIS_GENES | skeletal_genes

    # 5. Build gene pool (exclude scoliosis + skeletal genes)
    gene_pool = [
        info for symbol, info in all_genes.items()
        if symbol not in exclude_set
        and info["transcript_length"] > 0
    ]
    print(f"\nGene pool after exclusions: {len(gene_pool)} genes")

    # 6. Generate 100 random matched sets
    print("\nGenerating 100 random gene sets...")
    random_sets = []
    for i in range(100):
        gene_set = []
        used_symbols = set()
        for scol_gene in scoliosis_genes:
            target = scol_gene_info.get(scol_gene, {}).get("transcript_length", 5000)
            # Filter out already-used genes
            available = [g for g in gene_pool if g["symbol"] not in used_symbols]
            matched = match_random_gene(target, available)
            gene_set.append(matched["symbol"])
            used_symbols.add(matched["symbol"])
        random_sets.append(gene_set)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/100 sets")

    # 7. Save
    output_path = DATA_DIR / "random_gene_sets.json"
    with open(output_path, "w") as f:
        json.dump(random_sets, f, indent=2)
    print(f"\nSaved {len(random_sets)} gene sets to {output_path}")
    print(f"Each set has {len(random_sets[0])} genes")
    print(f"Example set 0: {random_sets[0][:5]}...")


if __name__ == "__main__":
    main()

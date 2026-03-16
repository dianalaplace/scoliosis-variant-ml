# scoliosis-variant-ml

> ML pipeline for identifying pathogenic variant signatures in scoliosis-associated genes.

Replacement test# scoliosis-variant-ml

> ML pipeline for identifying pathogenic variant signatures in scoliosis-associated genes using ClinVar, gnomAD, and unsupervised learning.

## Overview

This project investigates whether scoliosis-associated genes carry a **distinct pathogenic variant signature** compared to random gene sets. The pipeline integrates multi-source genomic data, engineers variant-level and gene-level features, and applies unsupervised learning (UMAP + KMeans) to characterize the variant landscape.

**Key question:** Do known scoliosis genes cluster separately from random genes based on their ClinVar/gnomAD variant profiles?

## Pipeline

```
00_random_controls.py      Generate 100 random control gene sets
01_fetch_clinvar.py        Fetch pathogenic/likely-pathogenic variants (ClinVar, NCBI Entrez)
01b_fetch_controls.py      Fetch ClinVar variants for all control sets
02_fetch_gnomad.py         Fetch population allele frequencies (gnomAD)
03_feature_engineering.py  Variant-level feature matrix construction
04_gene_level_features.py  Aggregate features per gene
05_clustering.py           UMAP dimensionality reduction + KMeans clustering
06_enrichment.py           Enrichment analysis vs. control distribution
07_figures.py              Publication-ready figure generation
```

## Data Sources

| Source | Description |
|--------|-------------|
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | Pathogenic / likely-pathogenic variants for 20 scoliosis genes |
| [gnomAD](https://gnomad.broadinstitute.org/) | Population allele frequencies for variant filtering |

## Methods

- **Variant fetching:** NCBI Entrez API (esearch + esummary, JSON)
- **Feature engineering:** variant-level and gene-level features (pathogenicity scores, allele frequencies, variant counts)
- **Dimensionality reduction:** UMAP (n_neighbors=10, min_dist=0.1)
- **Clustering:** KMeans (k=2-6, best k by silhouette score), StandardScaler normalization
- **Controls:** 100 random gene sets matched by gene count for empirical null distribution

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
bash run_local.sh
```

Or step by step:

```bash
python 00_random_controls.py
python 01_fetch_clinvar.py
python 01b_fetch_controls.py
python 02_fetch_gnomad.py
python 03_feature_engineering.py
python 04_gene_level_features.py
python 05_clustering.py
python 06_enrichment.py
python 07_figures.py
```

## Repository Structure

```
scoliosis-variant-ml/
├── data/                  # Raw and processed variant data (CSV)
├── results/               # Clustering outputs, enrichment scores, figures
├── 00_random_controls.py
├── 01_fetch_clinvar.py
├── 01b_fetch_controls.py
├── 02_fetch_gnomad.py
├── 03_feature_engineering.py
├── 04_gene_level_features.py
├── 05_clustering.py
├── 06_enrichment.py
├── 07_figures.py
├── requirements.txt
└── run_local.sh
```

## Status

Active development — pipeline complete, poster figures in progress.

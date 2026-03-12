#!/bin/bash
# Run the full pipeline locally.
# Step 02 will auto-detect gnomAD API and use real data if reachable.
#
# Usage:
#   ./run_local.sh           # full pipeline
#   ./run_local.sh --gnomad  # only re-run gnomAD + downstream (after ClinVar is done)

set -e

echo "=== Scoliosis Variant ML Pipeline ==="
echo ""

# Install dependencies
pip install -q -r requirements.txt

if [ "$1" == "--gnomad" ]; then
    echo "Re-running from gnomAD step..."
    python 02_fetch_gnomad.py
    python 03_feature_engineering.py
    python 04_gene_level_features.py
    python 05_clustering.py
    python 06_enrichment.py
    python 07_figures.py
    echo ""
    echo "Done! Check results/figures/ for updated poster figures."
    exit 0
fi

echo "Step 00: Random controls (~5 min)..."
python 00_random_controls.py

echo ""
echo "Step 01: ClinVar scoliosis genes (~2 min)..."
python 01_fetch_clinvar.py

echo ""
echo "Step 01b: ClinVar control sets (~30 min)..."
python 01b_fetch_controls.py

echo ""
echo "Step 02: gnomAD allele frequencies (~15 min with real API)..."
python 02_fetch_gnomad.py

echo ""
echo "Steps 03-07: Feature engineering, clustering, figures..."
python 03_feature_engineering.py
python 04_gene_level_features.py
python 05_clustering.py
python 06_enrichment.py
python 07_figures.py

echo ""
echo "=== Pipeline complete! ==="
echo "Figures saved to results/figures/"
echo ""

# Check if gnomAD data is real or simulated
python -c "
import pandas as pd
df = pd.read_csv('data/variants_with_gnomad.csv')
src = df.get('gnomad_source', pd.Series(['unknown'])).iloc[0]
if src == 'SIMULATED':
    print('⚠️  WARNING: gnomAD AF is SIMULATED. Run with real internet for publication data.')
    print('   Quick fix: ./run_local.sh --gnomad')
else:
    print('✓ gnomAD data is from real API — ready for publication!')
"

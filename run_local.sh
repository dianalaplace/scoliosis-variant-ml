#!/bin/bash
# Run the full pipeline locally.
# Step 02 will auto-detect gnomAD API and use real data if reachable.
#
# Usage:
#   ./run_local.sh           # full pipeline from scratch
#   ./run_local.sh --gnomad  # only re-run gnomAD + downstream
#   ./run_local.sh --from 04 # re-run from step 04 onward

set -e

cd "$(dirname "$0")"

echo "=== Scoliosis Variant ML Pipeline ==="
echo ""

# Install dependencies
pip install -q -r requirements.txt

if [ "$1" == "--gnomad" ]; then
    echo "Re-running from gnomAD step..."

    # Check that prerequisite data exists
    if [ ! -f data/clinvar_scoliosis.csv ]; then
        echo "ERROR: data/clinvar_scoliosis.csv not found!"
        echo "You need to run the full pipeline first:"
        echo "  ./run_local.sh"
        exit 1
    fi

    # Ensure control sets exist
    mkdir -p data/clinvar_controls
    CTRL_COUNT=$(find data/clinvar_controls -name "set_*.csv" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "Control sets found: $CTRL_COUNT"

    if [ "$CTRL_COUNT" -eq 0 ]; then
        echo ""
        echo "No control gene sets found. Need to generate them first."

        if [ ! -f data/random_gene_sets.json ]; then
            echo "Step 00: Generating random gene sets..."
            python 00_random_controls.py
        fi

        echo "Step 01b: Fetching ClinVar for control gene sets (~30 min)..."
        python 01b_fetch_controls.py

        CTRL_COUNT=$(find data/clinvar_controls -name "set_*.csv" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "Control sets generated: $CTRL_COUNT"

        if [ "$CTRL_COUNT" -eq 0 ]; then
            echo "ERROR: Failed to generate control sets. Check 01b_fetch_controls.py output."
            exit 1
        fi
    fi

    echo ""
    echo "Step 02: gnomAD (~15 min for scoliosis, ~30 min for $CTRL_COUNT control sets)..."
    python 02_fetch_gnomad.py

    echo ""
    echo "Steps 03-07..."
    python 03_feature_engineering.py
    python 04_gene_level_features.py
    python 05_clustering.py
    python 06_enrichment.py
    python 07_figures.py
    echo ""
    echo "Done! Check results/figures/ for updated poster figures."
    exit 0
fi

if [ "$1" == "--from" ]; then
    STEP=${2:-04}
    echo "Re-running from step $STEP..."
    for s in 04 05 06 07; do
        if [ "$(printf '%02d' "$STEP")" -le "$(printf '%02d' "$s")" ] 2>/dev/null || [ "$STEP" -le "$s" ] 2>/dev/null; then
            echo ""
            echo "Running step ${s}..."
            python "$(printf '%02d' "$s")"_*.py
        fi
    done
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

#!/bin/bash
# =============================================================================
# model.sh
# -----------------------------------------------------------------------------
# Runs the whole XAI-on-RDF pipeline in one go:
#   1. Data analysis                (notebooks/1__analysis.py)
#   2. Graph pre-processing         (notebooks/2__Graph_PreProcessing.py)
#   3. R-GCN model training         (notebooks/3__model_train.py)
#   4. GNNExplainer explainability  (notebooks/4__Explainability.py)
#   5. Captum explainability + report (notebooks/5__Explainability_Captus.py)
#
# All outputs are written under  outputs/  exactly like the repo:
#   outputs/plots/Analysis/                              (3 analysis plots)
#   outputs/plots/Model_Explainability/GNN_Explainer/    (2 GNNExplainer plots)
#   outputs/plots/Model_Explainability/Captus_Explainer/ (2 Captum plots)
#   outputs/reports/Evaluate_Explaination.txt            (explanation report)
#
# The generated plots are opened at the end so they are shown to the user.
# =============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Resolve the repo root from this script's location (CWD-independent)
# ----------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Python interpreter (has torch / torch-geometric / captum / rdflib installed)
PY="/usr/local/bin/python3"

# src/ holds the shared library modules imported by stages 3-5
export PYTHONPATH="$ROOT/src"
# Force a non-interactive matplotlib backend so plots render to files headlessly
export MPLBACKEND="Agg"

# ----------------------------------------------------------------------------
# Recreate the output folders (exact repo structure) even if they were deleted
# ----------------------------------------------------------------------------
mkdir -p "$ROOT/outputs/plots/Analysis"
mkdir -p "$ROOT/outputs/plots/Model_Explainability/GNN_Explainer"
mkdir -p "$ROOT/outputs/plots/Model_Explainability/Captus_Explainer"
mkdir -p "$ROOT/outputs/reports"

# ----------------------------------------------------------------------------
# Run the pipeline
# ----------------------------------------------------------------------------
echo "==> [1/5] Data Analysis"
"$PY" "$ROOT/notebooks/1__analysis.py"

echo "==> [2/5] Graph Pre-Processing"
"$PY" "$ROOT/notebooks/2__Graph_PreProcessing.py"

echo "==> [3/5] Model Training (R-GCN)"
"$PY" "$ROOT/notebooks/3__model_train.py"

echo "==> [4/5] Explainability (GNNExplainer)"
"$PY" "$ROOT/notebooks/4__Explainability.py"

echo "==> [5/5] Explainability (Captum) + Report"
"$PY" "$ROOT/notebooks/5__Explainability_Captus.py"

# ----------------------------------------------------------------------------
# Show the generated plots (analysis, GNN explainer, Captum explainer)
# ----------------------------------------------------------------------------
echo ""
echo "==> Opening generated plots ..."
open "$ROOT/outputs/plots/Analysis/"*.png \
     "$ROOT/outputs/plots/Model_Explainability/GNN_Explainer/"*.png \
     "$ROOT/outputs/plots/Model_Explainability/Captus_Explainer/"*.png 2>/dev/null || true

echo ""
echo "============================================================"
echo " Pipeline complete. Outputs are in: $ROOT/outputs"
echo "   - Analysis plots        : outputs/plots/Analysis/"
echo "   - GNN explainer plots    : outputs/plots/Model_Explainability/GNN_Explainer/"
echo "   - Captum explainer plots : outputs/plots/Model_Explainability/Captus_Explainer/"
echo "   - Explanation report     : outputs/reports/Evaluate_Explaination.txt"
echo "============================================================"

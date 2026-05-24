#!/bin/bash
set -e

echo "========================================"
echo "Real Dataset Pipeline"
echo "========================================"

echo ""
echo "[1/4] Downloading real datasets..."
python scripts/auto_download.py

echo ""
echo "[2/4] Ingesting and processing data..."
python scripts/data_ingestion.py

echo ""
echo "[3/4] Seeding ChromaDB..."
python scripts/seed_chroma.py

echo ""
echo "[4/4] Running baseline evaluation..."
python scripts/baseline_eval.py

echo ""
echo "========================================"
echo "Pipeline complete."
echo "Results saved to results/"
echo "========================================"

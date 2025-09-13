#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

python -m src.cli.commands build-index \
  --sample-pdf data/sample/2._Sample_Inspection_Report.pdf \
  --labels-xlsx data/sample/3._Risk_Severity.xlsx \
  --out ./.cache/rsrisk_index

python -m src.cli.commands predict \
  --pdf data/new/4._New_Inspection_Report.pdf \
  --index ./.cache/rsrisk_index \
  --out outputs/new_report_predictions.csv

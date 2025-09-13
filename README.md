# rightship-risk-classifier

Extract deficiencies from a RightShip inspection PDF and classify risk (High/Medium/Low) using LangChain with RAG few-shot + rule-based guardrails, plus evaluation utilities.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

Place files:
- `data/sample/3._Risk_Severity.xlsx`
- `data/new/4._New_Inspection_Report.pdf`

Build index:
```bash
python -m rsrisk.cli build-index \
  --labels-xlsx data/sample/3._Risk_Severity.xlsx \
  --out ./.cache/rsrisk_index.faiss
```

Predict on new report:
```bash
python -m rsrisk.cli predict \
  --pdf data/new/4._New_Inspection_Report.pdf \
  --index ./.cache/rsrisk_index.faiss \
  --out outputs/new_report_predictions.csv
```

Evaluate on sample (optional):
```bash
python -m rsrisk.cli eval \
  --sample-pdf data/sample/2._Sample_Inspection_Report.pdf \
  --labels-xlsx data/sample/3._Risk_Severity.xlsx \
  --index ./.cache/rsrisk_index.faiss
```
## API + Web Demo

Start the FastAPI server:

```bash
./scripts/serve_api.sh
```

Open your browser to `http://localhost:8000` to see a simple upload form. Upload a PDF inspection report to receive JSON with one item per extracted deficiency including `risk_llm`, `risk_final`, and `rationale`.

Notes:
- The server will look for an existing FAISS index at `./.cache/rsrisk_index.faiss`. If not found, it will auto-build an index from `data/sample/2._Sample_Inspection_Report.pdf` + `data/sample/3._Risk_Severity.xlsx` if present.
- Ensure OpenAI environment variables are set (see `.env.example`).

## Gradio UI (table output)

Run the Gradio app:

```bash
./scripts/serve_gradio.sh
```

Prerequisite: the FastAPI server must be running (the Gradio UI calls the API under the hood). In a separate terminal, start:

```bash
./scripts/serve_api.sh
```

Then open `http://localhost:7860`, upload a PDF, and you will see a table of deficiencies with columns: `deficiency`, `root_cause`, `corrective`, `preventive`, `risk_llm`, `risk_final`, `rationale`.

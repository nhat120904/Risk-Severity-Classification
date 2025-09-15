# rightship-risk-classifier

Extract deficiencies from a RightShip inspection PDF and classify risk (High/Medium/Low) using LangChain with RAG few-shot + rule-based guardrails, plus evaluation utilities.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Create your env file (see Environment below)
# cp .env.example .env
```

 
## API + Web Demo

Start the FastAPI server:

```bash
./scripts/serve_api.sh
```

API base: `http://localhost:8000`

Endpoints:
- `GET /v1/health` → `{ "status": "ok" }`
- `POST /v1/classify` (multipart form, field `pdf`) → JSON with items: `deficiency`, `root_cause`, `corrective`, `preventive`, `risk_llm`, `risk_final`, `rationale`, `evidence`, plus `rag_used` and an optional `notice` message.
  - Query params: `model`, `use_rag` (true/false), `embed_model`, `excel` (true to return an Excel file instead of JSON).

Example (JSON response):
```bash
curl -s -F pdf=@data/new/4._New_Inspection_Report.pdf \
  "http://localhost:8000/v1/classify?use_rag=true" | jq .
```

Example (Excel download):
```bash
curl -L -o outputs/new_report_predictions.xlsx \
  -F pdf=@data/new/4._New_Inspection_Report.pdf \
  "http://localhost:8000/v1/classify?excel=true"
```

Notes:
- The server keeps an index per embedding model at `./.cache/index__{embed_model}`. If not found, it auto-builds from `data/sample/2._Sample_Inspection_Report.pdf` + `data/sample/3._Risk_Severity.xlsx` when present.
- If no index and no sample data are available, the API will proceed without RAG (few-shot examples) by default and include a `notice` in the response. If you explicitly set `use_rag=true`, the API returns HTTP 400 with guidance.
- Ensure OpenAI environment variables are set (see Environment below).

## Sample Data and Vector Index

To enable RAG examples, place the following files:

- `data/sample/2._Sample_Inspection_Report.pdf`
- `data/sample/3._Risk_Severity.xlsx`

On first request, an index will be created at `./.cache/index__{embed_model}`. Subsequent requests reuse it.

Alternatively, you can pre-create an index by running a classification once with the sample files present, or by writing your own builder that calls `build_index_from_sample(pdf_path, labels_xlsx, embed_model)` and saves via `save_index(vs, ".cache/index__{embed_model}")`.

Without these files or an existing index, the service will still work, but `rag_used=false` and the UI will show a notice indicating that RAG is unavailable.

## Gradio UI (table output)

Run the Gradio app:

```bash
./scripts/serve_gradio.sh
```

Prerequisite: the FastAPI server must be running (the Gradio UI calls the API under the hood). In a separate terminal, start:

```bash
./scripts/serve_api.sh
```

Then open `http://localhost:7860`, upload a PDF, and you will see a table of deficiencies with columns: `deficiency`, `root_cause`, `corrective`, `preventive`, `risk_llm`, `risk_final`, `rationale`, `evidence`. A downloadable Excel with `Deficiency`/`Risk` is also produced in `outputs/`.

Config for UI:
- `RSRISK_API_BASE` (default `http://localhost:8000`)
- `UI_MODEL_CHOICES` (comma-separated, defaults to `gpt-4.1,gpt-4.1-mini,gpt-5,gpt-5-mini,gpt-5-nano`)
- `EMBED_MODEL` (e.g., `text-embedding-3-large`)

## Environment

Create `.env` in the project root and set at least:

```
OPENAI_API_KEY=sk-...
# Optional
OPENAI_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-large
UI_MODEL_CHOICES=gpt-4.1,gpt-4.1-mini,gpt-5,gpt-5-mini,gpt-5-nano

# Optional: Enable LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="your LangSmith endpoint"
LANGSMITH_API_KEY="your LangSmith API key"
LANGSMITH_PROJECT="your LangSmith project name"

# If using Azure OpenAI, also set as needed:
# OPENAI_API_TYPE=azure
# OPENAI_API_BASE=...
# OPENAI_API_VERSION=...
# OPENAI_DEPLOYMENT_NAME=...
```

## Evaluation

Run the evaluation script to classify the provided sample PDF and compare predictions against the true labels. It prints macro metrics and saves per-item predictions.

Quick start:
```bash
python scripts/evaluate_sample.py
```

Options:
```bash
python scripts/evaluate_sample.py \
  --pdf data/sample/2._Sample_Inspection_Report.pdf \
  --labels data/sample/3._Risk_Severity.xlsx \
  --rag \
  --model gpt-4.1-mini
```
- `--pdf`/`--labels`: override input paths
- `--rag`: use RAG few-shot examples
- `--model`: default model for OCR/extraction/classification


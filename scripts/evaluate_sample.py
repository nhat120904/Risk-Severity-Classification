#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score

# Ensure local imports work when run from repo root
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.services.ocr_service import load_pdf_text
from src.services.extraction_service import extract_records
from src.services.retrieval_service import build_index_from_sample
from src.services.classification_service import classify_record
from src.services.guardrails_service import apply_guardrails
from src.core.config import settings


DEFAULT_PDF = REPO_ROOT / "data/sample/2._Sample_Inspection_Report.pdf"
DEFAULT_XLSX = REPO_ROOT / "data/sample/3._Risk_Severity.xlsx"


def run_eval(pdf_path: Path, labels_xlsx: Path, use_rag: bool = True, model: str | None = None) -> int:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not labels_xlsx.exists():
        raise FileNotFoundError(f"Labels XLSX not found: {labels_xlsx}")

    print(f"Loading PDF text from: {pdf_path}")
    full_text = load_pdf_text(str(pdf_path), model_name=model)

    print("Extracting records (deficiency, root_cause, corrective, preventive)...")
    recs = extract_records(full_text, model_name=model, provider=("openai"))
    if not recs:
        print("No records extracted; cannot evaluate.")
        return 2

    print("Building retrieval index from sample + labels for RAG examples...")
    index = build_index_from_sample(str(pdf_path), str(labels_xlsx), embed_model=settings.embed_model)

    # Load ground-truth labels
    df = pd.read_excel(labels_xlsx)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Expect columns like: Deficiency (id), Risk (High/Medium/Low)
    if "risk" not in df.columns:
        raise ValueError("Expected a 'Risk' column in labels Excel.")

    id_to_label = dict(zip(df.get("deficiency", range(1, len(df) + 1)), df["risk"].astype(str)))

    y_true, y_pred = [], []
    items = []
    for i, rec in enumerate(recs, start=1):
        # True label by order (1-based), fallback to None if missing
        true_lbl = id_to_label.get(i, "None")
        out = classify_record(rec, index, model_name=model, provider=("openai"), use_rag=use_rag)
        final_lbl = apply_guardrails(rec, out.risk).value

        y_true.append(true_lbl)
        y_pred.append(final_lbl)

        items.append({
            "Deficiency": i,
            "true": true_lbl,
            "pred": final_lbl,
        })

    print("\nClassification report (macro avg):")

    labels = ["High", "Medium", "Low"]
    print(classification_report(y_true, y_pred, labels=labels, digits=3))

    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro")
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")

    print(f"Macro Recall: {macro_recall:.3f}")
    print(f"Macro F1:     {macro_f1:.3f}")

    # Save predictions for inspection
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sample_predictions.xlsx"
    pd.DataFrame(items).to_excel(out_path, index=False)
    print(f"Saved per-item predictions to: {out_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate risk classification on the sample PDF vs Excel labels.")
    parser.add_argument("--pdf", type=str, default=str(DEFAULT_PDF), help="Path to sample PDF")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_XLSX), help="Path to labels Excel")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG few-shot examples")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model name for OCR/extraction/classification")
    args = parser.parse_args()

    use_rag = not args.no_rag
    return_code = run_eval(Path(args.pdf), Path(args.labels), use_rag=use_rag, model=args.model)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()

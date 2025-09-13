import argparse
import json
from pathlib import Path
import pandas as pd
from rich import print

from src.services.ocr_service import load_pdf_text
from src.services.extraction_service import extract_records
from src.services.retrieval_service import build_index_from_sample, save_index, load_index
from src.services.classification_service import classify_record
from src.services.guardrails_service import apply_guardrails


CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)


def cmd_build_index(args):
    vs = build_index_from_sample(args.sample_pdf, args.labels_xlsx)
    save_index(vs, args.out)
    print(f"[green]Saved index to {args.out}")


def cmd_predict(args):
	text = load_pdf_text(args.pdf)
	recs = extract_records(text, model_name=args.extract_model, provider=args.extract_provider)
	vs = load_index(args.index)
	rows = []
	for rec in recs:
		out = classify_record(
			rec,
			vs,
			use_rag=args.use_rag,
			model_name=args.classify_model,
			provider=args.classify_provider,
		)
		final = apply_guardrails(rec, out.risk)
		rows.append({
			"deficiency": rec.deficiency,
			"root_cause": rec.root_cause,
			"corrective": rec.corrective,
			"preventive": rec.preventive,
			"risk_llm": out.risk.value,
			"risk_final": final.value,
			"rationale": out.rationale,
			"evidence": out.evidence,
		})
	df = pd.DataFrame(rows)
	if str(args.out).lower().endswith(".xlsx"):
		export_df = pd.DataFrame({
			"Deficiency": list(range(1, len(rows) + 1)),
			"Risk": pd.Series([r.get("risk_final") for r in rows]).astype(str),
		})
		export_df.to_excel(args.out, index=False)
	else:
		if "evidence" in df.columns:
			df = df.copy(); df["evidence"] = df["evidence"].apply(lambda x: json.dumps(x, ensure_ascii=False))
		df.to_csv(args.out, index=False)
	print(f"[green]Wrote predictions → {args.out}")


def cmd_eval(args):
    from src.eval.metrics import evaluate

    if args.index and Path(args.index).exists():
        vs = load_index(args.index)
    else:
        vs = build_index_from_sample(args.sample_pdf, args.labels_xlsx)

    text = load_pdf_text(args.sample_pdf)
    recs = extract_records(text, model_name=args.extract_model, provider=args.extract_provider)

    gold_df = pd.read_excel(args.labels_xlsx)
    gold_df.columns = [str(c).strip().lower() for c in gold_df.columns]
    gold_map = dict(zip(gold_df["deficiency"].astype(int), gold_df["risk"].astype(str)))

    preds, labels, report_rows = [], [], []
    for i, rec in enumerate(recs, start=1):
        out = classify_record(
            rec,
            vs,
            use_rag=args.use_rag,
            model_name=args.classify_model,
            provider=args.classify_provider,
        )
        final = apply_guardrails(rec, out.risk)
        gold_lbl = gold_map.get(i, "Low")
        preds.append(final.value)
        labels.append(gold_lbl)
        report_rows.append({
            "index": i,
            "deficiency": rec.deficiency,
            "pred": final.value,
            "gold": gold_lbl,
            "llm": out.risk.value,
            "rationale": out.rationale,
        })

    metrics = evaluate(preds, labels)
    print("[bold]Eval metrics:"); print(metrics)
    pd.DataFrame(report_rows).to_csv(OUT_DIR / "eval_report.csv", index=False)
    print(f"[yellow]Detailed report → {OUT_DIR / 'eval_report.csv'}")


def build_argparser():
	ap = argparse.ArgumentParser(prog="rsrisk")
	sub = ap.add_subparsers(dest="cmd", required=True)

	ap_b = sub.add_parser("build-index")
	ap_b.add_argument("--sample-pdf", required=True) 
	ap_b.add_argument("--labels-xlsx", required=True)
	ap_b.add_argument("--out", required=True)
	ap_b.set_defaults(func=cmd_build_index)

	ap_p = sub.add_parser("predict")
	ap_p.add_argument("--pdf", required=True)
	ap_p.add_argument("--index", required=True)
	ap_p.add_argument("--out", required=True)
	ap_p.add_argument("--use-rag", action="store_true", help="Use RAG few-shot examples (default from env)")
	ap_p.add_argument("--extract-model", required=False, help="LLM model for extraction")
	ap_p.add_argument("--extract-provider", required=False, help="LLM provider for extraction")
	ap_p.add_argument("--classify-model", required=False, help="LLM model for classification")
	ap_p.add_argument("--classify-provider", required=False, help="LLM provider for classification")
	ap_p.set_defaults(func=cmd_predict)

	ap_e = sub.add_parser("eval")
	ap_e.add_argument("--sample-pdf", required=True)
	ap_e.add_argument("--labels-xlsx", required=True)
	ap_e.add_argument("--index", required=False)
	ap_e.add_argument("--use-rag", action="store_true", help="Use RAG few-shot examples (default from env)")
	ap_e.add_argument("--extract-model", required=False, help="LLM model for extraction")
	ap_e.add_argument("--extract-provider", required=False, help="LLM provider for extraction")
	ap_e.add_argument("--classify-model", required=False, help="LLM model for classification")
	ap_e.add_argument("--classify-provider", required=False, help="LLM provider for classification")
	ap_e.set_defaults(func=cmd_eval)

	return ap


def main():
	ap = build_argparser()
	args = ap.parse_args()
	args.func(args)



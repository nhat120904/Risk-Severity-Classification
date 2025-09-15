from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("RSRISK_API_BASE", "http://localhost:8000")
MODEL_CHOICES = [m for m in os.getenv("UI_MODEL_CHOICES", "gpt-5,gpt-5-mini,gpt-5-nano").split(",") if m]
CUSTOM_CSS = """
/* Wrap table and allow scroll on overflow */
.wrap-table { max-height: 70vh; overflow: auto; }

/* Fix table layout so columns have fixed widths and wrapped text */
.wrap-table table { table-layout: fixed !important; width: 100%; }

/* Allow line wrapping in cells */
.wrap-table th, .wrap-table td {
  white-space: normal !important;      /* instead of nowrap/pre */
  overflow-wrap: anywhere;             /* wrap long words too */
  word-break: break-word;              /* fallback */
  line-height: 1.35;
}

/* Set width per column */
.wrap-table thead th:nth-child(1),
.wrap-table tbody td:nth-child(1) { max-width: 32ch; }  /* deficiency */
.wrap-table thead th:nth-child(2),
.wrap-table tbody td:nth-child(2) { max-width: 38ch; }  /* root_cause */
.wrap-table thead th:nth-child(3),
.wrap-table tbody td:nth-child(3) { max-width: 30ch; }  /* corrective */
.wrap-table thead th:nth-child(4),
.wrap-table tbody td:nth-child(4) { max-width: 30ch; }  /* preventive */
.wrap-table thead th:nth-child(5),
.wrap-table tbody td:nth-child(5) { width: 12ch; text-align: center; } /* risk_llm */
.wrap-table thead th:nth-child(6),
.wrap-table tbody td:nth-child(6) { width: 12ch; text-align: center; } /* risk_final */
.wrap-table thead th:nth-child(7),
.wrap-table tbody td:nth-child(7) { max-width: 36ch; }  /* rationale */

/* Sticky header (optional) */
.wrap-table thead th {
  position: sticky; top: 0; z-index: 1;
  background: var(--block-background-fill);
}
"""

def classify(
    pdf_path: str | Path,
    model_name: str | None = None,
    use_rag: bool = False,
    embed_model: str | None = None,
):
    # Call API once and build DataFrame
    url = f"{API_BASE}/v1/classify"
    params = {"use_rag": str(use_rag).lower()}
    if model_name:
        params["model"] = model_name
    if embed_model:
        params["embed_model"] = embed_model
    try:
        with open(pdf_path, "rb") as f:
            files = {"pdf": (Path(pdf_path).name, f, "application/pdf")}
            resp = requests.post(url, files=files, params=params, timeout=180)
        resp.raise_for_status()
        data = resp.json(); items = data.get("items", [])
        df = pd.DataFrame(items)
        notice = data.get("notice") or ""
    except requests.HTTPError as e:
        try:
            err = e.response.json()
            msg = err.get("detail") or str(e)
        except Exception:
            msg = str(e)
        return pd.DataFrame(), None, f"⚠️ {msg}"

    # Create Excel locally from JSON results (use final risk labels)
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(pdf_path).stem + "_predictions.xlsx")
    try:
        risks = df["risk_final"] if "risk_final" in df.columns else pd.Series(dtype=str)
        excel_df = pd.DataFrame({
            "Deficiency": list(range(1, len(risks) + 1)),
            "Risk": risks.astype(str).tolist(),
        })
        excel_df.to_excel(out_path, index=False)
    except Exception as e:
        print("Error creating Excel file: ", e)
        # Fallback: dump all rows to Excel if structure differs
        df.to_excel(out_path, index=False)
    return df, str(out_path), notice


with gr.Blocks(title="RightShip Risk Classifier", css=CUSTOM_CSS) as demo:
    gr.Markdown("## RightShip Risk Classifier\nUpload an inspection PDF to get per-deficiency risk classification.")
    with gr.Row():
        inp = gr.File(label="PDF Report", file_types=[".pdf"], type="filepath")
        model_in = gr.Dropdown(
            label="LLM Model",
            choices=MODEL_CHOICES,
            value=(MODEL_CHOICES[0] if MODEL_CHOICES else None),
        )
        use_rag_in = gr.Checkbox(label="Use RAG examples", value=False)
        embed_model_in = gr.Dropdown(
            label="Embedding Model",
            choices=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            value=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        )
    with gr.Row():
        pass
    with gr.Row():
        btn = gr.Button("Classify")
    with gr.Row():
        out = gr.Dataframe(
            headers=[
                "Deficiency", "Root Cause", "Corrective", "Preventive",
                "Risk (LLM)", "Risk (Final)", "Rationale", "Evidence"
            ],
            label="Results",
            interactive=False,
            elem_classes=["wrap-table"],
        )

    btn.click(
        fn=classify,
        inputs=[
            inp,
            model_in,
            use_rag_in,
            embed_model_in,
        ],
        outputs=[out, gr.File(label="Download Output"), gr.Markdown(label="Notice")],
    )


def main():
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()



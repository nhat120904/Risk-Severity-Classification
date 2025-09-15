from __future__ import annotations

import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.core.config import settings
from src.api.schemas import HealthResponse, ClassifyResponse, ClassifiedItem
from src.services.ocr_service import load_pdf_text
from src.services.extraction_service import extract_records
from src.services.retrieval_service import build_index_from_sample, load_index, save_index
from src.services.classification_service import classify_record
from src.services.guardrails_service import apply_guardrails


APP_CACHE = Path(".cache"); APP_CACHE.mkdir(exist_ok=True)


def _index_dir_for_embed_model(embed_model: str) -> Path:
    safe = embed_model.replace("/", "_")
    return APP_CACHE / f"index__{safe}"


DEFAULT_LABELS_XLSX = Path("data/sample/3._Risk_Severity.xlsx")
DEFAULT_SAMPLE_PDF = Path("data/sample/2._Sample_Inspection_Report.pdf")


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/classify", response_model=ClassifyResponse)
async def classify_pdf(
    pdf: UploadFile = File(...),
    model: str | None = Query(default=None),
    use_rag: bool | None = Query(default=None, description="Use RAG few-shot examples"),
    embed_model: str | None = Query(default=None, description="Embedding model for vector index"),
    excel: bool | None = Query(default=False, description="Return Excel file (Deficiency/Risk) instead of JSON"),
) -> ClassifyResponse | StreamingResponse:
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    tmp_path = APP_CACHE / f"upload_{uuid.uuid4().hex}.pdf"
    content = await pdf.read()
    tmp_path.write_bytes(content)

    try:
        em = embed_model or settings.embed_model
        index_path = _index_dir_for_embed_model(em)
        notice: str | None = None
        # Determine effective RAG usage based on query or settings
        effective_use_rag = settings.use_rag_examples if use_rag is None else use_rag
        if index_path.exists():
            index = load_index(str(index_path), embed_model=em)
        else:
            if DEFAULT_LABELS_XLSX.exists() and DEFAULT_SAMPLE_PDF.exists():
                vs = build_index_from_sample(str(DEFAULT_SAMPLE_PDF), str(DEFAULT_LABELS_XLSX), embed_model=em)
                save_index(vs, str(index_path))
                index = vs
            else:
                # Graceful fallback: proceed without RAG examples if not explicitly requested
                if effective_use_rag is True:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "RAG examples requested but no vector index found and sample files are missing. "
                            "Either provide an index (/.cache/index__*) or add data/sample files."
                        ),
                    )
                index = None
                effective_use_rag = False
                notice = (
                    "RAG examples unavailable: no vector index found and no sample data present. "
                    "Proceeding without RAG examples."
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        text = load_pdf_text(str(tmp_path), model_name=model)
        recs = extract_records(text, model_name=model, provider=("openai"))
        rows: list[ClassifiedItem] = []
        for rec in recs:
            out = classify_record(rec, index, model_name=model, provider=("openai"), use_rag=use_rag)
            final = apply_guardrails(rec, out.risk)
            rows.append(ClassifiedItem(
                deficiency=rec.deficiency,
                root_cause=rec.root_cause,
                corrective=rec.corrective,
                preventive=rec.preventive,
                risk_llm=out.risk,
                risk_final=final,
                rationale=out.rationale,
                evidence=out.evidence,
            ))
        if excel:
            # Ensure we export plain string labels ("High", "Medium", "Low") not Enum reprs ("Risk.High")
            risks = [getattr(r.risk_final, "value", str(r.risk_final)) for r in rows]
            df = pd.DataFrame({
                "Deficiency": list(range(1, len(risks) + 1)),
                "Risk": risks,
            })
            buf = io.BytesIO(); df.to_excel(buf, index=False); buf.seek(0)
            filename = (Path(pdf.filename).stem or "report") + "_predictions.xlsx"
            return StreamingResponse(
                buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
            )
        rag_used = bool(index is not None and effective_use_rag)
        return ClassifyResponse(count=len(rows), items=rows, rag_used=rag_used, notice=notice)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass



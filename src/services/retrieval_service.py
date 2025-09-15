import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from src.core.config import settings
from src.services.extraction_service import extract_records
from src.services.ocr_service import load_pdf_text

def build_index_from_sample(sample_pdf: str, labels_xlsx: str, embed_model: str | None = None) -> FAISS:
    full_text = load_pdf_text(sample_pdf)
    recs = extract_records(full_text)

    df = pd.read_excel(labels_xlsx)
    df.columns = [str(c).strip().lower() for c in df.columns]
    m = dict(zip(df["deficiency"].astype(int), df["risk"].astype(str)))

    docs = []
    for i, r in enumerate(recs, start=1):
        txt = (
            f"DEFICIENCY: {r.deficiency}\n"
            f"ROOT_CAUSE: {r.root_cause}\n"
            f"CORRECTIVE: {r.corrective}\n"
            f"PREVENTIVE: {r.preventive}"
        )
        lbl = m.get(i, "Low")
        docs.append(Document(page_content=txt, metadata={"label": lbl}))

    model = (embed_model or settings.embed_model)
    emb = OpenAIEmbeddings(model=model)
    return FAISS.from_documents(docs, emb)


def save_index(vs: FAISS, path: str):
	vs.save_local(path)


def load_index(path: str, embed_model: str | None = None) -> FAISS:
	model = (embed_model or settings.embed_model)
	return FAISS.load_local(path, OpenAIEmbeddings(model=model), allow_dangerous_deserialization=True)



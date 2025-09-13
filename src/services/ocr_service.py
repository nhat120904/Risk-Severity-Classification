from __future__ import annotations

from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

import base64
import io

import fitz  # PyMuPDF
from PIL import Image
from langchain_core.messages import HumanMessage

from src.services.llm_services import get_chat_llm


def _render_pdf_to_images(path: str | Path) -> list[Image.Image]:
	"""Render each PDF page to a PIL Image using PyMuPDF (no external deps)."""
	doc = fitz.open(str(path))
	images: list[Image.Image] = []
	try:
		for page in doc:
			pix = page.get_pixmap(dpi=200)
			png_bytes = pix.tobytes("png")
			img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
			images.append(img)
	finally:
		doc.close()
	return images


def _image_to_data_url(img: Image.Image) -> str:
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	b64 = base64.b64encode(buf.getvalue()).decode("ascii")
	return f"data:image/png;base64,{b64}"


def load_pdf_text(path: str | Path) -> str:
	"""Load text from a PDF. If no extractable text (scanned PDF), fall back to LLM OCR.

	Primary extraction uses PyMuPDF via LangChain. If pages contain no text,
	we render pages to images and use a vision-capable LLM to transcribe text.
	"""
	# 1) Try text extraction via PyMuPDF
	docs = PyMuPDFLoader(str(path)).load()
	if any((getattr(d, "page_content", "") or "").strip() for d in docs):
		return "\n".join((d.page_content or "") for d in docs)

	# 2) OCR fallback using vision LLM
	images = _render_pdf_to_images(path)
	if not images:
		return ""

	llm = get_chat_llm()
	page_texts: list[str] = []
	for img in images:
		data_url = _image_to_data_url(img)
		msg = HumanMessage(content=[
			{"type": "text", "text": (
				"You are an OCR engine. Transcribe the text exactly as seen, "
				"preserving line breaks. Output raw text only."
			)},
			{"type": "image_url", "image_url": {"url": data_url}},
		])
		resp = llm.invoke([msg])
		page_texts.append(str(getattr(resp, "content", resp)).strip())

	return "\n\n".join(page_texts).strip()



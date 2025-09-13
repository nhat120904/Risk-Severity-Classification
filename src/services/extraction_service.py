import re
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from src.core.schemas import DefRecord
from src.services.llm_services import get_chat_llm


_extract_parser = PydanticOutputParser(pydantic_object=DefRecord)
_extract_prompt = PromptTemplate(
	template=(
		"You are an extraction system for ship inspection reports.\n"
		"Extract JSON with keys: deficiency, root_cause, corrective, preventive.\n"
		"If a field is missing, use an empty string.\n"
		"Return ONLY JSON.\n\nTEXT:\n{block}\n\n{format_instructions}"
	),
	input_variables=["block"],
	partial_variables={"format_instructions": _extract_parser.get_format_instructions()},
)

_DEF_SPLIT = re.compile(r"\bDeficiency\s+\d+\b", flags=re.IGNORECASE)


def split_def_blocks(full_text: str) -> list[str]:
	parts = _DEF_SPLIT.split(full_text)
	return [p.strip() for p in parts if p.strip()]


def _regex_capture(label: str, text: str) -> str:
	# Capture multi-line field content until next known label or end
	labels = r"Deficiency|Root\s*Cause|Corrective(?:\s*Action)?|Preventive(?:\s*Action)?"
	# Group the label alternatives to ensure the capture group applies to all branches
	pattern = rf"(?:{label})\s*:?\s*(?P<val>.+?)(?=\n\s*(?:{labels})\s*:|\Z)"
	m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
	return (m.group("val").strip() if m else "")


def _extract_with_regex(block: str) -> DefRecord | None:
	# Try robust regex-first extraction
	deficiency = _regex_capture(r"Deficiency", block)
	root = _regex_capture(r"Root\s*Cause", block)
	corr = _regex_capture(r"Corrective(?:\s*Action)?|Corrective", block)
	prev = _regex_capture(r"Preventive(?:\s*Action)?|Preventive", block)
	if deficiency:
		return DefRecord(
			deficiency=deficiency,
			root_cause=root,
			corrective=corr,
			preventive=prev,
		)
	return None


def extract_records(full_text: str, model_name: str | None = None, provider: str | None = None) -> list[DefRecord]:
	blocks = split_def_blocks(full_text)
	recs: list[DefRecord] = []
	llm = None
	for b in blocks:
		# 1) Try regex-first heuristic extraction
		regex_rec = _extract_with_regex(b)
		if regex_rec is not None:
			recs.append(regex_rec)
			continue

		# 2) Fallback to LLM extraction
		try:
			if llm is None:
				llm = get_chat_llm(provider, model_name, temperature=0)
			rec = (_extract_prompt | llm | _extract_parser).invoke({"block": b})
			recs.append(rec)
		except Exception:
			# If LLM extraction also fails, skip this block (no simple fallback).
			continue
	return recs



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


def extract_records(full_text: str, model_name: str | None = None, provider: str | None = None) -> list[DefRecord]:
	blocks = split_def_blocks(full_text)
	recs: list[DefRecord] = []
	llm = get_chat_llm(provider, model_name, temperature=0)
	for b in blocks:
		try:
			rec = (_extract_prompt | llm | _extract_parser).invoke({"block": b})
			recs.append(rec)
		except Exception:
			# very simple fallback if LLM fails: try regex heuristics
			deficiency = re.search(r"Deficiency\s*:\s*(.*)", b, re.IGNORECASE)
			root = re.search(r"Root\s*Cause\s*:\s*(.*)", b, re.IGNORECASE)
			corr = re.search(r"Corrective\s*action\s*:\s*(.*)", b, re.IGNORECASE)
			prev = re.search(r"Preventive\s*action\s*:\s*(.*)", b, re.IGNORECASE)
			recs.append(DefRecord(
				deficiency=(deficiency.group(1).strip() if deficiency else b[:200]),
				root_cause=(root.group(1).strip() if root else ""),
				corrective=(corr.group(1).strip() if corr else ""),
				preventive=(prev.group(1).strip() if prev else ""),
			))
	return recs



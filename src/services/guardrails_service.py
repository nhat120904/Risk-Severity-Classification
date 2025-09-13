import re
from src.core.schemas import DefRecord, Risk


HIGH_PATTERNS = [
	"\\bemergency stop\\b",
	"\\bemergency signal\\b",
	"\\bfire extinguisher\\b",
	"\\bload ?line\\b",
	"\\biopp\\b",
	"\\blifeboat|rescue boat\\b",
	"\\bstability (computer|calculation).*not approved\\b",
	"\\bco2 system\\b",
	"\\bfoam system\\b",
]

MED_PATTERNS = [
	"\\bcertificate\\b",
	"\\blogbook\\b",
	"\\bprocedure\\b",
	"\\btraining\\b",
]

_COMPILED_HIGH = [re.compile(p, re.IGNORECASE) for p in HIGH_PATTERNS]
_COMPILED_MED = [re.compile(p, re.IGNORECASE) for p in MED_PATTERNS]


def apply_guardrails(rec: DefRecord, llm_label: Risk) -> Risk:
	txt = (f"{rec.deficiency} {rec.root_cause} {rec.corrective} {rec.preventive}").lower()
	if any(p.search(txt) for p in _COMPILED_HIGH):
		return Risk.High
	if llm_label == Risk.High and any(p.search(txt) for p in _COMPILED_MED):
		return Risk.Medium
	return llm_label



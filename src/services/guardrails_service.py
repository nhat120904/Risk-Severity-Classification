import re
from src.core.schemas import DefRecord, Risk

# High-anchors = things that, when degraded, directly threaten personnel/ship/environment.
HIGH_ANCHORS = [
    r"\bfire\s*extinguisher\b",
    r"\b(life\s*boat|rescue\s*boat|life\s*raft)\b",
    r"\bscba\b",                  
    r"\bgmdss\b",                  
    r"\bgeneral\s*alarm\b",
    r"\bemergency\s*generator\b",
    r"\bemergency\s*steering\b",
    r"\bco2\s+system\b",
    r"\bfoam\s+system\b",
]

# Failure/unsafe context
FAIL_CTX = (
    r"(inoperat|inoperative|not\s*(work|approved|available|provided|fitted|installed|compliant)|"
    r"defect|damag|leak|expired|missing|fault|unserviceable|non[- ]?functional|fail(ed|ure)|"
    r"corrod|rusted|broken|cracked|bent|seized|stuck|malfunction|unsafe)"
)

# Pre-compiled patterns for speed
_COMPILED_HIGH = [re.compile(p, re.IGNORECASE) for p in HIGH_ANCHORS]
_RE_FAIL       = re.compile(FAIL_CTX, re.IGNORECASE)

def _has_high_with_failure(problem_text: str, window: int = 48) -> bool:
    """
    Return True iff a High-anchor appears close (±window chars) to an explicit failure/unsafe context.
    This minimizes false High when the anchor is only named without any degradation.
    """
    for a in _COMPILED_HIGH:
        for m in a.finditer(problem_text):
            s, e = m.span()
            left, right = max(0, s - window), min(len(problem_text), e + window)
            if _RE_FAIL.search(problem_text[left:right]):
                return True
    return False

 

def apply_guardrails(rec: DefRecord, llm_label: Risk) -> Risk:
    """
    Heuristics:
    1) Inspect ONLY the problem statement (deficiency + root_cause) for guardrails.
       We intentionally ignore 'corrective' and 'preventive' to avoid false promotions.
    2) Promote to High if and only if a High-anchor is near a failure/unsafe context.
    3) Otherwise, keep the LLM label.
    """
    problem_text = f"{rec.deficiency or ''} {rec.root_cause or ''}".lower()

    high_with_failure = _has_high_with_failure(problem_text)

    # 1) Promotion to High (strong evidence of degraded lifesaving/emergency system)
    if llm_label != Risk.High and high_with_failure:
        return Risk.High

    # 2) Otherwise, respect the model's label
    return llm_label

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langsmith import traceable

from src.core.config import settings
from src.core.schemas import DefRecord, ClfOut
from src.services.llm_services import get_chat_llm


DEFINITIONS = (
    "High = Significant finding threatening personnel, ship, or environment; likely to impair emergency response "
    "or pollution prevention; could cause economic/reputational harm or PSC detention.\n"
    "Medium = Weakness in processes, procedures, or shipboard practices that increases risk but does not imminently "
    "impair emergency systems or pollution prevention.\n"
    "Low = Administrative/minor issues that are unlikely to cause an accident; at worst minor damage if realized."
)

DECISION_RULES = """
DECISION RULES (apply in order; first match wins):

1) HIGH if any of the following are affected or plausibly impaired:
   - Firefighting or Life-Saving Appliances (LSA): extinguishers, rescue boats, liferafts, SCBA, alarms.
   - Emergency systems: GMDSS, general alarm, emergency generator, steering (emergency mode).
   - Pollution prevention/critical statutory (MARPOL/IOPP/ISPS/ISM when safety is jeopardized; expired/invalid critical certs).
   - Structural or watertight integrity; navigation safety equipment with safety impact.
   - Systemic failure leading to immediate risk (repeated, widespread, or during operations).
   When unsure between High vs Medium on life-safety/pollution, choose HIGH. Evidence must reference at least one concrete item above.

2) MEDIUM if:
   - Process/procedure/training/records weaknesses (e.g., logbook, certificate filing) without immediate safety/pollution impact.
   - Documentation inconsistencies where a valid certificate/system remained in effect; safety not impaired.
   - Single-point human error corrected with preventive actions; no critical system impairment.

3) LOW if:
   - Minor administrative/cosmetic issues; unlikely to lead to an accident by themselves.
   - Isolated record-keeping/clerical items with no operational consequence and already corrected.

TIE-BREAKERS:
- Favor HIGH for life-safety/pollution-critical items.
- Missed drill entry but promptly corrected/trained => usually LOW if isolated.
- Do NOT choose HIGH unless the evidence explicitly names a life-safety/pollution-critical item per Rule 1.
"""

CLASSIFY_HEADER = (
    "You are an AI risk assessor for RightShip. Classify ship inspection DEFICIENCIES into High/Medium/Low.\n\n"
    "RISK DEFINITIONS:\n{definitions}\n\n"
    "{decision_rules}\n\n"
)

EXAMPLES_BLOCK = (
    "REFERENCE EXAMPLES (from similar labeled cases):\n{examples}\n\n"
)

CLASSIFY_TAIL = (
    "INSTRUCTIONS:\n"
    "- Read only the NEW RECORD content below.\n"
    "- Apply the DECISION RULES strictly and be conservative for life-safety/pollution.\n"
    "OUTPUT DISCIPLINE:\n"
    "- Return strict JSON with keys: risk, rationale, evidence.\n"
    "- rationale: ≤30 words, cite the rule briefly.\n"
    "- evidence: 1–3 verbatim spans from the NEW RECORD (short quotes).\n"
    "- No markdown, no extra keys, no explanations.\n\n"
    "NEW RECORD:\n{record}\n\n"
    "JSON SCHEMA REMINDER:\n"
    '{{"risk":"High|Medium|Low","rationale":"short","evidence":["quote1","quote2"]}}'
)


_parser = PydanticOutputParser(pydantic_object=ClfOut)


def _examples_text(docs) -> str:
    lines = []
    for d in docs:
        lbl = d.metadata.get("label", "?")
        txt = d.page_content.replace("\n", " ")
        if len(txt) > 650:
            txt = txt[:650] + " ..."
        lines.append(f"- Label: {lbl} | Text: {txt}")
    return "\n".join(lines)


def _build_record_text(rec: DefRecord) -> str:
    return (
        f"DEFICIENCY: {rec.deficiency}\n"
        f"ROOT_CAUSE: {rec.root_cause}\n"
        f"CORRECTIVE: {rec.corrective}\n"
        f"PREVENTIVE: {rec.preventive}"
    )


def _build_prompt_template(include_examples: bool) -> PromptTemplate:
    if include_examples:
        template = (
            CLASSIFY_HEADER
            + EXAMPLES_BLOCK
            + CLASSIFY_TAIL
            + "\n{format_instructions}"
        )
        input_vars = ["record", "examples"]
    else:
        template = (
            CLASSIFY_HEADER
            + CLASSIFY_TAIL
            + "\n{format_instructions}"
        )
        input_vars = ["record"]

    return PromptTemplate(
        template=template,
        input_variables=input_vars,
        partial_variables={
            "definitions": DEFINITIONS,
            "decision_rules": DECISION_RULES,
            "format_instructions": _parser.get_format_instructions(),
        },
    )


@traceable
def classify_record(
    rec: DefRecord,
    index: VectorStore,
    k: int = 3,
    model_name: str | None = None,
    provider: str | None = None,
    use_rag: bool | None = None,
) -> ClfOut:
    use_rag_examples = settings.use_rag_examples if use_rag is None else use_rag
    record_txt = _build_record_text(rec)

    if use_rag_examples and index is not None:
        retriever = index.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.25},
        )
        q = f"DEFICIENCY: {rec.deficiency}\nROOT_CAUSE: {rec.root_cause}"
        docs = retriever.invoke(q) or []
        examples_block = _examples_text(docs) if docs else ""
        include_examples = bool(examples_block)
    else:
        examples_block = ""
        include_examples = False

    prompt = _build_prompt_template(include_examples)

    llm = get_chat_llm(provider, model_name, temperature=0)
    chain = prompt | llm | _parser

    if include_examples:
        return chain.invoke({"record": record_txt, "examples": examples_block})
    else:
        return chain.invoke({"record": record_txt})



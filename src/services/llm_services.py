from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_chat_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0):
    if provider and provider.lower() != "openai":
        raise ValueError("Only 'openai' provider is supported.")
    mdl = model or os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return ChatOpenAI(model=mdl, temperature=temperature)



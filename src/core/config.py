from dataclasses import dataclass
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


@dataclass
class Settings:
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    use_rag_examples: bool = os.getenv("USE_RAG_EXAMPLES", "true").lower() in {"1", "true", "yes", "y"}

    api_type: str | None = os.getenv("OPENAI_API_TYPE")
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    api_base: str | None = os.getenv("OPENAI_API_BASE")
    api_version: str | None = os.getenv("OPENAI_API_VERSION")
    deployment: str | None = os.getenv("OPENAI_DEPLOYMENT_NAME")


settings = Settings()



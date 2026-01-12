# config.py
import os
from openai import OpenAI

def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Environment variable '{name}' is required but not set.")
    return val

# OpenAI
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
OPENAI_MODEL = get_env("OPENAI_MODEL", default="gpt-4.1-mini")  # 必要に応じて変更
OPENAI_EMBEDDING_MODEL = get_env("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

# Neo4j
NEO4J_URI = get_env("NEO4J_URI", default="bolt://localhost:7687")
NEO4J_USER = get_env("NEO4J_USER", default="neo4j")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD", required=True)

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
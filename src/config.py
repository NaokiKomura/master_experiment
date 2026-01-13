import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- パス設定 ---
# このファイル(src/config.py)のあるディレクトリ
SRC_DIR = Path(__file__).resolve().parent
# プロジェクトルート (srcの親ディレクトリ)
PROJECT_ROOT = SRC_DIR.parent

# .envファイルの読み込み (プロジェクトルートにあると想定)
load_dotenv(PROJECT_ROOT / ".env")

# --- OpenAI設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "256"))
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "") or None
GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p":1.0,
    "max_tokens":1000,
    "seed":16
}

# --- Neo4j設定 ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# --- ファイルパス設定 ---
# キャッシュファイルは src ディレクトリ内に保存
CACHE_FILE = SRC_DIR / "embedding_cache.json"

# 実験データCSV (プロジェクトルートにあると想定)
DATA_FILE = PROJECT_ROOT / "data/research_data.csv" 

# --- バリデーション ---
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set in .env")
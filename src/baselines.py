# baselines.py (修正案)
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# configから設定を読み込む
from config import OPENAI_API_KEY, CHAT_MODEL, GENERATION_PARAMS, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Local Text-RAG Knowledge Base (Markdown) ---
# Expected location: <repo_root>/data/text_rag_kb/*.md
REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_RAG_KB_DIR = REPO_ROOT / "data" / "text_rag_kb"
TEXT_RAG_CACHE_PATH = REPO_ROOT / "data" / "text_rag_kb_embeddings_cache.json"
FEWSHOT_JSONL_PATH = REPO_ROOT / "data" / "exp1_representatives.jsonl"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _read_markdown_files(kb_dir: Path) -> List[Tuple[str, str, float]]:
    """Return list of (filename, text, mtime)."""
    files = sorted(kb_dir.glob("*.md"))
    out: List[Tuple[str, str, float]] = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        out.append((fp.name, text, fp.stat().st_mtime))
    return out


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"items": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}


def _save_cache(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to save Text-RAG cache: {e}")


def _get_text_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def _build_kb_index(kb_dir: Path, cache_path: Path) -> List[Dict[str, Any]]:
    """Load markdown KB and return list of {name, text, embedding}. Uses a simple on-disk cache."""
    docs = _read_markdown_files(kb_dir)
    cache = _load_cache(cache_path)
    cache_items: Dict[str, Dict[str, Any]] = {i.get("name"): i for i in cache.get("items", []) if i.get("name")}

    new_items: List[Dict[str, Any]] = []
    changed = False

    for name, text, mtime in docs:
        cached = cache_items.get(name)
        if cached and float(cached.get("mtime", 0.0)) == float(mtime) and isinstance(cached.get("embedding"), list):
            new_items.append({"name": name, "text": cached.get("text", text), "mtime": mtime, "embedding": cached["embedding"]})
            continue

        # Re-embed when file is new or modified
        emb = _get_text_embedding(text)
        new_items.append({"name": name, "text": text, "mtime": mtime, "embedding": emb})
        changed = True

    if changed:
        _save_cache(cache_path, {"items": new_items})

    return new_items


def _retrieve_kb_context(query: str, *, top_k: int = 3) -> str:
    """Retrieve top_k markdown docs by embedding similarity and return a context string."""
    if not TEXT_RAG_KB_DIR.exists():
        raise RuntimeError(
            f"Text RAG 用KBディレクトリが見つかりません: {TEXT_RAG_KB_DIR}. "
            f"data/text_rag_kb/ に .md ファイルを配置してください。"
        )

    kb_items = _build_kb_index(TEXT_RAG_KB_DIR, TEXT_RAG_CACHE_PATH)
    if not kb_items:
        raise RuntimeError(
            f"Text RAG 用KBに .md ファイルがありません: {TEXT_RAG_KB_DIR}. "
            f"最低1つ以上の .md を配置してください。"
        )

    q_emb = _get_text_embedding(query)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in kb_items:
        emb = item.get("embedding")
        if not isinstance(emb, list):
            continue
        score = _cosine_similarity(q_emb, emb)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, int(top_k))]

    parts: List[str] = []
    for score, item in top:
        name = item.get("name", "")
        text = item.get("text", "")
        # Context is limited to avoid overlong prompts
        snippet = text.strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + "\n... (truncated)"
        parts.append(f"[DOC: {name} | score={score:.3f}]\n{snippet}")

    return "\n\n".join(parts)


def _load_fewshot_examples(path: Path, max_examples: int = 8) -> List[Dict[str, Any]]:
    """Load few-shot examples from JSONL.

    Expected keys per line (best-effort):
      - text (or copy/ad_text)
      - label (or y/target/tag)
      - reason (optional)

    Returns a list of dicts with keys: text, label, reason.
    """
    if not path.exists():
        logging.warning(f"Few-shot examples file not found: {path}")
        return []

    examples: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                text = obj.get("text") or obj.get("copy") or obj.get("ad_text") or obj.get("sentence")
                label = obj.get("label")
                if label is None:
                    label = obj.get("y") if "y" in obj else obj.get("target")
                if label is None:
                    label = obj.get("tag")

                # label normalize to int 0/1
                try:
                    label_int = int(label)
                except Exception:
                    label_int = 1 if str(label).strip().lower() in {"1", "true", "yes", "risk", "positive"} else 0

                reason = obj.get("reason") or obj.get("rationale") or obj.get("note") or ""

                if not text:
                    continue

                examples.append({"text": str(text), "label": int(label_int), "reason": str(reason)})
                if len(examples) >= int(max_examples):
                    break

    except Exception as e:
        logging.error(f"Failed to load few-shot examples from {path}: {e}")
        return []

    return examples

def _call_llm(prompt):
    """共通のLLM呼び出し関数"""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,  # configのモデル(GPT-5等)を使用
            messages=[{"role": "user", "content": prompt}],
            temperature=GENERATION_PARAMS["temperature"],
            # max_tokens等は必要に応じて
        )
        content = response.choices[0].message.content.strip()
        # 簡易的に数字だけ抽出（論文の0/1判定に合わせて）
        if "1" in content: return 1
        return 0
    except Exception as e:
        logging.error(f"LLM Error: {e}")
        return 0

# B0: Zero-shot
def predict_zero_shot(text):
    prompt = f"""
    あなたは広告倫理の専門家です。以下の広告コピーに「炎上リスク」があるか判定してください。
    広告コピー: {text}
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    return _call_llm(prompt)

# B1: Few-shot (事例データは呼び出し元から注入するか、ここで定義)
def predict_few_shot(text, examples=None):
    # examples が None/空 の場合は data/exp1_representatives.jsonl を使用
    if not examples:
        examples = _load_fewshot_examples(FEWSHOT_JSONL_PATH, max_examples=8)

    # それでも空なら最低限のフォールバック
    if not examples:
        examples = [
            {"text": "家事はママの仕事", "label": 1, "reason": "性役割の固定化"},
            {"text": "プロなら残業は当たり前", "label": 1, "reason": "高圧的"},
            {"text": "皆様の暮らしを応援します", "label": 0, "reason": "問題なし"},
        ]
    
    examples_str = "\n".join([f"例: {ex['text']}\n判定: {ex['label']}\n理由: {ex['reason']}" for ex in examples])
    prompt = f"""
    あなたは広告倫理の専門家です。以下の過去の判定事例を参考に、対象の広告コピーに「炎上リスク」があるか判定してください。
    【参考事例】
    {examples_str}
    【対象広告】
    {text}
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    return _call_llm(prompt)

# B2: Text RAG (Local Markdown KB)
def predict_text_rag(text, driver=None):
    # driver is kept only for API compatibility (batch_experiment.py may pass Neo4j driver)
    try:
        context_str = _retrieve_kb_context(text, top_k=3)
    except Exception as e:
        logging.error(f"Text RAG retrieval error: {e}")
        context_str = "関連情報なし"

    prompt = f"""
    あなたは広告倫理の専門家です。以下の「関連する社会的知識」に基づいて、広告コピーのリスクを判定してください。
    【関連知識】
    {context_str}
    【対象広告】
    {text}
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    return _call_llm(prompt)
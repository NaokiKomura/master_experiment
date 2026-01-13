from __future__ import annotations

import json
import uuid
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from openai import OpenAI, APIConnectionError, RateLimitError, APIError

# configから設定を読み込み
try:
    from config import OPENAI_API_KEY, CHAT_MODEL
except ImportError:
    # srcディレクトリ外から実行された場合の対策
    from .config import OPENAI_API_KEY, CHAT_MODEL

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Batch用の簡易API（batch_experiment.py から呼び出す想定） ---
_PROCESSOR_SINGLETON: Optional["AdContentProcessor"] = None


def get_processor_singleton() -> "AdContentProcessor":
    """AdContentProcessor のシングルトンを返す。

    バッチ評価では多数の広告を連続処理するため、毎回初期化（モデル設定等）しない。
    """
    global _PROCESSOR_SINGLETON
    if _PROCESSOR_SINGLETON is None:
        _PROCESSOR_SINGLETON = AdContentProcessor()
    return _PROCESSOR_SINGLETON

# バージョン管理用定数
PROMPT_VERSION = "v2.0_fact_extraction"
PROCESSOR_VERSION = "2025-01-13_strict_schema"

class AdContentProcessor:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = CHAT_MODEL
        self.temperature = 0.0

    def _generate_deterministic_id(self, input_text: str, meta: Dict[str, Any]) -> str:
        if meta and 'csv_id' in meta:
            seed = f"csv_{meta['csv_id']}"
        else:
            seed = hashlib.md5(input_text.encode('utf-8')).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

    def _clean_list_strings(self, raw_list: Any, max_items: int = 10) -> List[str]:
        if not isinstance(raw_list, list):
            return []
        cleaned = []
        seen = set()
        for item in raw_list:
            if isinstance(item, str):
                s = item.strip()
                if s and s not in seen:
                    cleaned.append(s)
                    seen.add(s)
        return cleaned[:max_items]

    def _normalize_payload(self, raw_json: Dict[str, Any]) -> Dict[str, Any]:
        raw_ctx = raw_json.get("context", {})
        if not isinstance(raw_ctx, dict): raw_ctx = {}
        
        context = {
            "media_type": str(raw_ctx.get("media_type", "unknown")),
            "timing": str(raw_ctx.get("timing", "unknown")),
            "target": str(raw_ctx.get("target", "unknown"))
        }

        raw_exprs = raw_json.get("expressions", [])
        if not isinstance(raw_exprs, list): raw_exprs = []

        expressions = []
        for expr in raw_exprs:
            if not isinstance(expr, dict): continue
            text = str(expr.get("text", "")).strip()
            if not text: continue

            associations = self._clean_list_strings(expr.get("associations"), max_items=5)
            roles = self._clean_list_strings(expr.get("roles"), max_items=5)
            evidence = str(expr.get("evidence", "")).strip()

            expressions.append({
                "text": text,
                "associations": associations,
                "roles": roles,
                "evidence": evidence
            })

        return {"context": context, "expressions": expressions}

    def analyze_ad_content(self, input_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not input_text:
            raise ValueError("Input text is empty.")

        meta = meta or {}
        ad_uuid = self._generate_deterministic_id(input_text, meta)

        processing_meta = {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": PROMPT_VERSION,
            "processor_version": PROCESSOR_VERSION,
            "original_meta": meta
        }

        system_prompt = """
        あなたは広告リスク分析のためのデータ構造化スペシャリストです。
        入力された広告コピーを分析し、以下のJSON形式で特徴を抽出してください。
        
        【重要方針: 事実の抽出】
        - 「炎上リスクがあるか」等の評価・判断は行わないでください。
        - 書かれている事実、そこから直接連想される一般的な概念のみを抽出してください。
        
        【抽出項目】
        1. expressions: 広告内の主要なフレーズ。
        2. associations: 各表現から連想される「検索可能な名詞句」。
           - 禁止: 「不快」「差別的」などの評価語。
           - 推奨: "ワンオペ", "性別役割分業", "ルッキズム" などの社会的概念・キーワード。
           - 上限: 各表現につき最大5つまで。
        3. roles: 表現内で描かれている人物像や役割（例: "主婦", "サラリーマン"）。
        4. evidence: その連想や役割を抽出した根拠。
           - 必須: 根拠となる広告内の原文を『』で引用すること。
        
        【出力フォーマット(JSON)】
        {
          "context": {
            "media_type": "TVCM / Web / SNS など",
            "timing": "Morning / Night / Emergency など",
            "target": "想定ターゲット層"
          },
          "expressions": [
            {
              "text": "フレーズ原文",
              "associations": ["連想語1", "連想語2"],
              "roles": ["役割1", "役割2"],
              "evidence": "原文の『〜』という表現から抽出"
            }
          ]
        }
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"広告コピー: {input_text}"}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )

                result_content = response.choices[0].message.content
                raw_data = json.loads(result_content)
                normalized_data = self._normalize_payload(raw_data)

                payload = {
                    "ad_id": ad_uuid,
                    "input_text": input_text,
                    "meta": processing_meta,
                    "context": normalized_data["context"],
                    "expressions": normalized_data["expressions"]
                }
                
                logger.info(f"Analyzed Ad {ad_uuid}: {len(payload['expressions'])} expressions extracted.")
                return payload

            except (RateLimitError, APIConnectionError, APIError) as e:
                wait_time = 2 ** attempt
                logger.warning(f"API Error ({e}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error in analysis: {e}")
                if attempt == max_retries - 1: raise
                time.sleep(1)

        raise RuntimeError(f"Failed to analyze content after {max_retries} retries.")


def extract_facts(
    input_text: str,
    ad_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """バッチ実験用: 広告文からfact extractionを行い、loaderが扱えるpayload(dict)を返す。

    rag_app.py はクラス `AdContentProcessor.analyze_ad_content()` を直接使うが、
    batch_experiment.py では統一的な関数APIを想定しているため、ここで薄いラッパーを提供する。

    Args:
        input_text: 広告コピー本文
        ad_id: 既知の広告ID（CSVのID等）。指定された場合はpayloadのad_idをこの値に固定する。
        meta: 任意メタ情報（csv_id, brand 等）。processor内部の決定論的ID生成にも影響し得る。

    Returns:
        payload: {
          "ad_id": str,
          "input_text": str,
          "meta": dict,
          "context": dict,
          "expressions": list[dict]
        }
    """
    # loader/load_to_neo4j のログにcsv_idが出るよう、指定が無ければ ad_id をcsv_idとして埋める
    if meta is None and ad_id is not None:
        meta = {"csv_id": str(ad_id)}

    proc = get_processor_singleton()
    payload = proc.analyze_ad_content(input_text=input_text, meta=meta)

    # batch側のIDを優先して整合を取りやすくする
    if ad_id is not None:
        payload["ad_id"] = str(ad_id)

    return payload

if __name__ == "__main__":
    # 簡易テスト
    try:
        proc = AdContentProcessor()
        print("Processor initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")
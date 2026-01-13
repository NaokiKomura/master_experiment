import os
import json
import uuid
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, APIError

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# バージョン管理用定数 (実験ログに残すため)
PROMPT_VERSION = "v2.0_fact_extraction"
PROCESSOR_VERSION = "2025-01-13_strict_schema"

class AdContentProcessor:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o" 
        self.temperature = 0.0

    def _generate_deterministic_id(self, input_text: str, meta: Dict[str, Any]) -> str:
        """
        入力内容に基づいて常に同じIDを生成する（再実行時の重複防止）
        csv_idがあればそれを優先シードにし、なければテキストハッシュを使う
        """
        if meta and 'csv_id' in meta:
            seed = f"csv_{meta['csv_id']}"
        else:
            # テキストのハッシュをシードにする
            seed = hashlib.md5(input_text.encode('utf-8')).hexdigest()
        
        # UUID5 (SHA-1 hash of a namespace identifier and a name)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

    def _clean_list_strings(self, raw_list: Any, max_items: int = 10) -> List[str]:
        """リスト型を強制し、文字列のクリーニングと重複除去を行う"""
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
        """
        LLMの出力を loader.py が期待するスキーマに強制変換・補完する
        """
        # Contextの正規化
        raw_ctx = raw_json.get("context", {})
        if not isinstance(raw_ctx, dict): raw_ctx = {}
        
        context = {
            "media_type": str(raw_ctx.get("media_type", "unknown")),
            "timing": str(raw_ctx.get("timing", "unknown")),
            "target": str(raw_ctx.get("target", "unknown"))
        }

        # Expressionsの正規化
        raw_exprs = raw_json.get("expressions", [])
        if not isinstance(raw_exprs, list): raw_exprs = []

        expressions = []
        for expr in raw_exprs:
            if not isinstance(expr, dict): continue
            
            text = str(expr.get("text", "")).strip()
            if not text: continue # テキストがないExpressionは無意味なので捨てる

            # リスト項目のクリーニング
            associations = self._clean_list_strings(expr.get("associations"), max_items=5)
            roles = self._clean_list_strings(expr.get("roles"), max_items=5)
            
            evidence = str(expr.get("evidence", "")).strip()

            expressions.append({
                "text": text,
                "associations": associations,
                "roles": roles,
                "evidence": evidence
            })

        return {
            "context": context,
            "expressions": expressions
        }

    def analyze_ad_content(self, input_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        広告テキストを分析し、オントロジーマッピング用の構造化データを抽出する。
        リトライ処理とスキーマバリデーションを含む。
        """
        if not input_text:
            raise ValueError("Input text is empty.")

        meta = meta or {}
        ad_uuid = self._generate_deterministic_id(input_text, meta)

        # 実験メタデータの記録
        processing_meta = {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": PROMPT_VERSION,
            "processor_version": PROCESSOR_VERSION,
            "original_meta": meta
        }

        system_prompt = f"""
        あなたは広告リスク分析のためのデータ構造化スペシャリストです。
        入力された広告コピーを分析し、以下のJSON形式で特徴を抽出してください。
        
        【重要方針: 事実の抽出】
        - 「炎上リスクがあるか」等の評価・判断は行わないでください。
        - 書かれている事実、そこから直接連想される一般的な概念のみを抽出してください。
        
        【抽出項目】
        1. expressions: 広告内の主要なフレーズ。
        2. associations: 各表現から連想される「検索可能な名詞句」。
           - 禁止: 「不快」「差別的」などの評価語。
           - 追加の制約: 原文にない「深読み」や「意図の推測」は行わないこと。
             言葉そのものが持つ直接的な意味のみを抽出してください。
             (Bad: "料理" -> "女性の家事負担", Good: "料理" -> "家事")
        3. roles: 表現内で描かれている人物像や役割（例: "主婦", "サラリーマン"）。
        4. evidence: その連想や役割を抽出した根拠。
           - 必須: 根拠となる広告内の原文を『』で引用すること。
        
        【出力フォーマット(JSON)】
        {{
          "context": {{
            "media_type": "TVCM / Web / SNS など",
            "timing": "Morning / Night / Emergency など",
            "target": "想定ターゲット層"
          }},
          "expressions": [
            {{
              "text": "フレーズ原文",
              "associations": ["連想語1", "連想語2"],
              "roles": ["役割1", "役割2"],
              "evidence": "原文の『〜』という表現から抽出"
            }}
          ]
        }}
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

                # バリデーションと正規化
                normalized_data = self._normalize_payload(raw_data)

                # 最終ペイロードの構築
                payload = {
                    "ad_id": ad_uuid,
                    "input_text": input_text,
                    "meta": processing_meta, # 拡張されたメタデータ
                    "context": normalized_data["context"],
                    "expressions": normalized_data["expressions"]
                }
                
                logger.info(f"Analyzed Ad {ad_uuid}: {len(payload['expressions'])} expressions extracted.")
                return payload

            except (RateLimitError, APIConnectionError, APIError) as e:
                wait_time = 2 ** attempt
                logger.warning(f"API Error ({e}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except json.JSONDecodeError:
                logger.error("JSON Decode Error from LLM response.")
                # JSONエラーはリトライしても直らない場合が多いが、一時的な崩れならリトライ価値あり
                if attempt == max_retries - 1: raise
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error in analysis: {e}")
                raise

        raise RuntimeError(f"Failed to analyze content after {max_retries} retries.")

# --- テスト実行用 ---
if __name__ == "__main__":
    processor = AdContentProcessor()
    
    # テストデータ (CSV IDあり)
    sample_text = "家事はママの仕事、がんばって。家族のために。"
    sample_meta = {"csv_id": "TEST_999", "brand": "TestBrand"}
    
    try:
        # 1回目
        result1 = processor.analyze_ad_content(sample_text, sample_meta)
        print(f"ID 1: {result1['ad_id']}")
        
        # 2回目 (同じIDになるか確認)
        result2 = processor.analyze_ad_content(sample_text, sample_meta)
        print(f"ID 2: {result2['ad_id']}")
        
        assert result1['ad_id'] == result2['ad_id']
        print("Deterministic ID check passed.")
        
        print(json.dumps(result1, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")
import uuid
import os
from dotenv import load_dotenv
from openai import OpenAI
from models import AdAnalysisResult

# 環境変数の読み込み
load_dotenv()

# OpenAIクライアント初期化
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY が環境変数に設定されていません。")

client = OpenAI(api_key=openai_api_key)

SYSTEM_PROMPT = """
あなたは広告の社会的リスクを分析する専門AIです。
入力された広告情報から、以下の手順で情報を抽出してください。

Phase 2-A: 事実抽出
1. 広告を「表現単位(Expression)」に分割する。
2. それぞれの表現に「根拠(Evidence)」を必ず紐付ける。
3. 誰が何をしているか(DepictedRole)を客観的に抽出する。

Phase 2-B: フレーミング判定
1. その表現が特定の概念を「強化(REINFORCES)」「当然視(NORMALIZES)」していないか判定する。
2. Association(連想)は「家事＝母の責務」のような短い名詞句にする。

出力は指定されたJSONフォーマットに従ってください。
"""

def get_embedding(text: str) -> list:
    """テキストをベクトル化する関数 (1536次元)"""
    if not text:
        return []
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def analyze_ad_content(copy_text: str, visual_desc: str) -> dict:
    """
    LLM分析 + ベクトル化を行い、Neo4j投入用データを生成する
    """
    print("1. LLM Reasoning...")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"コピー: {copy_text}\n\n映像描写: {visual_desc}"},
        ],
        response_format=AdAnalysisResult,
    )
    
    raw_result = completion.choices[0].message.parsed
    
    # ID生成とデータ整形
    ad_id = str(uuid.uuid4())
    
    payload = {
        "ad_id": ad_id,
        "meta": {
            "name": "Generated Ad Analysis", 
            "medium": "TV_Commercial",
            "url": "http://example.com"
        },
        "contexts": [],
        "expressions": []
    }
    
    for ctx in raw_result.contexts:
        payload["contexts"].append({
            "id": str(uuid.uuid4()),
            "type": ctx.type,
            "desc": ctx.desc,
            "source_text": ctx.source_text
        })
        
    print("2. Generating Embeddings...")
    for expr in raw_result.expressions:
        expr_id = str(uuid.uuid4())
        evidence_id = str(uuid.uuid4())
        
        # ベクトル化を実行
        embedding_vector = get_embedding(expr.text)

        expr_dict = {
            "id": expr_id,
            "text": expr.text,
            "embedding": embedding_vector,
            "modality": expr.modality.value,
            "evidence": {
                "id": evidence_id,
                "quote": expr.evidence.quote,
                "source": expr.evidence.source,
                "modality": expr.modality.value
            },
            "depicted_roles": [role.model_dump() for role in expr.depicted_roles],
            "targets": [tgt.model_dump() for tgt in expr.targets],
            "evokes": [ev.model_dump() for ev in expr.evokes],
            "framing": [frm.model_dump() for frm in expr.framing]
        }
        payload["expressions"].append(expr_dict)
        
    return payload
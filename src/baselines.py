# baselines.py (修正案)
import logging
from openai import OpenAI
# configから設定を読み込む
from config import OPENAI_API_KEY, CHAT_MODEL, GENERATION_PARAMS, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

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
    # examplesがNoneならデフォルトの事例（論文付録相当）を使用
    if not examples:
        examples = [
            {"text": "家事はママの仕事", "label": 1, "reason": "性役割の固定化"},
            {"text": "プロなら残業は当たり前", "label": 1, "reason": "高圧的"},
            {"text": "皆様の暮らしを応援します", "label": 0, "reason": "問題なし"}
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

# B2: Text RAG
def predict_text_rag(text, driver):
    # Embedding
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL) # configのモデル
    vector = resp.data[0].embedding
    
    # 単純な類似検索
    query = """
    CALL db.index.vector.queryNodes('concept_index', 3, $vector)
    YIELD node AS concept, score
    RETURN concept.definition AS text, score
    """
    with driver.session() as session:
        result = session.run(query, vector=vector)
        retrieved_texts = [r["text"] for r in result]
    
    context_str = "\n".join(retrieved_texts) if retrieved_texts else "関連情報なし"
    
    prompt = f"""
    あなたは広告倫理の専門家です。以下の「関連する社会的知識」に基づいて、広告コピーのリスクを判定してください。
    【関連知識】
    {context_str}
    【対象広告】
    {text}
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    return _call_llm(prompt)
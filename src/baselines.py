import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# B0: Zero-shot (コンテキストなし、知識なし)
def predict_zero_shot(text):
    prompt = f"""
    あなたは広告倫理の専門家です。以下の広告コピーに「炎上リスク」があるか判定してください。
    
    広告コピー: {text}
    
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    response = client.chat.completions.create(
        model="gpt-4o", # 論文ではGPT-5としていますが、実装上は利用可能な最新モデル
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return int(response.choices[0].message.content.strip())

# B1: Few-shot (事例あり、知識グラフなし)
def predict_few_shot(text, few_shot_examples):
    # few_shot_examplesは (text, label, reason) のリストを想定
    examples_str = "\n".join([f"例: {ex['text']}\n判定: {ex['label']}\n理由: {ex['reason']}" for ex in few_shot_examples])
    
    prompt = f"""
    あなたは広告倫理の専門家です。以下の過去の判定事例を参考に、対象の広告コピーに「炎上リスク」があるか判定してください。
    
    【参考事例】
    {examples_str}
    
    【対象広告】
    {text}
    
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return int(response.choices[0].message.content.strip())

# B2: Text RAG (非構造化文書検索)
# 簡易実装として、Neo4jのベクトルインデックスを使いつつ、グラフ探索（LEADS_TO等）をせずに
# 類似した「定義文(Definition)」だけを取得して判断させる手法
def predict_text_rag(text, driver):
    # 1. 入力テキストの埋め込み
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    vector = resp.data[0].embedding
    
    # 2. 類似テキスト（オントロジーの定義文）の検索（グラフ構造は無視）
    query = """
    CALL db.index.vector.queryNodes('concept_index', 3, $vector)
    YIELD node AS concept, score
    RETURN concept.definition AS text, score
    """
    with driver.session() as session:
        result = session.run(query, vector=vector)
        retrieved_texts = [r["text"] for r in result]
    
    context_str = "\n".join(retrieved_texts)
    
    prompt = f"""
    あなたは広告倫理の専門家です。以下の「関連する社会的知識」に基づいて、広告コピーのリスクを判定してください。
    
    【関連知識】
    {context_str}
    
    【対象広告】
    {text}
    
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return int(response.choices[0].message.content.strip())
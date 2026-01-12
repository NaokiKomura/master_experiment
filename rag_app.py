import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

# 環境変数の読み込み
load_dotenv()

# 設定
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")
AUTH = (neo4j_user, neo4j_password)
URI = neo4j_uri

def get_ad_risk_context(ad_name_keyword):
    """指定された広告に関連するリスクパスをNeo4jから検索する"""
    query = """
    MATCH (ad:Ad)
    WHERE ad.name CONTAINS $keyword
    
    MATCH path = (ad)-[:HAS_EXPRESSION]->(exp:Expression)
                 -[:EVOKES]->(assoc:Association)
                 -[:MAPS_TO]->(conc:Concept)
                 -[:LEADS_TO]->(risk:RiskFactor)
                 -[:VIOLATES]->(norm:Norm)
    
    RETURN 
        exp.description AS expression,
        exp.modality AS modality,
        assoc.name AS association,
        conc.name AS concept,
        conc.definition AS concept_def,
        risk.name AS risk_factor,
        norm.description AS violated_norm
    """
    
    driver = GraphDatabase.driver(URI, auth=AUTH)
    context_list = []
    
    try:
        with driver.session() as session:
            result = session.run(query, keyword=ad_name_keyword)
            for record in result:
                context_list.append({
                    "描写": record["expression"],
                    "媒体": record["modality"],
                    "連想": record["association"],
                    "抵触概念": record["concept"],
                    "概念定義": record["concept_def"],
                    "リスク要因": record["risk_factor"],
                    "法的・倫理的規範": record["violated_norm"]
                })
    finally:
        driver.close()
    return context_list

def generate_risk_report(ad_name, context_data):
    """検索結果(Context)を元に、LLMがリスク評価レポートを生成する"""
    if not context_data:
        return "【分析結果】\n指定された広告に明確なリスクパス（事実→概念→規範のつながり）は見つかりませんでした。"

    context_str = json.dumps(context_data, indent=2, ensure_ascii=False)
    
    system_prompt = """
    あなたは企業の「広告倫理審査アドバイザー」です。
    提供された「Graph RAG検索結果」だけに基づいて、リスク評価レポートを作成してください。
    
    重要：
    - 「グラフに存在するパス」のみを根拠にすること。
    - 描写(Expression)とリスク(Risk)が論理的にどう繋がっているかを解説すること。
    - もし接続が不自然な場合は「グラフ上の接続に疑問がある」と指摘すること。
    """
    
    user_prompt = f"""
    対象広告: {ad_name}
    
    【Graph RAG 検索結果】
    {context_str}
    
    レポートを作成してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    target_ad = "洗剤" 
    print(f"Searching graph for '{target_ad}'...")
    
    context = get_ad_risk_context(target_ad)
    print("Generating report...")
    report = generate_risk_report(target_ad, context)
    
    print("\n" + "="*30)
    print("【Graph RAG リスク評価レポート】")
    print("="*30)
    print(report)
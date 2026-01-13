import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 環境変数の読み込み
load_dotenv()

# 設定
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

if not password:
    raise ValueError("NEO4J_PASSWORD が環境変数に設定されていません。")

AUTH = (user, password)

# ベクトルインデックス作成用クエリ
CREATE_INDEX_QUERY = """
CREATE VECTOR INDEX expression_embedding_index IF NOT EXISTS
FOR (n:Expression)
ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
"""

# データ投入用クエリ
CYPHER_QUERY = """
// --- 0. Core ---
MERGE (ad:Ad {id: $payload.ad_id})
ON CREATE SET ad.name = $payload.meta.name, ad.url = $payload.meta.url

MERGE (med:Medium {name: $payload.meta.medium})
MERGE (ad)-[:PUBLISHED_ON]->(med)

// --- A-2. Context ---
FOREACH (ctxData IN $payload.contexts |
    MERGE (ctx:Context {id: ctxData.id})
    ON CREATE SET ctx.type = ctxData.type, ctx.description = ctxData.desc
    MERGE (ad)-[:HAS_CONTEXT]->(ctx)
)

// --- A-1. Expression ---
WITH ad, $payload.expressions AS exprList
UNWIND exprList AS exprData

MERGE (exp:Expression {id: exprData.id})
ON CREATE SET 
    exp.description = exprData.text, 
    exp.modality = exprData.modality,
    exp.embedding = exprData.embedding

MERGE (ad)-[:HAS_EXPRESSION]->(exp)

// --- A-3. Evidence ---
MERGE (ev:Evidence {id: exprData.evidence.id})
ON CREATE SET ev.quote = exprData.evidence.quote, ev.source = exprData.evidence.source
MERGE (exp)-[:SUPPORTED_BY]->(ev)

// --- A-4. Facts ---
FOREACH (roleData IN exprData.depicted_roles |
    MERGE (role:DepictedRole {name: roleData.name})
    MERGE (exp)-[r:DEPICTS]->(role) SET r.action = roleData.action
)
FOREACH (tgtData IN exprData.targets |
    MERGE (tgt:TargetAudience {name: tgtData.name})
    MERGE (exp)-[:TARGETS]->(tgt)
)

// --- A-5. Evokes ---
FOREACH (evokeData IN exprData.evokes |
    MERGE (assoc:Association {name: evokeData.name})
    MERGE (exp)-[r:EVOKES]->(assoc)
    SET r.confidence = evokeData.confidence, r.salience = evokeData.salience
)

// --- Phase 2-B. Framing ---
FOREACH (frameData IN exprData.framing |
    MERGE (conc:Concept {name: frameData.concept})
    FOREACH (_ IN CASE WHEN frameData.type = 'REINFORCES' THEN [1] ELSE [] END |
        MERGE (exp)-[r:REINFORCES]->(conc) SET r.score = frameData.score, r.reason = frameData.reason
    )
    FOREACH (_ IN CASE WHEN frameData.type = 'NORMALIZES' THEN [1] ELSE [] END |
        MERGE (exp)-[r:NORMALIZES]->(conc) SET r.score = frameData.score, r.reason = frameData.reason
    )
    FOREACH (_ IN CASE WHEN frameData.type = 'IMPLIES' THEN [1] ELSE [] END |
        MERGE (exp)-[r:IMPLIES]->(conc) SET r.score = frameData.score, r.reason = frameData.reason
    )
)
"""

def create_index(driver):
    """ベクトル検索用のインデックスを作成する"""
    with driver.session() as session:
        session.run(CREATE_INDEX_QUERY)
        print("Vector index ensured.")

def load_to_neo4j(payload: dict):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        
        # 最初にインデックスがあるか確認して作成
        create_index(driver)
        
        with driver.session() as session:
            session.run(CYPHER_QUERY, payload=payload)
            print(f"Successfully loaded Ad ID: {payload['ad_id']} with Embeddings")

if __name__ == "__main__":
    from processor import analyze_ad_content
    
    # テストデータ
    sample_copy = "ママの真っ白な洗濯物、家族の笑顔。"
    sample_visual = "明るいリビングで、エプロンをした母親が一人で洗濯物を畳んでいる。"
    
    print("Analyzing with Embeddings...")
    payload = analyze_ad_content(sample_copy, sample_visual)
    
    print("Loading to Neo4j...")
    load_to_neo4j(payload)
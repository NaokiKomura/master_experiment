import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# --- 設定 ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

def _norm_key(s: str) -> str:
    """Normalize a string for use as a Neo4j merge key."""
    if s is None:
        return ""
    # Basic normalization: strip, collapse whitespace
    return " ".join(str(s).strip().split())


def ensure_instance_schema(session) -> None:
    """Create constraints/indexes for input-graph instance nodes (idempotent)."""
    # Uniqueness constraints
    session.run("CREATE CONSTRAINT ad_id_unique IF NOT EXISTS FOR (n:Ad) REQUIRE n.id IS UNIQUE")
    session.run("CREATE CONSTRAINT expr_unique IF NOT EXISTS FOR (n:Expression) REQUIRE (n.ad_id, n.index) IS UNIQUE")
    session.run("CREATE CONSTRAINT evidence_unique IF NOT EXISTS FOR (n:Evidence) REQUIRE (n.ad_id, n.index) IS UNIQUE")
    session.run("CREATE CONSTRAINT ctx_unique IF NOT EXISTS FOR (n:PlacementContext) REQUIRE n.ad_id IS UNIQUE")
    session.run("CREATE CONSTRAINT assoc_name_unique IF NOT EXISTS FOR (n:Association) REQUIRE n.name IS UNIQUE")
    session.run("CREATE CONSTRAINT role_name_unique IF NOT EXISTS FOR (n:DepictedRole) REQUIRE n.name IS UNIQUE")

    # Helpful lookup indexes
    session.run("CREATE INDEX ad_csv_id_idx IF NOT EXISTS FOR (n:Ad) ON (n.csv_id)")
    session.run("CREATE INDEX assoc_name_idx IF NOT EXISTS FOR (n:Association) ON (n.name)")
    session.run("CREATE INDEX role_name_idx IF NOT EXISTS FOR (n:DepictedRole) ON (n.name)")

def load_to_neo4j(payload):
    """
    Processorが分析したJSONペイロードをNeo4jにロードし、
    分析対象の「入力グラフ（Input Graph）」を構築する。
    """
    if not NEO4J_PASSWORD:
        logger.error("NEO4J_PASSWORD is not set.")
        return

    ad_id = payload.get('ad_id')
    if not ad_id:
        logger.error("Payload missing 'ad_id'. Skipping.")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
    
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            ensure_instance_schema(session)
            # 1. Adノードの作成
            # CSV等のメタデータも含めて保存
            meta = payload.get('meta', {})
            logger.info(f"Loading Ad: {ad_id} (CSV_ID: {meta.get('csv_id', 'N/A')})")
            
            session.run("""
                MERGE (ad:Ad {id: $ad_id})
                SET ad.csv_id = $csv_id,
                    ad.brand = $brand,
                    ad.input_text = $input_text,
                    ad.copy_text = $copy_text,
                    ad.timestamp = datetime()
            """,
            ad_id=ad_id,
            csv_id=meta.get('csv_id'),
            brand=meta.get('brand'),
            input_text=payload.get('input_text', ''),
            copy_text=payload.get('input_text', ''))

            # 2. PlacementContext (掲出文脈) の作成
            # 論文の「コンテキストミスマッチ」を判定するための入力ノード
            context = payload.get('context', {})
            if context:
                session.run("""
                    MATCH (ad:Ad {id: $ad_id})
                    MERGE (ctx:PlacementContext {ad_id: $ad_id})
                    SET ctx.media_type = $media,
                        ctx.timing = $timing,
                        ctx.target = $target
                    MERGE (ad)-[:PLACED_IN]->(ctx)
                """, 
                ad_id=ad_id,
                media=context.get('media_type', 'unknown'),
                timing=context.get('timing', 'unknown'),
                target=context.get('target', 'unknown'))

            # 3. Expressions, Associations, Roles, Evidence の展開
            # processor.py の出力構造に依存します
            expressions = payload.get('expressions', [])
            
            for i, expr in enumerate(expressions):
                expr_text = expr.get('text', '')
                if not expr_text: continue

                # Expressionノード（広告内の具体的な表現箇所）
                # インデックスiを使って一意性を担保
                session.run("""
                    MATCH (ad:Ad {id: $ad_id})
                    MERGE (e:Expression {ad_id: $ad_id, index: $idx})
                    SET e.text = $text
                    MERGE (ad)-[:HAS_EXPRESSION]->(e)
                """, ad_id=ad_id, idx=i, text=expr_text)

                # --- Association (連想) ---
                # mapper.py で処理対象となる重要ノード
                # 名前(name)でMERGEすることで、全広告共通の「連想語プール」を作る（Embedding計算コスト削減）
                for assoc_text in expr.get('associations', []):
                    assoc_key = _norm_key(assoc_text)
                    if not assoc_key:
                        continue
                    session.run("""
                        MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                        MERGE (a:Association {name: $name})
                        ON CREATE SET a.raw = $raw
                        MERGE (e)-[:EVOKES]->(a)
                    """, ad_id=ad_id, idx=i, name=assoc_key, raw=assoc_text)

                # --- DepictedRole (描かれた役割) ---
                # オントロジーのRoleConceptに紐づく入力ノード
                for role_text in expr.get('roles', []):
                    role_key = _norm_key(role_text)
                    if not role_key:
                        continue
                    session.run("""
                        MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                        MERGE (r:DepictedRole {name: $name})
                        ON CREATE SET r.raw = $raw
                        MERGE (e)-[:DEPICTS]->(r)
                    """, ad_id=ad_id, idx=i, name=role_key, raw=role_text)

                # --- Evidence (根拠) ---
                # LLMが抽出した「なぜそう判断したか」の引用テキスト
                evidence_text = expr.get('evidence', '')
                if evidence_text:
                    session.run("""
                        MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                        MERGE (ev:Evidence {ad_id: $ad_id, index: $idx})
                        SET ev.text = $text
                        MERGE (ev)-[:SUPPORTS]->(e)
                    """, ad_id=ad_id, idx=i, text=evidence_text)

            logger.info(f"Successfully loaded Ad {ad_id} and its graph structure.")

    except Exception as e:
        logger.error(f"Error loading to Neo4j: {e}")
    finally:
        if 'driver' in locals():
            driver.close()

def clear_ad_data(driver=None):
    """
    実験用：オントロジー（知識）を残したまま、入力された広告データ（インスタンス）のみを削除する
    """
    query = """
    MATCH (n)
    WHERE n:Ad OR n:Expression OR n:PlacementContext OR n:Evidence OR n:DepictedRole
    DETACH DELETE n
    """
    # Note: Association は他でも使われる可能性があるため、孤立した場合のみ消すなどの配慮がいるが、
    # 実験のリセットとしては Association も消して再生成させるのが安全
    query_assoc = """
    MATCH (a:Association)
    WHERE NOT (a)-[:MAPS_TO]->(:Concept)
      AND NOT (:Expression)-[:EVOKES]->(a)
    DETACH DELETE a
    """
    
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
        close_driver = True

    try:
        with driver.session() as session:
            logger.info("Clearing Ad instance data...")
            session.run(query)
            # マッピング済みでないAssociationも掃除する場合
            # session.run(query_assoc) 
            logger.info("Ad data cleared.")
    finally:
        if close_driver:
            driver.close()

if __name__ == "__main__":
    # テスト用ダミーデータ
    dummy_payload = {
        "ad_id": "test_uuid_12345",
        "input_text": "家事はママの仕事、がんばって。",
        "meta": {"csv_id": "999", "brand": "TestBrand"},
        "context": {"media_type": "TVCM", "timing": "Morning"},
        "expressions": [
            {
                "text": "家事はママの仕事",
                "associations": ["性別役割分業", "ワンオペ育児", "母親の負担"],
                "roles": ["母親", "主婦"],
                "evidence": "「ママの仕事」と明言している点"
            }
        ]
    }
    load_to_neo4j(dummy_payload)
import os
import sys
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from neo4j import GraphDatabase

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# --- 設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# 実験パラメータ (論文の再現性に関わる重要定数)
SIMILARITY_THRESHOLD = 0.72   # 類似度閾値
MARGIN_THRESHOLD = 0.02       # 2位との差がこれ未満なら「曖昧」と判定
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_CANDIDATES = 3          # 候補取得数

# クライアント初期化 (事前チェック付き)
if not OPENAI_API_KEY:
    logger.error("CRITICAL: OPENAI_API_KEY is not set. Aborting.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding_with_retry(text, max_retries=3):
    """OpenAI APIを用いて埋め込みを取得する（リトライ処理付き）"""
    if not text: return None

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return response.data[0].embedding
        except (RateLimitError, APIConnectionError, APIError) as e:
            wait_time = 2 ** attempt
            logger.warning(f"API Error ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error for text '{text[:10]}...': {e}")
            return None
    
    logger.error(f"Failed to get embedding after {max_retries} attempts.")
    return None

def check_preconditions(driver):
    """インデックス存在確認など、実行前の健全性チェック"""
    try:
        with driver.session() as session:
            # concept_index の存在確認
            result = session.run("SHOW VECTOR INDEXES YIELD name WHERE name = 'concept_index'")
            if not result.single():
                logger.error("CRITICAL: Vector index 'concept_index' not found in Neo4j.")
                logger.error("Please run 'ontology_loader.py' first to create indexes.")
                return False
            logger.info("Pre-flight check passed: 'concept_index' exists.")
            return True
    except Exception as e:
        logger.error(f"Pre-flight check failed: {e}")
        return False

def map_associations_to_concepts():
    if not NEO4J_PASSWORD:
        logger.error("CRITICAL: NEO4J_PASSWORD is not set.")
        sys.exit(1)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        logger.error(f"Neo4j Connection Failed: {e}")
        return

    # 事前チェック
    if not check_preconditions(driver):
        driver.close()
        sys.exit(1)

    with driver.session() as session:
        # 1. 未マッピングのAssociationを取得
        # 推論済みの MAPS_TO または CANDIDATE_OF があるものはスキップ
        fetch_query = """
            MATCH (a:Association)
            WHERE NOT (a)-[:MAPS_TO {source: 'inference'}]->(:Concept)
              AND NOT (a)-[:CANDIDATE_OF {source: 'inference'}]->(:Concept)
            RETURN elementId(a) AS id, 
                   COALESCE(a.name, a.text) AS text, 
                   a.embedding AS embedding
        """
        result = session.run(fetch_query)
        associations = list(result)
        
        logger.info(f"Found {len(associations)} unmapped associations.")
        
        for record in associations:
            assoc_id = record["id"]
            assoc_text = record["text"]
            current_embedding = record["embedding"]
            
            if not assoc_text:
                continue

            logger.info(f"Processing '{assoc_text}'...")

            # 2. Embeddingの準備
            vector = current_embedding
            if not vector:
                vector = get_embedding_with_retry(assoc_text)
                if vector:
                    # 生成したEmbeddingを保存（再利用のため）
                    session.run("MATCH (a:Association) WHERE elementId(a) = $id SET a.embedding = $vector", 
                                id=assoc_id, vector=vector)
                else:
                    continue

            # 3. ベクトル検索 (Top-K & Margin)
            search_query = """
            CALL db.index.vector.queryNodes('concept_index', $k, $vector)
            YIELD node AS concept, score
            RETURN concept.name AS concept_name, score
            ORDER BY score DESC
            """
            
            matches = list(session.run(search_query, k=TOP_K_CANDIDATES, vector=vector))
            
            if not matches:
                logger.info("  -> No matches found.")
                continue

            best_match = matches[0]
            best_score = best_match["score"]
            best_concept = best_match["concept_name"]

            # マージン計算
            margin = 0.0
            second_concept_log = ""
            if len(matches) > 1:
                second_score = matches[1]["score"]
                margin = best_score - second_score
                second_concept_log = f", 2nd: {matches[1]['concept_name']} ({second_score:.4f})"

            logger.info(f"  -> Top: {best_concept} ({best_score:.4f}){second_concept_log}, Margin: {margin:.4f}")

            # 4. 意思決定とリンク作成
            if best_score < SIMILARITY_THRESHOLD:
                logger.info(f"  -> Below threshold ({SIMILARITY_THRESHOLD}). Skipped.")
                continue

            # 改善点: source='inference' を指定し、Exemplarの上書きを防止
            # 改善点: Margin判定でリレーションタイプを分岐
            if margin >= MARGIN_THRESHOLD:
                # 確信度が高い -> MAPS_TO (確定)
                rel_type = "MAPS_TO"
                log_msg = "Creating Confirmed Link (MAPS_TO)"
            else:
                # 確信度は高いが競合あり -> CANDIDATE_OF (候補/要確認)
                # 論文の「誤検知抑制」に対応
                rel_type = "CANDIDATE_OF"
                log_msg = f"Creating Tentative Link (CANDIDATE_OF) - Low Margin < {MARGIN_THRESHOLD}"

            logger.info(f"  -> {log_msg}")

            # クエリ構築 (動的リレーションタイプはPython側で文字列埋め込みするが、安全のためif分岐で制御)
            # source, similarity, threshold, margin, top_k を全て保存
            if rel_type == "MAPS_TO":
                query = """
                MATCH (a:Association) WHERE elementId(a) = $assoc_id
                MATCH (c:Concept {name: $concept_name})
                MERGE (a)-[r:MAPS_TO {source: 'inference'}]->(c)
                SET r.similarity = $score,
                    r.threshold_used = $threshold,
                    r.margin = $margin,
                    r.top_k = $top_k,
                    r.embedding_model = $model,
                    r.timestamp = datetime()
                """
            else:
                query = """
                MATCH (a:Association) WHERE elementId(a) = $assoc_id
                MATCH (c:Concept {name: $concept_name})
                MERGE (a)-[r:CANDIDATE_OF {source: 'inference'}]->(c)
                SET r.similarity = $score,
                    r.threshold_used = $threshold,
                    r.margin = $margin,
                    r.top_k = $top_k,
                    r.embedding_model = $model,
                    r.timestamp = datetime()
                """

            session.run(query, 
                        assoc_id=assoc_id, 
                        concept_name=best_concept, 
                        score=best_score,
                        threshold=SIMILARITY_THRESHOLD,
                        margin=margin,
                        top_k=TOP_K_CANDIDATES,
                        model=EMBEDDING_MODEL)

    driver.close()
    logger.info("Mapping process completed.")

if __name__ == "__main__":
    map_associations_to_concepts()

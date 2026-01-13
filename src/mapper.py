import os
import sys
import time
import logging
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from neo4j import GraphDatabase

try:
    from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_AUTH, EMBEDDING_MODEL
except ImportError:
    from .config import OPENAI_API_KEY, NEO4J_URI, NEO4J_AUTH, EMBEDDING_MODEL

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 実験パラメータ
SIMILARITY_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.02
TOP_K_CANDIDATES = 5

if not OPENAI_API_KEY:
    logger.error("CRITICAL: OPENAI_API_KEY is not set.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# (以下、関数定義は前回の回答と同じロジックですが、設定変数はconfigから取得したものを使います)
def get_embedding_with_retry(text, max_retries=3):
    if not text: return None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return response.data[0].embedding
        except (RateLimitError, APIConnectionError, APIError) as e:
            time.sleep(2 ** attempt)
        except Exception as e:
            return None
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
    if not NEO4J_AUTH[1]:
        logger.error("CRITICAL: NEO4J_PASSWORD is not set.")
        sys.exit(1)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
    except Exception as e:
        logger.error(f"Neo4j Connection Failed: {e}")
        return

    # 事前チェック
    if not check_preconditions(driver):
        driver.close()
        sys.exit(1)

    with driver.session() as session:
        # 1. 未マッピング取得
        fetch_query = """
            MATCH (a:Association)
            WHERE NOT (a)-[:MAPS_TO {source: 'inference'}]->(:Concept)
              AND NOT (a)-[:CANDIDATE_OF {source: 'inference'}]->(:Concept)
            RETURN elementId(a) AS id, COALESCE(a.name, a.text) AS text, a.embedding AS embedding
        """
        result = session.run(fetch_query)
        associations = list(result)
        logger.info(f"Found {len(associations)} unmapped associations.")
        
        for record in associations:
            assoc_id = record["id"]
            assoc_text = record["text"]
            current_embedding = record["embedding"]
            
            if not assoc_text: continue

            vector = current_embedding
            if not vector:
                vector = get_embedding_with_retry(assoc_text)
                if vector:
                    session.run("MATCH (a:Association) WHERE elementId(a) = $id SET a.embedding = $vector", 
                                id=assoc_id, vector=vector)
                else:
                    continue

            # ベクトル検索
            search_query = """
            CALL db.index.vector.queryNodes('concept_index', $k, $vector)
            YIELD node AS concept, score
            RETURN concept.name AS concept_name, score
            ORDER BY score DESC
            """
            matches = list(session.run(search_query, k=TOP_K_CANDIDATES, vector=vector))
            
            if not matches: continue

            best_match = matches[0]
            best_score = best_match["score"]
            best_concept = best_match["concept_name"]
            margin = 0.0
            if len(matches) > 1:
                margin = best_score - matches[1]["score"]

            if best_score < SIMILARITY_THRESHOLD: continue

            rel_type = "MAPS_TO" if margin >= MARGIN_THRESHOLD else "CANDIDATE_OF"
            
            query = f"""
            MATCH (a:Association) WHERE elementId(a) = $assoc_id
            MATCH (c:Concept {{name: $concept_name}})
            MERGE (a)-[r:{rel_type} {{source: 'inference'}}]->(c)
            SET r.similarity = $score, r.margin = $margin, r.threshold_used = $threshold, r.timestamp = datetime()
            """
            session.run(query, assoc_id=assoc_id, concept_name=best_concept, 
                        score=best_score, margin=margin, threshold=SIMILARITY_THRESHOLD)

    driver.close()
    logger.info("Mapping process completed.")

if __name__ == "__main__":
    map_associations_to_concepts()
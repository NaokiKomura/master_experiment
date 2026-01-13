from __future__ import annotations

import sys
import time
import logging
from typing import Any, Dict, List, Optional

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

# import 時点では落とさない（batch環境でimportだけしたいケースがあるため）
_OPENAI_CLIENT: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OPENAI_CLIENT

def get_embedding_with_retry(text: str, max_retries: int = 3) -> Optional[List[float]]:
    """OpenAI Embedding 取得（簡易リトライ付き）"""
    if not text:
        return None
    client = _get_openai_client()
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return list(response.data[0].embedding)
        except (RateLimitError, APIConnectionError, APIError) as e:
            time.sleep(2 ** attempt)
        except Exception as e:
            return None
    return None

def check_preconditions(driver: Any) -> bool:
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

def map_associations_to_concepts(
    driver: Optional[Any] = None,
    *,
    ad_id: Optional[str] = None,
    top_k: int = TOP_K_CANDIDATES,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Association -> Concept のマッピング（推論リンク）を作成する。

    - Streamlit デモからは引数無しで呼び出せる（全Associationを対象）
    - batch_experiment.py からは driver/ad_id を渡して広告単位で処理できる

    Args:
        driver: Neo4j driver（Noneの場合は config から生成して内部でcloseする）
        ad_id: 指定時はその広告からEVOKESされるAssociationだけ対象
        top_k: ベクトル検索候補数
        similarity_threshold: best_score がこの値以上のときだけリンクを張る
        margin_threshold: (best - second) がこの値以上なら MAPS_TO、未満なら CANDIDATE_OF
        overwrite: 既存の推論リンク（source='inference'）を削除して再作成する

    Returns:
        dict: 実行サマリ
    """
    if driver is None:
        if not NEO4J_AUTH[1]:
            raise RuntimeError("NEO4J_PASSWORD is not set.")
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        close_driver = True
    else:
        close_driver = False

    try:
        driver.verify_connectivity()
        if not check_preconditions(driver):
            raise RuntimeError("Pre-flight check failed: concept_index not found.")

        with driver.session() as session:
            if ad_id is None:
                fetch_query = """
                    MATCH (a:Association)
                    WHERE NOT (a)-[:MAPS_TO {source: 'inference'}]->(:Concept)
                      AND NOT (a)-[:CANDIDATE_OF {source: 'inference'}]->(:Concept)
                    RETURN elementId(a) AS id, COALESCE(a.name, a.text) AS text, a.embedding AS embedding
                """
                associations = list(session.run(fetch_query))
            else:
                if overwrite:
                    fetch_query = """
                        MATCH (ad:Ad {id: $ad_id})-[:HAS_EXPRESSION]->(:Expression)-[:EVOKES]->(a:Association)
                        RETURN DISTINCT elementId(a) AS id, COALESCE(a.name, a.text) AS text, a.embedding AS embedding
                    """
                    associations = list(session.run(fetch_query, ad_id=str(ad_id)))
                else:
                    fetch_query = """
                        MATCH (ad:Ad {id: $ad_id})-[:HAS_EXPRESSION]->(:Expression)-[:EVOKES]->(a:Association)
                        WHERE NOT (a)-[:MAPS_TO {source: 'inference'}]->(:Concept)
                          AND NOT (a)-[:CANDIDATE_OF {source: 'inference'}]->(:Concept)
                        RETURN DISTINCT elementId(a) AS id, COALESCE(a.name, a.text) AS text, a.embedding AS embedding
                    """
                    associations = list(session.run(fetch_query, ad_id=str(ad_id)))

            logger.info(f"Found {len(associations)} candidate associations. (ad_id={ad_id})")

            mapped = 0
            skipped = 0
            for record in associations:
                assoc_id = record["id"]
                assoc_text = record["text"]
                current_embedding = record.get("embedding")

                if not assoc_text:
                    skipped += 1
                    continue

                if overwrite:
                    session.run(
                        """
                        MATCH (a:Association) WHERE elementId(a) = $id
                        OPTIONAL MATCH (a)-[r:MAPS_TO|CANDIDATE_OF {source: 'inference'}]->(:Concept)
                        DELETE r
                        """,
                        id=assoc_id,
                    )

                vector = current_embedding
                if not vector:
                    vector = get_embedding_with_retry(str(assoc_text))
                    if vector:
                        session.run(
                            "MATCH (a:Association) WHERE elementId(a) = $id SET a.embedding = $vector",
                            id=assoc_id,
                            vector=vector,
                        )
                    else:
                        skipped += 1
                        continue

                matches = list(
                    session.run(
                        """
                        CALL db.index.vector.queryNodes('concept_index', $k, $vector)
                        YIELD node AS concept, score
                        RETURN concept.name AS concept_name, score
                        ORDER BY score DESC
                        """,
                        k=int(top_k),
                        vector=vector,
                    )
                )
                if not matches:
                    skipped += 1
                    continue

                best_score = float(matches[0]["score"])
                best_concept = matches[0]["concept_name"]
                margin = 0.0
                if len(matches) > 1:
                    try:
                        margin = best_score - float(matches[1]["score"])
                    except Exception:
                        margin = 0.0

                if best_score < float(similarity_threshold):
                    skipped += 1
                    continue

                rel_type = "MAPS_TO" if margin >= float(margin_threshold) else "CANDIDATE_OF"
                session.run(
                    f"""
                    MATCH (a:Association) WHERE elementId(a) = $assoc_id
                    MATCH (c:Concept {{name: $concept_name}})
                    MERGE (a)-[r:{rel_type} {{source: 'inference'}}]->(c)
                    SET r.similarity = $score,
                        r.margin = $margin,
                        r.threshold_used = $threshold,
                        r.timestamp = datetime()
                    """,
                    assoc_id=assoc_id,
                    concept_name=best_concept,
                    score=best_score,
                    margin=float(margin),
                    threshold=float(similarity_threshold),
                )
                mapped += 1

        logger.info("Mapping process completed.")
        return {
            "ad_id": str(ad_id) if ad_id is not None else None,
            "n_candidates": len(associations),
            "mapped": int(mapped),
            "skipped": int(skipped),
            "top_k": int(top_k),
            "similarity_threshold": float(similarity_threshold),
            "margin_threshold": float(margin_threshold),
            "overwrite": bool(overwrite),
        }

    finally:
        if close_driver:
            driver.close()

if __name__ == "__main__":
    map_associations_to_concepts()
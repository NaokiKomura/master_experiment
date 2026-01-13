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

            # context_concept_index の存在確認（PlacementContext -> ContextConcept 用）
            result2 = session.run("SHOW VECTOR INDEXES YIELD name WHERE name = 'context_concept_index'")
            if not result2.single():
                logger.error("CRITICAL: Vector index 'context_concept_index' not found in Neo4j.")
                logger.error("Please create it (e.g., via ontology_loader.py or Neo4j Browser) before mapping PlacementContext.")
                return False

            logger.info("Pre-flight check passed: 'concept_index' and 'context_concept_index' exist.")
            return True
    except Exception as e:
        logger.error(f"Pre-flight check failed: {e}")
        return False

def map_placement_contexts_to_context_concepts(
    driver,
    ad_id=None,
    top_k: int = 8,
    similarity_threshold: float = 0.70,
    overwrite: bool = False,
):
    """
    PlacementContext -> ContextConcept のベクトル検索マッピングを作成する。

    - MAPS_TO: score >= similarity_threshold
    - CANDIDATE_OF: score >= (similarity_threshold - 0.10) かつ MAPS_TO 未満

    前提:
    - Neo4j に vector index `context_concept_index` が存在する
    - ContextConcept.embedding が格納済み（ontology_loader で投入）
    """
    with driver.session() as session:
        # 対象PlacementContext取得（広告単位 or 全件）
        if ad_id is not None:
            pcs = session.run(
                """
                MATCH (ad:Ad {id:$ad_id})-[:PLACED_IN]->(pc:PlacementContext)
                RETURN elementId(pc) AS pc_eid,
                       pc.media_type AS media_type,
                       pc.timing AS timing,
                       pc.target AS target
                """,
                ad_id=str(ad_id),
            )
        else:
            pcs = session.run(
                """
                MATCH (pc:PlacementContext)
                RETURN elementId(pc) AS pc_eid,
                       pc.media_type AS media_type,
                       pc.timing AS timing,
                       pc.target AS target
                """
            )

        pcs = list(pcs)

        # overwrite 対応（既存リンク削除）
        if overwrite:
            if ad_id is not None:
                session.run(
                    """
                    MATCH (ad:Ad {id:$ad_id})-[:PLACED_IN]->(pc:PlacementContext)-[r:MAPS_TO|CANDIDATE_OF]->(:ContextConcept)
                    DELETE r
                    """,
                    ad_id=str(ad_id),
                )
            else:
                session.run(
                    """
                    MATCH (pc:PlacementContext)-[r:MAPS_TO|CANDIDATE_OF]->(:ContextConcept)
                    DELETE r
                    """
                )

        min_score = float(similarity_threshold) - 0.10

        for rec in pcs:
            pc_eid = rec["pc_eid"]
            # PlacementContext をテキスト化（embedding用）
            text = " ".join(
                [
                    rec.get("media_type") or "",
                    rec.get("timing") or "",
                    rec.get("target") or "",
                ]
            ).strip()
            if not text:
                continue

            # 既存 mapper.py の埋め込み生成関数に合わせて呼び出し名を調整してください
            # 例: emb = get_embedding_with_retry(text)
            emb = get_embedding_with_retry(text)  # ←あなたの mapper.py に存在する関数名に合わせる
            if emb is None:
                continue

            session.run(
                """
                MATCH (pc) WHERE elementId(pc) = $pc_eid

                CALL db.index.vector.queryNodes('context_concept_index', $k, $emb)
                YIELD node AS cc, score

                WITH pc, cc, score
                WHERE score >= $min_score

                MERGE (pc)-[rel:CANDIDATE_OF {source:'inference'}]->(cc)
                SET rel.similarity = score,
                    rel.margin = score - $threshold

                FOREACH (_ IN CASE WHEN score >= $threshold THEN [1] ELSE [] END |
                    SET rel:MAPS_TO
                )
                """,
                pc_eid=pc_eid,
                k=int(top_k),
                emb=emb,
                min_score=float(min_score),
                threshold=float(similarity_threshold),
            )

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

            # --- PlacementContext -> ContextConcept のマッピング ---
            # Association のマッピングとは独立に実行する（論文の文脈推論を有効化するため）
            try:
                map_placement_contexts_to_context_concepts(
                    driver,
                    ad_id=ad_id,
                    top_k=int(top_k),
                    similarity_threshold=float(similarity_threshold),
                    overwrite=bool(overwrite),
                )
            except Exception as e:
                logger.error(f"PlacementContext mapping failed (ad_id={ad_id}): {e}")

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
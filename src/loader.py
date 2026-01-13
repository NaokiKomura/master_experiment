from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from neo4j import GraphDatabase
try:
    from config import NEO4J_URI, NEO4J_AUTH
except ImportError:
    from .config import NEO4J_URI, NEO4J_AUTH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _flatten_meta(meta: Any) -> Dict[str, Any]:
    """processor.py が返す meta 構造を loader 向けに平坦化する。

    processor.AdContentProcessor.analyze_ad_content() では
    meta={"model":..., "original_meta": {...}} のようにネストして返る。
    既存の load_to_neo4j は meta.get('csv_id') 等を想定しているため、ここで吸収する。
    """
    if not isinstance(meta, dict):
        return {}
    orig = meta.get("original_meta")
    if isinstance(orig, dict):
        merged = dict(orig)
        # processing meta も残す（必要なら参照できる）
        merged["_processing_meta"] = {k: v for k, v in meta.items() if k != "original_meta"}
        return merged
    return meta


def _clear_existing_ad_instance(session: Any, ad_id: str) -> None:
    """指定 ad_id のインスタンス（広告入力由来ノード）を削除する。

    - Association / DepictedRole は他広告と共有し得るため削除しない
    - Evidence/Expression/Context/Ad を削除
    """
    session.run(
        """
        MATCH (ad:Ad {id: $ad_id})
        OPTIONAL MATCH (ad)-[:PLACED_IN]->(ctx:PlacementContext)
        OPTIONAL MATCH (ad)-[:HAS_EXPRESSION]->(e:Expression)
        OPTIONAL MATCH (ev:Evidence)-[:SUPPORTS]->(e)
        DETACH DELETE ad, ctx, e, ev
        """,
        ad_id=ad_id,
    )


def upsert_ad_instance(
    driver: Any,
    *,
    ad_id: str,
    text: str,
    tag: int,
    label: Optional[int],
    facts: Dict[str, Any],
    overwrite: bool = False,
) -> Dict[str, Any]:
    """バッチ実験用: Neo4j に広告インスタンスを投入する。

    Args:
        driver: Neo4j driver（呼び出し側で接続済み）
        ad_id: 広告ID（CSVのID等）
        text: 元広告本文
        tag: 正解ラベル（2値: 0/1）
        label: 炎上分類（任意）
        facts: processor.extract_facts が返す payload
        overwrite: 同一ad_idが既に存在する場合、削除して作り直す

    Returns:
        dict: ロード結果の簡易サマリ
    """
    payload = dict(facts or {})
    payload["ad_id"] = str(ad_id)
    payload["input_text"] = text
    payload["meta"] = _flatten_meta(payload.get("meta"))

    if not payload.get("meta"):
        payload["meta"] = {"csv_id": str(ad_id)}
    payload.setdefault("context", {})
    payload.setdefault("expressions", [])

    with driver.session() as session:
        if overwrite:
            _clear_existing_ad_instance(session, str(ad_id))

        meta = payload.get("meta", {})

        # 1. Adノード
        session.run(
            """
            MERGE (ad:Ad {id: $ad_id})
            SET ad.csv_id = $csv_id,
                ad.brand = $brand,
                ad.copy_text = $copy_text,
                ad.tag_true = $tag_true,
                ad.label_true = $label_true,
                ad.timestamp = datetime()
            """,
            ad_id=str(ad_id),
            csv_id=meta.get("csv_id"),
            brand=meta.get("brand"),
            copy_text=text,
            tag_true=int(tag),
            label_true=int(label) if label is not None else None,
        )

        # 2. PlacementContext
        context = payload.get("context", {}) or {}
        if context:
            session.run(
                """
                MATCH (ad:Ad {id: $ad_id})
                MERGE (ctx:PlacementContext {ad_id: $ad_id})
                SET ctx.media_type = $media,
                    ctx.timing = $timing,
                    ctx.target = $target
                MERGE (ad)-[:PLACED_IN]->(ctx)
                """,
                ad_id=str(ad_id),
                media=context.get("media_type", "unknown"),
                timing=context.get("timing", "unknown"),
                target=context.get("target", "unknown"),
            )

        # 3. Expressions / Associations / Roles / Evidence
        expressions = payload.get("expressions", []) or []
        for i, expr in enumerate(expressions):
            expr_text = (expr or {}).get("text", "")
            if not expr_text:
                continue

            session.run(
                """
                MATCH (ad:Ad {id: $ad_id})
                MERGE (e:Expression {ad_id: $ad_id, index: $idx})
                SET e.text = $text
                MERGE (ad)-[:HAS_EXPRESSION]->(e)
                """,
                ad_id=str(ad_id),
                idx=i,
                text=expr_text,
            )

            for assoc_text in (expr or {}).get("associations", []) or []:
                session.run(
                    """
                    MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                    MERGE (a:Association {name: $name})
                    MERGE (e)-[:EVOKES]->(a)
                    """,
                    ad_id=str(ad_id),
                    idx=i,
                    name=assoc_text,
                )

            for role_text in (expr or {}).get("roles", []) or []:
                session.run(
                    """
                    MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                    MERGE (r:DepictedRole {name: $name})
                    MERGE (e)-[:DEPICTS]->(r)
                    """,
                    ad_id=str(ad_id),
                    idx=i,
                    name=role_text,
                )

            evidence_text = (expr or {}).get("evidence", "")
            if evidence_text:
                session.run(
                    """
                    MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                    MERGE (ev:Evidence {ad_id: $ad_id, index: $idx})
                    SET ev.text = $text
                    MERGE (ev)-[:SUPPORTS]->(e)
                    """,
                    ad_id=str(ad_id),
                    idx=i,
                    text=evidence_text,
                )

    return {
        "ad_id": str(ad_id),
        "n_expressions": len(payload.get("expressions", []) or []),
        "overwrite": bool(overwrite),
    }

def load_to_neo4j(payload):
    if not NEO4J_AUTH[1]:
        logger.error("NEO4J_PASSWORD is not set.")
        return

    ad_id = payload.get('ad_id')
    if not ad_id: return

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            # 1. Adノードの作成
            # CSV等のメタデータも含めて保存
            meta = payload.get('meta', {})
            logger.info(f"Loading Ad: {ad_id} (CSV_ID: {meta.get('csv_id', 'N/A')})")
            
            session.run("""
                MERGE (ad:Ad {id: $ad_id})
                SET ad.csv_id = $csv_id,
                    ad.brand = $brand,
                    ad.copy_text = $copy_text,
                    ad.timestamp = datetime()
            """, 
            ad_id=ad_id, 
            csv_id=meta.get('csv_id'), 
            brand=meta.get('brand'),
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
                    session.run("""
                        MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                        MERGE (a:Association {name: $name})
                        MERGE (e)-[:EVOKES]->(a)
                    """, ad_id=ad_id, idx=i, name=assoc_text)

                # --- DepictedRole (描かれた役割) ---
                # オントロジーのRoleConceptに紐づく入力ノード
                for role_text in expr.get('roles', []):
                    session.run("""
                        MATCH (e:Expression {ad_id: $ad_id, index: $idx})
                        MERGE (r:DepictedRole {name: $name})
                        MERGE (e)-[:DEPICTS]->(r)
                    """, ad_id=ad_id, idx=i, name=role_text)

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
    WHERE NOT (a)--(:Concept)  // 概念にマッピングされていない（入力由来の）連想のみ削除
    DETACH DELETE a
    """
    
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
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
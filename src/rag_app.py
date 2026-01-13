"""rag_app.py

Streamlit デモアプリ + バッチ実験用の Graph RAG（リスクパス抽出）API。

- Streamlit 実行: srcディレクトリで `streamlit run rag_app.py`
- バッチ実行（例: batch_experiment.py）からは `extract_risk_paths()` を呼び出す

方針:
- import 時に Streamlit を必須にしない（バッチ環境で import エラーにならない）
- Graph RAG の中核は Neo4j クエリでパスを返す
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI

# configからインポート
from config import NEO4J_URI, NEO4J_AUTH, OPENAI_API_KEY

# 自作モジュールのインポート (同一ディレクトリにある前提)
from processor import AdContentProcessor
from loader import load_to_neo4j, clear_ad_data
from mapper import map_associations_to_concepts

# Streamlit はデモ実行時のみ必要（import-safe）
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None

def generate_risk_explanation(input_text, risk_paths, era):
    """
    検索されたグラフパス(根拠)に基づいて、リスクの説明文を生成する
    """
    if not risk_paths:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    # グラフのパス情報をコンテキスト化
    context_str = ""
    for i, path in enumerate(risk_paths):
        context_str += f"""
        [Path {i+1}]
        - 表現: {path['expression']}
        - 連想: {path['association']}
        - 抵触概念: {path['concept']} (定義: {path['definition']})
        - 炎上要因: {path['risk_label']}
        - 違反規範: {path['norm']}
        - 影響集団: {', '.join(path['affected_groups'])}
        """

    system_prompt = f"""
    あなたは広告リスク管理の専門コンサルタントです。
    ユーザーが入力した広告コピーに対し、知識グラフから検出された「リスクの根拠（推論パス）」が提供されます。
    これに基づき、マーケティング担当者向けの「リスク評価レポート」を作成してください。

    【制約事項】
    1. 提供された[Path]情報のみを根拠にしてください（ハルシネーション禁止）。
    2. 判定基準の時代は「{era}」です。その時代の価値観に沿って解説してください。
    3. 結論を先に述べ、その後に具体的な理由を記述してください。
    4. トーンは客観的かつ論理的に。
    """

    user_prompt = f"""
    【対象広告コピー】
    {input_text}

    【検出されたリスクパス（根拠）】
    {context_str}

    上記に基づき、この広告がなぜ炎上リスクを持つのか、具体的に説明してください。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {e}"

# --- Batch/CLI usable API ---

def _compute_risk_score(paths: List[Dict[str, Any]]) -> float:
    """リスクスコア（連続値）をパス集合から作る。

    目的:
    - PR-AUC 等のランキング指標が算出できるよう、0/1以外のスコアを提供する

    現状のスコア定義（シンプル）:
    - MAPS_TO/CANDIDATE_OF の similarity の最大値を採用
    - similarity が無い場合は 0.0

    ※論文側で別定義（例: margin を加味、パス数加点等）にしたい場合はここを差し替える。
    """
    sims: List[float] = []
    for p in paths:
        try:
            v = p.get("similarity", None)
            if v is None:
                continue
            sims.append(float(v))
        except Exception:
            continue
    return max(sims) if sims else 0.0


def extract_risk_paths(
    driver: Any,
    ad_id: str,
    max_paths: int = 20,
    era: str = "2020s",
) -> Dict[str, Any]:
    """Graph RAG 相当: 指定広告(ad_id)についてリスク推論パスを抽出する。

    batch_experiment.py から利用することを想定した関数。

    Returns:
        {
          "risk_score": float,           # 連続値（ランキング用）
          "paths": List[dict],           # 根拠パス（最大 max_paths）
          "era": str,
          "ad_id": str,
        }

    備考:
    - 2値判定は batch_experiment.py 側で `len(paths)>0 or risk_score>0` として行う。
    """
    paths = get_risk_analysis(driver, ad_id, era, limit=50)

    # max_paths 制限（重要: DB側で LIMIT していないためここで絞る）
    if max_paths is not None and max_paths > 0:
        paths = paths[: int(max_paths)]

    risk_score = _compute_risk_score(paths)
    return {
        "risk_score": float(risk_score),
        "paths": list(paths),
        "era": era,
        "ad_id": ad_id,
    }

# --- 検索機能 (Retrieval) ---

def get_risk_analysis(driver, ad_id: str, era: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    指定された時代(era)に基づいてリスクパスを探索する
    """
    query = """
    MATCH (ad:Ad {id: $ad_id})
    // 1. 広告表現から連想へ
    MATCH (ad)-[:HAS_EXPRESSION]->(expr:Expression)-[:EVOKES]->(assoc:Association)
    
    // 2. 連想から概念へ (推論リンク または 知識リンク)
    MATCH (assoc)-[link:MAPS_TO|CANDIDATE_OF]->(concept:Concept)
    
    // 3. 概念からリスク・規範へ
    MATCH (concept)-[:LEADS_TO]->(risk:RiskFactor)-[:VIOLATES]->(norm:Norm)
    OPTIONAL MATCH (risk)-[:OFFENDS]->(group:AffectedGroup)

    // 4. 時代のフィルタリング
    WHERE $era IN concept.valid_eras
    
    RETURN 
        expr.text as expression,
        assoc.name as association,
        type(link) as link_type,
        link.similarity as similarity,
        link.margin as margin,
        concept.name as concept,
        concept.definition as definition,
        risk.label as risk_label,
        norm.name as norm,
        collect(DISTINCT group.name) as affected_groups
    ORDER BY similarity DESC, risk_label
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, ad_id=ad_id, era=era, limit=int(limit))
        return [record.data() for record in result]

def main():
    if st is None:
        raise RuntimeError("streamlit がインストールされていません。デモ起動には `pip install streamlit` が必要です。")

    st.set_page_config(page_title="Ad Risk Graph RAG Demo", layout="wide")

    st.title("🛡️ Ad Risk Analysis System")
    
    show_debug = False

    with st.sidebar:
        selected_era = st.selectbox("📅 判定基準の時代", ["2020s", "2010s"])
        if st.button("Clear Data"):
            clear_ad_data()
            st.success("Cleared.")

    col1, col2 = st.columns(2)
    with col1:
        input_text = st.text_area("広告コピーを入力", height=150)
        analyze_btn = st.button("Analyze")

    if analyze_btn and input_text:
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # --- Step 1: Processor ---
            processor = AdContentProcessor()
            status_text.text("Step 1/3: Extracting facts from text (LLM)...")
            progress_bar.progress(30)
            
            meta = {"csv_id": "DEMO_APP", "brand": "DemoBrand"}
            payload = processor.analyze_ad_content(input_text, meta)
            ad_id = payload['ad_id']
            
            if show_debug:
                with col1:
                    st.json(payload)

            # --- Step 2: Loader ---
            status_text.text("Step 2/3: Loading structure to Knowledge Graph...")
            progress_bar.progress(60)
            load_to_neo4j(payload)

            # --- Step 3: Mapper ---
            status_text.text("Step 3/3: Inferring semantic connections (Vector Search)...")
            progress_bar.progress(80)
            map_associations_to_concepts()
            
            progress_bar.progress(100)
            status_text.text("Analysis Complete.")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            # --- Step 4: Graph RAG (Retrieve & Generate) ---
            driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            results = get_risk_analysis(driver, ad_id, selected_era, limit=50)
            driver.close()

            with col2:
                st.subheader(f"2. Analysis Results ({selected_era})")
                
                if not results:
                    st.success("✅ No significant risks detected in this era.")
                    st.info("※ 時代設定を変えるとリスクが検知される可能性があります。")
                else:
                    # --- 追加機能: 生成されたリスク説明の表示 ---
                    st.markdown("### 📝 AI Risk Assessment")
                    with st.spinner("Generating explanation..."):
                        explanation = generate_risk_explanation(input_text, results, selected_era)
                        st.info(explanation)

                    # --- 既存機能: 詳細パスの表示 ---
                    st.markdown("### 🔍 Evidence Paths (Graph Trace)")
                    df = pd.DataFrame(results)
                    for risk_label in df['risk_label'].unique():
                        st.write(f"**🔥 {risk_label}**")
                        subset = df[df['risk_label'] == risk_label]
                        for _, row in subset.iterrows():
                            with st.expander(f"表現: 「{row['expression']}」 → 概念: {row['concept']}"):
                                st.markdown(f"""
                                - **連想**: {row['association']}
                                - **抵触した概念**: {row['concept']}
                                  - 定義: *{row['definition']}*
                                - **違反規範**: {row['norm']}
                                - **影響集団**: {', '.join(row['affected_groups'])}
                                - **判定タイプ**: {row['link_type']} (Similarity: {row['similarity']:.3f})
                                """)

        except Exception as e:
            st.error(f"Error occurred: {e}")

    st.markdown("---")
    st.markdown("### 📊 System Logic")
    st.caption("""
    1. **Fact Extraction**: 広告文から事実を抽出
    2. **Graph Mapping**: 社会的概念へ接続
    3. **Path Finding**: 炎上パスを探索
    4. **Explanation**: 根拠パスに基づき解説を生成
    """)

if __name__ == "__main__":
    main()
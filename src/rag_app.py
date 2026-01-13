# streamlit run rag_app.py で実行可能

import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# 自作モジュールのインポート
from processor import AdContentProcessor
from loader import load_to_neo4j, clear_ad_data
from mapper import map_associations_to_concepts

# 環境変数の読み込み
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ページ設定
st.set_page_config(page_title="Ad Risk Graph RAG Demo", layout="wide")

# --- 生成機能 (Generation) ---

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

# --- 検索機能 (Retrieval) ---

def get_risk_analysis(driver, ad_id, era):
    """
    指定された時代(era)に基づいてリスクパスを探索する
    (修正: NormやGroupが欠けていてもパスを表示するようにOPTIONAL MATCH化)
    """
    query = """
    MATCH (ad:Ad {id: $ad_id})
    // 1. 広告表現から連想へ (必須)
    MATCH (ad)-[:HAS_EXPRESSION]->(expr:Expression)-[:EVOKES]->(assoc:Association)
    
    // 2. 連想から概念へ (必須: ここが切れていればリスクなし判定で正しい)
    MATCH (assoc)-[link:MAPS_TO|CANDIDATE_OF]->(concept:Concept)
    
    // 3. 概念からリスク要因へ (必須: ここまで繋がれば「リスクあり」とみなす)
    MATCH (concept)-[:LEADS_TO]->(risk:RiskFactor)

    // 4. 時代のフィルタリング
    WHERE (concept.valid_eras IS NULL OR $era IN concept.valid_eras)

    // --- 修正箇所: ここから下を OPTIONAL MATCH に変更 ---
    // 規範や影響集団が未定義でも、RiskFactorまで到達していれば表示する
    
    OPTIONAL MATCH (risk)-[:VIOLATES]->(norm:Norm)
    OPTIONAL MATCH (risk)-[:OFFENDS]->(group:AffectedGroup)
    
    RETURN 
        expr.text as expression,
        assoc.name as association,
        type(link) as link_type,
        link.similarity as similarity,
        concept.name as concept,
        concept.definition as definition,
        risk.label as risk_label,
        // normやgroupがない場合は「未定義」等の文字列を返すようCoalesceする
        coalesce(norm.name, "規範定義なし") as norm,
        collect(DISTINCT group.name) as affected_groups
    ORDER BY risk_label
    """
    with driver.session() as session:
        result = session.run(query, ad_id=ad_id, era=era)
        return [record.data() for record in result]

# --- メインアプリケーション ---

def main():
    st.title("🛡️ Ad Risk Analysis System (Graph RAG)")
    st.markdown("論文「Graph RAGを用いた広告炎上リスクの分析」プロトタイプ")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_era = st.selectbox("📅 判定基準の時代 (Era)", ["2020s", "2010s"], index=0)
        st.divider()
        show_debug = st.checkbox("Show Graph Payload", value=False)
        if st.button("Clear Cache & Data"):
            clear_ad_data()
            st.success("Ad data cleared.")

    # 入力エリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Ad Copy")
        default_text = "家事はママの仕事、がんばって。家族のために。"
        input_text = st.text_area("広告コピーを入力", value=default_text, height=150)
        analyze_btn = st.button("🔍 Analyze Risk", type="primary")

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
            driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
            results = get_risk_analysis(driver, ad_id, selected_era)
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
import streamlit as st
import time
import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI

# configからインポート
from config import NEO4J_URI, NEO4J_AUTH, OPENAI_API_KEY

# 自作モジュールのインポート (同一ディレクトリにある前提)
from processor import AdContentProcessor
from loader import load_to_neo4j, clear_ad_data
from mapper import map_associations_to_concepts

st.set_page_config(page_title="Ad Risk Graph RAG Demo", layout="wide")

# ... (generate_risk_explanation, get_risk_analysis 関数は前回のロジックを使用)
# get_risk_analysis 内のクエリは OPTIONAL MATCH を使用した最新版を使ってください

def main():
    st.title("🛡️ Ad Risk Analysis System")
    
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
        try:
            # 1. Process
            processor = AdContentProcessor()
            payload = processor.analyze_ad_content(input_text, {"csv_id": "DEMO"})
            ad_id = payload['ad_id']
            
            # 2. Load
            load_to_neo4j(payload)
            
            # 3. Map
            map_associations_to_concepts()
            
            # 4. RAG
            driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            # ここで get_risk_analysis を呼び出す
            # results = get_risk_analysis(driver, ad_id, selected_era)
            driver.close()
            
            # 結果表示ロジック...

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
import pandas as pd
import time
import os
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report
from neo4j import GraphDatabase

# モジュールのインポート
from processor import analyze_ad_content
from loader import load_to_neo4j
from mapper import map_associations_to_concepts
from ontology_loader import setup_ontology

# 環境変数の読み込み
load_dotenv()

# 設定
CSV_FILE = "research_data.csv"
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")
AUTH = (neo4j_user, neo4j_password)
URI = neo4j_uri

def reset_database(driver):
    """DBを初期化し、オントロジーのみ再ロードする"""
    print("\n[Step 1] Initializing Database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print(" -> Database cleared.")
    
    # オントロジー（知識）の再ロード
    setup_ontology()
    print(" -> Ontology re-loaded.")

def process_batch_data(df):
    """CSVデータを順次処理してNeo4jに投入する"""
    print(f"\n[Step 2] Processing {len(df)} ads from CSV...")
    
    ad_id_map = {} 
    
    for index, row in df.iterrows():
        input_text = row['copy_candidates']
        true_label = row['Tag']
        
        print(f" -> [{index+1}/{len(df)}] Analyzing: {input_text[:15]}...")
        
        try:
            payload = analyze_ad_content(input_text, f"（テキスト内容を参照: {input_text}）")
            
            payload['meta']['csv_id'] = row['id']
            load_to_neo4j(payload)
            
            ad_id_map[payload['ad_id']] = {
                'csv_id': row['id'],
                'text': input_text,
                'true_label': true_label
            }
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error analyzing row {index}: {e}")
    
    return ad_id_map

def evaluate_results(driver, ad_id_map):
    """リスクパスの有無を確認し、精度を計算する"""
    print("\n[Step 4] Evaluating Risk Detection Accuracy...")
    
    y_true = []
    y_pred = []
    results = []

    with driver.session() as session:
        for ad_uuid, info in ad_id_map.items():
            query = """
            MATCH (ad:Ad {id: $ad_id})
            MATCH path = (ad)-[:HAS_EXPRESSION]->()-[:EVOKES]->()-[:MAPS_TO]->()-[:LEADS_TO]->(risk:RiskFactor)
            RETURN count(path) > 0 as is_risky, collect(distinct risk.name) as risks
            """
            
            result = session.run(query, ad_id=ad_uuid).single()
            is_risky = 1 if result['is_risky'] else 0
            detected_risks = result['risks']
            
            y_true.append(info['true_label'])
            y_pred.append(is_risky)
            
            results.append({
                'csv_id': info['csv_id'],
                'text': info['text'],
                'true': info['true_label'],
                'pred': is_risky,
                'risks': detected_risks,
                'result': 'Correct' if info['true_label'] == is_risky else 'Miss'
            })

    print("\n" + "="*40)
    print("【検証結果レポート】")
    print("="*40)
    
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print(f"Confusion Matrix:\n{cm}")
        print(" [TN, FP]")
        print(" [FN, TP]")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Safe (0)', 'Risky (1)']))
    except Exception as e:
        print(f"Evaluation Metrics Error: {e}")
    
    print("\n【誤判定リスト (Miss Cases)】")
    misses = [r for r in results if r['result'] == 'Miss']
    if not misses:
        print("なし (全問正解！)")
    else:
        for m in misses:
            status = "FP (過検知)" if m['pred'] == 1 else "FN (見逃し)"
            print(f"- {status}: ID={m['csv_id']} 「{m['text'][:20]}...」")
            if m['pred'] == 1:
                print(f"  -> 誤って検出されたリスク: {m['risks']}")

    return results

def main():
    df = pd.read_csv(CSV_FILE)
    
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"Neo4j Connection Failed: {e}")
        return

    try:
        # 1. DB初期化
        reset_database(driver)
        
        # 2. データ処理 & 投入
        ad_id_map = process_batch_data(df)
        
        # 3. マッピング実行
        print("\n[Step 3] Mapping Associations to Ontology...")
        map_associations_to_concepts()
        
        # 4. 評価
        evaluate_results(driver, ad_id_map)
        
    finally:
        driver.close()

if __name__ == "__main__":
    main()
import pandas as pd
import time
import os
import json
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score
from neo4j import GraphDatabase
from openai import OpenAI

# 既存モジュールのインポート
# ※ baselines.py を作らず、ここに比較手法関数を統合しています
from processor import analyze_ad_content
from loader import load_to_neo4j
from mapper import map_associations_to_concepts
from ontology_loader import setup_ontology  # ontology_loader側で引数(version)を受け取れる前提

# 環境変数の読み込み
load_dotenv()

# 設定
CSV_FILE = "research_data.csv"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")
AUTH = (neo4j_user, neo4j_password)
URI = neo4j_uri

# ==========================================
# 比較手法の実装 (B0, B1, B2) - 論文 4.5節
# ==========================================

def predict_zero_shot(text):
    """B0: Zero-shot (ベースライン)"""
    prompt = f"""
    あなたは広告倫理の専門家です。以下の広告コピーに「炎上リスク」があるか判定してください。
    
    広告コピー: {text}
    
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return 0

def predict_few_shot(text):
    """B1: Few-shot (コンテキスト学習)"""
    # 論文の付録や本文にあるような事例を埋め込みます
    examples = """
    例1:
    コピー: 「家事はママの仕事。」
    判定: 1
    理由: 性別役割分業の固定化

    例2:
    コピー: 「誰でも使いやすい洗剤。」
    判定: 0
    理由: 機能訴求であり問題なし
    """
    
    prompt = f"""
    あなたは広告倫理の専門家です。以下の過去の判定事例を参考に、対象の広告コピーに「炎上リスク」があるか判定してください。
    
    【参考事例】
    {examples}
    
    【対象広告】
    {text}
    
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return 0

def predict_text_rag(text, driver):
    """B2: Text RAG (非構造化RAG)"""
    # 1. 埋め込み作成
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    vector = resp.data[0].embedding
    
    # 2. 類似定義文の検索 (グラフ構造は無視し、Conceptの定義文のみ取得)
    query = """
    CALL db.index.vector.queryNodes('concept_index', 3, $vector)
    YIELD node AS concept, score
    RETURN concept.definition AS text, score
    """
    with driver.session() as session:
        result = session.run(query, vector=vector)
        retrieved_texts = [r["text"] for r in result]
    
    context_str = "\n".join(retrieved_texts) if retrieved_texts else "関連情報なし"
    
    prompt = f"""
    あなたは広告倫理の専門家です。以下の「関連する社会的知識」に基づいて、広告コピーのリスクを判定してください。
    
    【関連知識】
    {context_str}
    
    【対象広告】
    {text}
    
    リスクがある場合は「1」、ない場合は「0」とだけ出力してください。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return 0

# ==========================================
# 共通ユーティリティ
# ==========================================

def calculate_metrics(y_true, y_pred, y_scores=None):
    """論文表5に対応する評価指標を計算"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    pr_auc = 0.0
    if y_scores is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "PR-AUC": pr_auc
    }

def process_and_load_data(df, use_cache=True):
    """
    CSVデータを処理してNeo4jに投入する。
    use_cache=Trueの場合、一度分析したpayloadをメモリに保存して再利用する（実験3用）
    """
    print(f"\nProcessing {len(df)} ads...")
    ad_id_map = {} 
    
    # グローバルキャッシュ（簡易的）
    if not hasattr(process_and_load_data, "cache"):
        process_and_load_data.cache = {}

    for index, row in df.iterrows():
        input_text = row['copy_candidates']
        csv_id = row['id']
        true_label = row['Tag']
        
        # キャッシュ確認
        if use_cache and csv_id in process_and_load_data.cache:
            payload = process_and_load_data.cache[csv_id]
            # IDだけは新規生成(Neo4j内での重複回避)も可能だが、
            # 今回はDELETEしてから入れる前提なので同じIDでOK
        else:
            print(f" -> [{index+1}/{len(df)}] Analyzing (LLM): {input_text[:15]}...")
            try:
                payload = analyze_ad_content(input_text, f"（テキスト参照: {input_text}）")
                payload['meta']['csv_id'] = csv_id
                process_and_load_data.cache[csv_id] = payload
                time.sleep(0.5)
            except Exception as e:
                print(f"    Error analyzing row {index}: {e}")
                continue

        # Neo4jへの投入
        load_to_neo4j(payload)
        
        ad_id_map[payload['ad_id']] = {
            'csv_id': csv_id,
            'text': input_text,
            'true_label': true_label
        }
    
    return ad_id_map

# ==========================================
# 実験メイン処理
# ==========================================

def run_experiment_2_comparative(driver, df):
    """実験2：手法間比較 (4.5節)"""
    print("\n" + "="*50)
    print("【実験2】システムの信頼性・効率性評価 (Comparison)")
    print("="*50)

    # 1. 提案手法（P）の準備: DB初期化 & データ投入 & マッピング
    print("\n[Setup] Preparing Graph RAG environment...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n") # DBクリア
    setup_ontology(version="present") # 現在のオントロジー
    
    ad_id_map = process_and_load_data(df, use_cache=True)
    map_associations_to_concepts()

    results = {
        "B0_ZeroShot": {"true": [], "pred": []},
        "B1_FewShot":  {"true": [], "pred": []},
        "B2_TextRAG":  {"true": [], "pred": []},
        "P_GraphRAG":  {"true": [], "pred": []}
    }

    print("\n[Execution] Running predictions for all methods...")
    
    with driver.session() as session:
        for ad_uuid, info in ad_id_map.items():
            text = info['text']
            true_label = info['true_label']
            
            # --- B0: Zero-shot ---
            results["B0_ZeroShot"]["true"].append(true_label)
            results["B0_ZeroShot"]["pred"].append(predict_zero_shot(text))

            # --- B1: Few-shot ---
            results["B1_FewShot"]["true"].append(true_label)
            results["B1_FewShot"]["pred"].append(predict_few_shot(text))

            # --- B2: Text RAG ---
            results["B2_TextRAG"]["true"].append(true_label)
            results["B2_TextRAG"]["pred"].append(predict_text_rag(text, driver))

            # --- P: Graph RAG (Proposed) ---
            query = """
            MATCH (ad:Ad {id: $ad_id})
            MATCH path = (ad)-[:HAS_EXPRESSION]->()-[:EVOKES]->()-[:MAPS_TO]->()-[:LEADS_TO]->(risk:RiskFactor)
            RETURN count(path) > 0 as is_risky
            """
            res = session.run(query, ad_id=ad_uuid).single()
            p_pred = 1 if res and res['is_risky'] else 0
            
            results["P_GraphRAG"]["true"].append(true_label)
            results["P_GraphRAG"]["pred"].append(p_pred)

    # 結果表示
    print(f"\n{'Method':<15} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'PR-AUC':<6}")
    print("-" * 65)
    for method, data in results.items():
        # 注: PR-AUC計算には本来確率値が必要ですが、簡易的に0/1ラベルを使います
        metrics = calculate_metrics(data["true"], data["pred"], data["pred"])
        print(f"{method:<15} | {metrics['Accuracy']:.3f}  | {metrics['Precision']:.3f}  | {metrics['Recall']:.3f}  | {metrics['F1']:.3f}  | {metrics['PR-AUC']:.3f}")


def run_experiment_3_sensitivity(driver, df):
    """実験3：動的リスク感度分析 (4.6節)"""
    print("\n" + "="*50)
    print("【実験3】動的リスクに対する感度分析 (Sensitivity Analysis)")
    print("="*50)
    
    versions = ["past", "present"]
    history = {}

    for ver in versions:
        print(f"\n--- Testing Ontology Version: {ver.upper()} ---")
        
        # 1. 環境リセット & 指定バージョンのオントロジーロード
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        # setup_ontologyが引数(version)を受け取るように ontology_loader.py を修正済みである前提
        try:
            setup_ontology(version=ver) 
        except TypeError:
            print("Warning: setup_ontology does not accept version. Using default.")
            setup_ontology()

        # 2. データ投入 (キャッシュ利用で高速化)
        ad_id_map = process_and_load_data(df, use_cache=True)
        
        # 3. マッピング (オントロジーが変わるので結果が変わる)
        map_associations_to_concepts()
        
        # 4. リスク判定
        detected_risks = {}
        with driver.session() as session:
            for ad_uuid, info in ad_id_map.items():
                query = """
                MATCH (ad:Ad {id: $ad_id})
                MATCH path = (ad)-[:HAS_EXPRESSION]->()-[:EVOKES]->()-[:MAPS_TO]->()-[:LEADS_TO]->(risk:RiskFactor)
                RETURN count(path) as path_count
                """
                res = session.run(query, ad_id=ad_uuid).single()
                score = res['path_count'] # パス数を簡易リスクスコアとする
                detected_risks[info['csv_id']] = score
        
        history[ver] = detected_risks

    # 分析結果の比較
    print("\n【感度分析レポート】")
    print(f"{'ID':<5} | {'Text (Head)':<20} | {'Past Score':<10} | {'Present Score':<12} | {'Change'}")
    print("-" * 70)
    
    flip_count = 0
    total_risky_now = 0
    
    for csv_id in history["present"]:
        score_past = history["past"].get(csv_id, 0)
        score_now = history["present"].get(csv_id, 0)
        
        # リスクなし(0) -> リスクあり(>0) への変化
        is_flip = (score_past == 0 and score_now > 0)
        if is_flip: flip_count += 1
        if score_now > 0: total_risky_now += 1
        
        # 変化があったもの、あるいは現在リスクがあるものを表示
        if is_flip or score_now > 0:
            # テキスト取得用
            text_disp = ""
            for _, v in process_and_load_data.cache.items():
                 if v['meta']['csv_id'] == csv_id:
                     text_disp = v['expressions'][0]['text'][:20]
                     break
            
            change_str = "DETECTED (Flip)" if is_flip else "Same"
            print(f"{csv_id:<5} | {text_disp:<20} | {score_past:<10} | {score_now:<12} | {change_str}")

    print("-" * 70)
    print(f"Total Label Flips (Safe -> Risky): {flip_count}")
    print(f"Total Risky Ads (Present): {total_risky_now}")


def main():
    # データ読み込み
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return
    
    df = pd.read_csv(CSV_FILE)
    
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        
        # 実験選択メニュー
        print("Select Experiment to Run:")
        print("1: Experiment 2 (Comparative Evaluation)")
        print("2: Experiment 3 (Sensitivity Analysis)")
        print("3: Run Both")
        choice = input("Enter choice (1-3): ")

        if choice in ["1", "3"]:
            run_experiment_2_comparative(driver, df)
        
        if choice in ["2", "3"]:
            run_experiment_3_sensitivity(driver, df)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    main()
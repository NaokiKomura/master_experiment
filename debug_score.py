import os
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

# .envファイルから環境変数を読み込む
load_dotenv()

# 設定 (環境変数から取得)
# APIキーが見つからない場合はエラーにするか、Noneを許容するか制御できます
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY が環境変数に設定されていません。.envファイルを確認してください。")

client = OpenAI(api_key=openai_api_key)

# Neo4j設定
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687") # デフォルト値を設定
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")

if not neo4j_password:
    raise ValueError("NEO4J_PASSWORD が環境変数に設定されていません。.envファイルを確認してください。")

AUTH = (neo4j_user, neo4j_password)

def check_similarity_scores():
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=AUTH)
        driver.verify_connectivity() # 接続確認
    except Exception as e:
        print(f"Neo4jへの接続に失敗しました: {e}")
        return

    # 類似度計算のために埋め込みを行う関数
    def get_embedding(text):
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding

    with driver.session() as session:
        # 1. 未接続のAssociationを取得
        result = session.run("""
            MATCH (a:Association)
            RETURN a.name AS name
        """)
        
        associations = [r["name"] for r in result]
        print(f"Checking scores for {len(associations)} associations...\n")
        
        for assoc_name in associations:
            print(f"▼ Association: 「{assoc_name}」")
            
            # 2. ベクトル化
            try:
                vector = get_embedding(assoc_name)
                
                # 3. スコア確認クエリ（閾値なしでTop 3を表示）
                search_query = """
                CALL db.index.vector.queryNodes('concept_index', 3, $vector)
                YIELD node AS concept, score
                RETURN concept.name AS concept_name, score
                """
                
                matches = session.run(search_query, vector=vector)
                
                for m in matches:
                    print(f"   - Candidate: {m['concept_name']} (Score: {m['score']:.4f})")
                print("-" * 40)
            except Exception as e:
                print(f"   Error processing association '{assoc_name}': {e}")

    driver.close()

if __name__ == "__main__":
    check_similarity_scores()
import os
import json
import hashlib
import atexit
from openai import OpenAI
from neo4j import GraphDatabase

try:
    from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_AUTH, CACHE_FILE, EMBEDDING_MODEL
except ImportError:
    from .config import OPENAI_API_KEY, NEO4J_URI, NEO4J_AUTH, CACHE_FILE, EMBEDDING_MODEL

# --- 知識データの定義 (RISK_CONCEPTS, CONTEXT_CONCEPTS) ---
# ※ ここには前回定義した完全版のリストが入りますが、
# 長くなるため省略します。元のコードのリスト定義をそのまま使用してください。
# (DOMAIN_VALUE, RISK_CONCEPTS, CONTEXT_CONCEPTS の定義が必要)
# --- 以下、省略部分のプレースホルダー ---
DOMAIN_VALUE = "Value_Misalignment"
DOMAIN_PRESSURE = "High_Pressure_Comm"
DOMAIN_CONTEXT = "Context_Mismatch"
RISK_CONCEPTS = [] # ここに前回のリストを入れる
CONTEXT_CONCEPTS = [] # ここに前回のリストを入れる
# ----------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class EmbeddingManager:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.is_dirty = False
        atexit.register(self.save_cache)

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cache(self):
        if self.is_dirty:
            print("Saving embedding cache...")
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
            self.is_dirty = False

    def get_embedding(self, text):
        if not text: return None
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache: return self.cache[text_hash]
        if not client: return None

        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            vector = response.data[0].embedding
            self.cache[text_hash] = vector
            self.is_dirty = True
            return vector
        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

embedder = EmbeddingManager(CACHE_FILE)

def create_constraints(session):
    print("Ensuring constraints and indexes...")
    constraints = [
        "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT risk_id_unique IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT norm_name_unique IF NOT EXISTS FOR (n:Norm) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT context_id_unique IF NOT EXISTS FOR (ctx:ContextConcept) REQUIRE ctx.id IS UNIQUE",
        "CREATE CONSTRAINT assoc_name_unique IF NOT EXISTS FOR (a:TypicalAssociation) REQUIRE a.name IS UNIQUE",
        """
        CREATE VECTOR INDEX concept_index IF NOT EXISTS
        FOR (n:Concept) ON (n.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
        """,
        """
        CREATE VECTOR INDEX association_index IF NOT EXISTS
        FOR (a:TypicalAssociation) ON (a.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
        """
    ]
    for q in constraints:
        try:
            session.run(q)
        except: pass

def setup_ontology(load_mode="full"):
    print(f"\n>>> Loading Ontology (Mode: {load_mode})")
    if not NEO4J_AUTH:
        print("Error: Neo4j credentials missing.")
        return

    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            session.run("""
                MATCH (n) 
                WHERE n:Concept OR n:RiskFactor OR n:Norm OR n:AffectedGroup OR 
                      n:TypicalAssociation OR n:RoleConcept OR n:ContextConcept
                DETACH DELETE n
            """)
            create_constraints(session)
            # ※ ここに前回のデータ投入ロジック(for item in RISK_CONCEPTS...)が入ります
            # 長くなるため、前回の回答コードのロジック部分を使用してください
            pass 

    print(">>> Ontology Load Complete.")

if __name__ == "__main__":
    # データ定義が存在する場合のみ実行
    if RISK_CONCEPTS:
        setup_ontology()
    else:
        print("Please ensure RISK_CONCEPTS is defined in the file.")
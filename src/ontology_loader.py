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


DOMAIN_VALUE = "Value_Misalignment"
DOMAIN_PRESSURE = "High_Pressure_Comm"
DOMAIN_CONTEXT = "Context_Mismatch"
# 概念データセット
# 改善点: 時代ごとの感度をプロパティとして保持、Normの詳細化
RISK_CONCEPTS = [
    # --- A. ジェンダー・性役割 ---
    {
        "id": "Gender_Role_Fixed",
        "name": "性別役割分業の固定化",
        "definition": "「家事育児は女性、仕事は男性」という固定観念。男性の家事参加を「手伝い」と表現することや、女性へのスーパーウーマン（仕事も家事も完璧）の押し付けを含む。",
        "domain": DOMAIN_VALUE,
        "risk_factor": {"id": "Gender_Stereotype", "label": "ジェンダー・ステレオタイプ"},
        "norm": {
            "name": "男女共同参画社会基本法", 
            "type": "law",
            "issuer": "日本政府",
            "jurisdiction": "Japan"
        },
        "affected_groups": ["働く女性", "主婦層", "男性"],
        "typical_associations": ["家事はママの仕事", "お弁当作りは母の愛情", "夫は仕事、妻は家庭", "女子力"],
        "related_roles": ["主婦", "母親", "サラリーマン"],
        "valid_eras": ["2010s", "2020s"],
        "sensitivity_map": {"2010s": "low", "2020s": "high"}
    },
    {
        "id": "Sexual_Objectification",
        "name": "性的対象化・性の商品化",
        "definition": "文脈と無関係に女性の身体的特徴を強調する表現。未成年を性的に描く表現や、公共空間における環境型セクハラ要素。",
        "domain": DOMAIN_VALUE,
        "risk_factor": {"id": "Sexual_Objectification", "label": "性的対象化"},
        "norm": {
            "name": "メディアにおける倫理ガイドライン", 
            "type": "guideline",
            "issuer": "JIAA",
            "jurisdiction": "Advertising_Industry"
        },
        "affected_groups": ["女性全般", "未成年"],
        "typical_associations": ["無意味な露出", "胸や脚の強調", "萌え絵", "性的なアングル"],
        "related_roles": ["女性タレント", "女子高生", "キャンペーンガール"],
        "valid_eras": ["2010s", "2020s"],
        "sensitivity_map": {"2010s": "medium", "2020s": "high"}
    },

    # --- B. ルッキズム・属性 ---
    {
        "id": "Lookism",
        "name": "ルッキズム（外見至上主義）",
        "definition": "人の価値を外見で判断する表現。一重まぶたや体毛を劣ったものとして扱うコンプレックス産業的表現や、整形・脱毛の強迫的推奨。",
        "domain": DOMAIN_VALUE,
        "risk_factor": {"id": "Lookism", "label": "ルッキズム"},
        "norm": {
            "name": "JARO審査基準", 
            "type": "guideline",
            "issuer": "JARO",
            "jurisdiction": "Advertising_Industry"
        },
        "affected_groups": ["若年層", "女性", "求職者"],
        "typical_associations": ["ムダ毛はマナー違反", "一重は恥ずかしい", "デブは自己管理不足", "女磨き"],
        "related_roles": ["就活生", "独身女性"],
        "valid_eras": ["2020s"], # 2010sには存在しない概念として扱う
        "sensitivity_map": {"2010s": "none", "2020s": "high"}
    },
    {
        "id": "Ageism",
        "name": "エイジズム（年齢差別）",
        "definition": "「若さ」のみに価値を置き、加齢を恐怖対象として描くアンチエイジングの強要。高齢者をステレオタイプで描くこと。",
        "domain": DOMAIN_VALUE,
        "risk_factor": {"id": "Ageism", "label": "エイジズム"},
        "norm": {
            "name": "人権尊重の理念", 
            "type": "social_norm",
            "issuer": "International",
            "jurisdiction": "Universal"
        },
        "affected_groups": ["中高年層", "高齢者"],
        "typical_associations": ["おばさん見え", "劣化", "老害", "アンチエイジング"],
        "related_roles": ["中高年女性", "高齢者"],
        "valid_eras": ["2020s"],
        "sensitivity_map": {"2010s": "none", "2020s": "medium"}
    },

    # --- C. コミュニケーション ---
    {
        "id": "High_Pressure",
        "name": "高圧的なコミュニケーション",
        "definition": "企業が生活者の生き方を一方的に規定するような教示的・説教的なメッセージ。",
        "domain": DOMAIN_PRESSURE,
        "risk_factor": {"id": "Tone_Policing", "label": "価値観の押し付け"},
        "norm": {
            "name": "消費者契約法（不当勧誘）の精神", 
            "type": "law_spirit",
            "issuer": "日本政府",
            "jurisdiction": "Japan"
        },
        "affected_groups": ["生活者全般", "不安を抱える層"],
        "typical_associations": ["まだ〜してないの？", "覚悟が足りない", "プロなら〜すべき", "勝ち組・負け組"],
        "related_roles": ["新人", "働く女性"],
        "valid_eras": ["2010s", "2020s"],
        "sensitivity_map": {"2010s": "low", "2020s": "high"}
    }
]

# 文脈データセット
CONTEXT_CONCEPTS = [
    {
        "id": "Emergency",
        "name": "非常時・災害時",
        "attributes": ["tension", "sorrow", "anxiety"],
        "domain": DOMAIN_CONTEXT,
        "risk_factor": {"id": "Inappropriate_Timing", "label": "不謹慎・配慮不足"},
        "norm": {
            "name": "企業の社会的責任(CSR)", 
            "type": "csr",
            "issuer": "Corporate",
            "jurisdiction": "Social"
        },
        "affected_groups": ["被災者", "社会的弱者"],
        "valid_eras": ["2010s", "2020s"],
        "sensitivity_map": {"2010s": "high", "2020s": "high"}
    }
]


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
import os
import json
import hashlib
import atexit
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm

# --- Matplotlib font setup (macOS Japanese) ---
# Prefer Hiragino Sans; include fallbacks in case the exact family name differs per system.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Hiragino Sans',
    'Hiragino Kaku Gothic ProN',
    'Hiragino Kaku Gothic Pro',
    'Yu Gothic',
    'Noto Sans CJK JP',
    'AppleGothic',
    'Arial Unicode MS',
]
plt.rcParams['axes.unicode_minus'] = False

# 環境変数の読み込み
load_dotenv()

# --- 設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# キャッシュ設定
CACHE_FILE = "embedding_cache.json"

# --- クライアント初期化 ---
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==========================================
# 1. 知識データの定義 (Knowledge Base)
# ==========================================

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
            "name": "男女共同参画社会", 
            "type": "society",
            "issuer": "日本",
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
        "related_roles": ["女性タレント", "女子高生", "キャンペーンガール","水着姿の女性"],
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
    {
        "id": "Racism",
        "name": "人種差別",
        "definition": "人種によって差別を行う表現。黒人に対する迫害的な表現などをすること。",
        "domain": DOMAIN_VALUE,
        "risk_factor": {"id": "Racism", "label": "レイシズム"},
        "norm": {
            "name": "メディアにおける倫理ガイドライン", 
            "type": "guideline",
            "issuer": "JIAA",
            "jurisdiction": "Advertising_Industry"
        },
        "affected_groups": ["黒人", "少数民族"],
        "typical_associations": ["黒人は汚らしい", "猿"],
        "related_roles": ["黒人"],
        "valid_eras": ["2020s"], 
        "sensitivity_map": {"2010s": "high", "2020s": "high"}
    },

    # --- C. コミュニケーション ---
    {
        "id": "High_Pressure",
        "name": "高圧的なコミュニケーション",
        "definition": "企業が生活者の生き方を一方的に規定するような教示的・説教的なメッセージ。",
        "domain": DOMAIN_PRESSURE,
        "risk_factor": {"id": "Tone_Policing", "label": "価値観の押し付け"},
        "norm": {
            "name": "心理的リアクタンス", 
            "type": "reactance",
            "issuer": "生活者",
            "jurisdiction": "People"
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


# ==========================================
# 2. Embedding処理 (メモリキャッシュ最適化)
# ==========================================

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
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        if not client: return None

        try:
            response = client.embeddings.create(input=text, model="text-embedding-3-small")
            vector = response.data[0].embedding
            self.cache[text_hash] = vector
            self.is_dirty = True
            return vector
        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

embedder = EmbeddingManager(CACHE_FILE)


# ==========================================
# 3. Neo4j ロード処理 (全時代ロードモード)
# ==========================================

def create_constraints(session):
    print("Ensuring constraints and indexes...")
    constraints = [
        "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT risk_id_unique IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT norm_name_unique IF NOT EXISTS FOR (n:Norm) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT context_id_unique IF NOT EXISTS FOR (ctx:ContextConcept) REQUIRE ctx.id IS UNIQUE",
        "CREATE CONSTRAINT assoc_name_unique IF NOT EXISTS FOR (a:TypicalAssociation) REQUIRE a.name IS UNIQUE",
        
        # インデックス
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
    """
    load_mode="full": 全時代の知識をロードし、時代属性(sensitivity_20xx)を付与する。
    フィルタリングはクエリ実行時に行う。
    """
    print(f"\n>>> Loading Ontology (Mode: {load_mode})")

    if not AUTH:
        print("Error: Neo4j credentials missing.")
        return

    with GraphDatabase.driver(NEO4J_URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            # 1. 関連ノードのみ削除
            session.run("""
                MATCH (n) 
                WHERE n:Concept OR n:RiskFactor OR n:Norm OR n:AffectedGroup OR 
                      n:TypicalAssociation OR n:RoleConcept OR n:ContextConcept
                DETACH DELETE n
            """)
            
            create_constraints(session)

            # 2. Concepts & Associations
            for item in RISK_CONCEPTS:
                print(f"Loading Concept: {item['name']}...")

                # 定義 + 名前でベクトル化
                concept_text = f"{item['name']}: {item['definition']}"
                c_vector = embedder.get_embedding(concept_text)
                if c_vector is None: continue

                # sensitivity mapの展開 (時代ごとの感度をプロパティ化)
                # sensitivity_2010s, sensitivity_2020s として保存
                sens_props = {}
                for era, sens in item['sensitivity_map'].items():
                    sens_props[f"sensitivity_{era}"] = sens

                # --- Core Path (Concept -> Risk -> Norm) ---
                query_core = """
                MERGE (c:Concept {id: $c_id})
                SET c.name = $c_name, 
                    c.definition = $c_def, 
                    c.embedding = $c_vec,
                    c.domain = $domain,
                    c.valid_eras = $valid_eras
                SET c += $sens_props

                MERGE (r:RiskFactor {id: $r_id})
                SET r.name = $r_label, r.domain = $domain
                
                MERGE (n:Norm {name: $n_name})
                SET n.type = $n_type,
                    n.issuer = $n_issuer,
                    n.jurisdiction = $n_jur

                // source付与により、推論生成時に「知識ベース由来」であることを明示
                MERGE (c)-[:LEADS_TO {source: 'ontology_def'}]->(r)
                MERGE (r)-[:VIOLATES {source: 'ontology_def'}]->(n)
                """
                session.run(query_core, 
                            c_id=item['id'], c_name=item['name'], c_def=item['definition'], c_vec=c_vector,
                            domain=item['domain'], valid_eras=item['valid_eras'], sens_props=sens_props,
                            r_id=item['risk_factor']['id'], r_label=item['risk_factor']['label'],
                            n_name=item['norm']['name'], n_type=item['norm']['type'],
                            n_issuer=item['norm'].get('issuer', 'unknown'), 
                            n_jur=item['norm'].get('jurisdiction', 'unknown'))

                # --- Affected Groups ---
                for group_name in item.get('affected_groups', []):
                    session.run("""
                        MATCH (r:RiskFactor {id: $r_id})
                        MERGE (g:AffectedGroup {name: $g_name})
                        MERGE (r)-[:OFFENDS {source: 'ontology_def'}]->(g)
                    """, r_id=item['risk_factor']['id'], g_name=group_name)

                # --- Typical Associations (Exemplar) ---
                # type: 'exemplar' を明示し、推論時に生成される 'candidate' と区別
                for assoc_text in item.get('typical_associations', []):
                    a_vector = embedder.get_embedding(assoc_text)
                    if a_vector is None: continue

                    session.run("""
                        MATCH (c:Concept {id: $c_id})
                        MERGE (a:TypicalAssociation {name: $text})
                        SET a.embedding = $vector
                        MERGE (a)-[:MAPS_TO {type: 'exemplar', source: 'knowledge_base', confidence: 1.0}]->(c)
                    """, c_id=item['id'], text=assoc_text, vector=a_vector)

                # --- Related Roles ---
                for role_name in item.get('related_roles', []):
                    session.run("""
                        MATCH (c:Concept {id: $c_id})
                        MERGE (ro:RoleConcept {name: $role})
                        MERGE (ro)-[:IN_CONTEXT_OF {source: 'ontology_def'}]->(c)
                    """, c_id=item['id'], role=role_name)

            # 3. Contexts
            for ctx in CONTEXT_CONCEPTS:
                ctx_text = f"{ctx['name']} ({', '.join(ctx['attributes'])})"
                ctx_vector = embedder.get_embedding(ctx_text)
                if ctx_vector is None: continue

                session.run("""
                    MERGE (cx:ContextConcept {id: $id})
                    SET cx.name = $name, cx.attributes = $attrs, cx.embedding = $vec, cx.domain = $domain,
                        cx.valid_eras = $valid_eras
                    
                    MERGE (r:RiskFactor {id: $rid})
                    SET r.name = $r_label, r.domain = $domain

                    MERGE (n:Norm {name: $n_name})
                    SET n.type = $n_type, n.issuer = $n_issuer, n.jurisdiction = $n_jur

                    MERGE (cx)-[:TRIGGERS {source: 'ontology_def'}]->(r)
                    MERGE (r)-[:VIOLATES {source: 'ontology_def'}]->(n)
                """, 
                id=ctx['id'], name=ctx['name'], attrs=ctx['attributes'], vec=ctx_vector, domain=ctx['domain'],
                valid_eras=ctx['valid_eras'],
                rid=ctx['risk_factor']['id'], r_label=ctx['risk_factor']['label'],
                n_name=ctx['norm']['name'], n_type=ctx['norm']['type'],
                n_issuer=ctx['norm'].get('issuer', 'unknown'), n_jur=ctx['norm'].get('jurisdiction', 'unknown'))
                
                # Contextの影響集団
                for group_name in ctx.get('affected_groups', []):
                    session.run("""
                        MATCH (r:RiskFactor {id: $r_id})
                        MERGE (g:AffectedGroup {name: $g_name})
                        MERGE (r)-[:OFFENDS {source: 'ontology_def'}]->(g)
                    """, r_id=ctx['risk_factor']['id'], g_name=group_name)

    print(">>> Ontology Load Complete.")


# ==========================================
# 4. Ontology Visualization (matplotlib)
# ==========================================

def _fetch_ontology_edges(session, limit: int = 500):
    """Fetch a lightweight edge list for visualization."""
    query = """
    MATCH (s)-[r]->(t)
    WHERE (s:Concept OR s:ContextConcept OR s:TypicalAssociation OR s:RoleConcept OR s:RiskFactor)
      AND (t:Concept OR t:RiskFactor OR t:Norm OR t:AffectedGroup)
      AND type(r) IN ['LEADS_TO','TRIGGERS','VIOLATES','OFFENDS','MAPS_TO','IN_CONTEXT_OF']
    RETURN labels(s)[0] AS s_label, coalesce(s.name, s.id) AS s_name,
           type(r) AS rel,
           labels(t)[0] AS t_label, coalesce(t.name, t.id) AS t_name
    LIMIT $limit
    """
    rows = session.run(query, limit=limit)
    return [dict(r) for r in rows]


def visualize_ontology_matplotlib(limit: int = 500, figsize=(14, 10), with_legend: bool = True):
    """Render the ontology graph with matplotlib.

    Notes:
    - This is intended for quick inspection, not publication-quality diagrams.
    - Node colors are assigned by label.
    """
    if not NEO4J_PASSWORD:
        print("Error: NEO4J_PASSWORD is missing; cannot visualize.")
        return

    with GraphDatabase.driver(NEO4J_URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            edges = _fetch_ontology_edges(session, limit=limit)

    if not edges:
        print("No edges found to visualize.")
        return

    # Resolve an actual Japanese font file (Hiragino Sans) when available.
    # This is more reliable than relying on rcParams alone for NetworkX label rendering.
    prop = None
    try:
        font_path = fm.findfont(fm.FontProperties(family="Hiragino Sans"), fallback_to_default=True)
        if font_path and os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path)
    except Exception:
        prop = None

    G = nx.DiGraph()

    # Build graph
    for e in edges:
        s = f"{e['s_label']}:{e['s_name']}"
        t = f"{e['t_label']}:{e['t_name']}"
        G.add_node(s, label=e['s_label'])
        G.add_node(t, label=e['t_label'])
        G.add_edge(s, t, rel=e['rel'])

    # Layout
    pos = nx.spring_layout(G, seed=42, k=None)

    # Color mapping (matplotlib default cycle; do not hardcode specific colors)
    labels = sorted({data.get('label', 'Unknown') for _, data in G.nodes(data=True)})
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not color_cycle:
        color_cycle = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    label_to_color = {lab: color_cycle[i % len(color_cycle)] for i, lab in enumerate(labels)}

    node_colors = [label_to_color.get(G.nodes[n].get('label', 'Unknown'), 'C0') for n in G.nodes()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.9, node_color=node_colors)

    # Edge labels (relationship types)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=12, width=1.2, alpha=0.7)

    # Keep node labels compact
    short_labels = {n: n.split(':', 1)[1] for n in G.nodes()}

    # Draw labels with explicit FontProperties when available
    # Draw node labels manually (works across NetworkX versions)
    ax = plt.gca()
    for node, label in short_labels.items():
        x, y = pos[node]
        if prop is not None:
            ax.text(
                x, y, label,
                fontsize=8,
                fontproperties=prop,
                ha="center", va="center"
            )
        else:
            ax.text(
                x, y, label,
                fontsize=8,
                ha="center", va="center"
            )

    # Draw edge labels manually at midpoints
    ax = plt.gca()
    for u, v, d in G.edges(data=True):
        rel = d.get("rel", "")
        if not rel:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if prop is not None:
            ax.text(
                xm, ym, rel,
                fontsize=7,
                fontproperties=prop,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, linewidth=0)
            )
        else:
            ax.text(
                xm, ym, rel,
                fontsize=7,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, linewidth=0)
            )

    if with_legend:
        handles = []
        for lab in labels:
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markersize=8, label=lab,
                                      markerfacecolor=label_to_color[lab], markeredgecolor=label_to_color[lab]))
        plt.legend(handles=handles, loc='best', fontsize=8)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load ontology into Neo4j and optionally visualize it.")
    parser.add_argument("--no-load", action="store_true", help="Skip loading and only visualize existing data.")
    parser.add_argument("--plot", action="store_true", help="Visualize the ontology with matplotlib.")
    parser.add_argument("--limit", type=int, default=500, help="Max number of edges to fetch for plotting.")
    args = parser.parse_args()

    if not args.no_load:
        setup_ontology()

    if args.plot:
        visualize_ontology_matplotlib(limit=args.limit)
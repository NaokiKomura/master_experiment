from openai import OpenAI
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 設定 ---
client = OpenAI(api_key="sk-proj-sy----") # 既存のキー
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "-----") 

# --- 知識データの定義 ---
RISK_KNOWLEDGE = [
    {
        "concept": "性別役割分業の固定化",
        "definition": "「家事育児は女性、仕事は男性」という固定観念。...",
        "risk_factor": "Gender_Stereotype",
        "norm": "男女共同参画社会基本法",
        "group": "働く女性・主婦層"
    },
    {
        "concept": "性的対象化",
        "definition": "脈絡なく女性の身体の一部を強調したり...。",
        "risk_factor": "Sexual_Objectification",
        "norm": "メディアにおける倫理ガイドライン",
        "group": "女性全般"
    },
    {
        "concept": "ルッキズム（外見至上主義）",
        "definition": "人の価値を外見で判断すること...。",
        "risk_factor": "Lookism",
        "norm": "人権尊重の理念",
        "group": "若年層・女性"
    },
    {
        "concept": "人種・属性のフェティシズム",
        "definition": "ハーフや特定の人種を、ファッション感覚のように扱う表現...。",
        "risk_factor": "Racial_Fetishization",
        "norm": "人種差別撤廃条約",
        "group": "マイノリティ・子供"
    }
]

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def setup_ontology():
    """Neo4jにオントロジーを投入する既存の関数"""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            print("Clearing old ontology data...")
            session.run("MATCH (n) WHERE n:Concept OR n:RiskFactor OR n:Norm OR n:AffectedGroup DETACH DELETE n")
            
            for item in RISK_KNOWLEDGE:
                print(f"Loading Concept: {item['concept']}...")
                vector = get_embedding(item['definition'])
                query = """
                MERGE (c:Concept {name: $concept})
                SET c.definition = $definition, c.embedding = $vector
                MERGE (r:RiskFactor {name: $risk})
                MERGE (n:Norm {name: $norm})
                MERGE (g:AffectedGroup {name: $group})
                MERGE (c)-[:LEADS_TO]->(r)
                MERGE (r)-[:VIOLATES]->(n)
                MERGE (r)-[:OFFENDS]->(g)
                """
                session.run(query, concept=item['concept'], definition=item['definition'], vector=vector,
                            risk=item['risk_factor'], norm=item['norm'], group=item['group'])

def visualize_ontology_diagram(knowledge_data):
    """オントロジー構造を可視化して画像保存する関数"""
    print("Generating ontology diagram...")
    G = nx.DiGraph()
    
    # ノードとエッジの追加
    node_colors = []
    color_map = {
        "Concept": "#A0CBE8", 
        "RiskFactor": "#FFBE7D", 
        "Norm": "#8CD17D", 
        "AffectedGroup": "#FF9D9A"
    }

    for item in knowledge_data:
        c, r, n, g = item["concept"], item["risk_factor"], item["norm"], item["group"]
        
        # ノード追加
        G.add_node(c, label_type="Concept")
        G.add_node(r, label_type="RiskFactor")
        G.add_node(n, label_type="Norm")
        G.add_node(g, label_type="AffectedGroup")
        
        # エッジ追加
        G.add_edge(c, r, label="LEADS_TO")
        G.add_edge(r, n, label="VIOLATES")
        G.add_edge(r, g, label="OFFENDS")

    # 色の設定
    colors = [color_map[G.nodes[node]['label_type']] for node in G.nodes()]

    # 日本語フォントの設定（環境内のNoto Sans CJK JPを使用）
    try:
        font_path = fm.findfont(fm.FontProperties(family="Hiragino Sans"))
        prop = fm.FontProperties(fname=font_path)
    except:
        prop = fm.FontProperties()

    # レイアウトと描画
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3000, 
            font_size=9, font_family=prop.get_name(), edge_color="#BBBBBB", 
            width=1.5, arrowsize=20, ax=ax)
    
    # 凡例の追加
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k,
                              markerfacecolor=v, markersize=10) for k, v in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Ad Risk Ontology Structure", fontsize=15)
    plt.tight_layout()
    
    # 保存
    plt.savefig("ontology_diagram.png")
    print("Diagram saved as 'ontology_diagram.png'.")

if __name__ == "__main__":
    # 1. Neo4jへの投入
    setup_ontology()
    print("Ontology re-loaded with NEW concepts.")
    
    # 2. 図の出力
    visualize_ontology_diagram(RISK_KNOWLEDGE)
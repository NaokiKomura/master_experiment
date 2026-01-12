import os
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

# 環境変数の読み込み
load_dotenv()

# 設定
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")
AUTH = (neo4j_user, neo4j_password)

URI = neo4j_uri

SIMILARITY_THRESHOLD = 0.72  # 調整後の値

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def map_associations_to_concepts():
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"Neo4j Connection Failed: {e}")
        return

    with driver.session() as session:
        # 1. まだConceptに紐付いていないAssociationを取得
        result = session.run("""
            MATCH (a:Association)
            WHERE NOT (a)-[:MAPS_TO]->(:Concept)
            RETURN a.name AS name, elementId(a) AS id
        """)
        
        associations = list(result)
        print(f"Found {len(associations)} unmapped associations.")
        
        for record in associations:
            assoc_name = record["name"]
            assoc_id = record["id"]
            
            print(f"Mapping '{assoc_name}'...")
            
            vector = get_embedding(assoc_name)
            
            search_query = """
            CALL db.index.vector.queryNodes('concept_index', 3, $vector)
            YIELD node AS concept, score
            WHERE score >= $threshold
            RETURN concept.name AS concept_name, score
            """
            
            matches = session.run(search_query, vector=vector, threshold=SIMILARITY_THRESHOLD)
            
            best_match = matches.single()
            if best_match:
                concept_name = best_match["concept_name"]
                score = best_match["score"]
                print(f"  -> Matched with: {concept_name} (Score: {score:.4f})")
                
                link_query = """
                MATCH (a:Association) WHERE elementId(a) = $assoc_id
                MATCH (c:Concept {name: $concept_name})
                MERGE (a)-[r:MAPS_TO]->(c)
                SET r.similarity = $score
                """
                session.run(link_query, assoc_id=assoc_id, concept_name=concept_name, score=score)
            else:
                print("  -> No match found in ontology.")

    driver.close()

if __name__ == "__main__":
    map_associations_to_concepts()
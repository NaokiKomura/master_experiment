import logging
from neo4j import GraphDatabase
try:
    from config import NEO4J_URI, NEO4J_AUTH
except ImportError:
    from .config import NEO4J_URI, NEO4J_AUTH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_to_neo4j(payload):
    if not NEO4J_AUTH[1]:
        logger.error("NEO4J_PASSWORD is not set.")
        return

    ad_id = payload.get('ad_id')
    if not ad_id: return

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            meta = payload.get('meta', {})
            logger.info(f"Loading Ad: {ad_id}")
            
            session.run("""
                MERGE (ad:Ad {id: $ad_id})
                SET ad.csv_id = $csv_id, ad.copy_text = $copy_text, ad.timestamp = datetime()
            """, ad_id=ad_id, csv_id=meta.get('csv_id'), copy_text=payload.get('input_text', ''))

            # (以下、Expressions, Associations, Roles などのロード処理は前回のコードと同様)
            # ...
    except Exception as e:
        logger.error(f"Error loading to Neo4j: {e}")
    finally:
        driver.close()

def clear_ad_data():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        with driver.session() as session:
            logger.info("Clearing Ad instance data...")
            session.run("MATCH (n) WHERE n:Ad OR n:Expression OR n:PlacementContext OR n:Evidence OR n:DepictedRole DETACH DELETE n")
            session.run("MATCH (a:Association) WHERE NOT (a)--(:Concept) DETACH DELETE a")
    finally:
        driver.close()
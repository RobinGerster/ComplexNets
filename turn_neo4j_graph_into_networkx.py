import pandas as pd
from neo4j import GraphDatabase
import networkx as nx
import dill
from neo4j.time import DateTime, Date

# Connect to Neo4j
HOST = 'neo4j://localhost'
USERNAME = 'neo4j'
PASSWORD = 'password'
uri = "bolt://localhost:7687" # Replace with your Neo4j Desktop bolt address (should be the same)
username = "neo4j" # Replace with your Neo4j Desktop username (should be the same)
password = "password" # Replace with your Neo4j Desktop password (should be the same)

# Setup necessary driver and gds for running cypher in our DBMS
driver = GraphDatabase.driver(uri, auth=(username, password))

query_nodes = """
MATCH (n) RETURN n SKIP $skip LIMIT $limit
"""

query_rels = """
MATCH (n)-[r]->(c) RETURN r SKIP $skip LIMIT $limit
"""

G = nx.MultiDiGraph()
chunk_size = 10000

# Process nodes in chunks
skip = 0
while True:
    print("skip is:", skip, '/ 790000', end='\r')
    results_nodes = driver.session().run(query_nodes, skip=skip, limit=chunk_size)
    nodes = results_nodes.graph()._nodes.values()

    if not nodes:
        break

    for node in nodes:
        props = {}
        for k, v in node._properties.items():
            if isinstance(v, DateTime) or isinstance(v, Date):
                props[k] = v.to_native()
            else:
                props[k] = v
        # print(props)
        G.add_node(node.element_id, labels=node._labels, properties=props)

    skip += chunk_size

print("Got Nodes")
print("getting relationships")
# Process relationships in chunks
skip = 0
chunk_size = 10000
while True:
    print("skip is:", skip, '/ 1770000', end='\r')
    results_rels = driver.session().run(query_rels, skip=skip, limit=chunk_size)
    rels = list(results_rels.graph()._relationships.values())

    if not rels:
        break

    for rel in rels:
        props = {}
        for k, v in rel._properties.items():
            if isinstance(v, DateTime) or isinstance(v, Date):
                props[k] = v.to_native()
            else:
                props[k] = v
        # if len(props.items()) > 1:
        #     print(props)
        G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, type=rel.type, properties=props)

    skip += chunk_size

print("Saving")
# Save the graph to a file using pickle
with open('graph.pkl', 'wb') as f:
    dill.dump(G, f)

# Iterate over each node and print its label
for node in G.nodes():
    labels = G.nodes[node]['labels']
    print("User" in labels)
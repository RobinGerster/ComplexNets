import pandas as pd
import os
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

# Connect to Neo4j
HOST = 'neo4j://localhost'
USERNAME = 'neo4j'
PASSWORD = 'password'
uri = "bolt://localhost:7687/neo4j" # Replace with your Neo4j Desktop bolt address (should be the same)
username = "neo4j" # Replace with your Neo4j Desktop username (should be the same)
password = "password" # Replace with your Neo4j Desktop password (should be the same)

# Setup necessary driver and gds for running cypher in our DBMS
driver = GraphDatabase.driver(uri, auth=(username, password))
gds = GraphDataScience(driver)

#Example script to return a table of node counts.
# total node counts
result = gds.run_cypher('''
    CALL apoc.meta.stats()
    YIELD labels
    UNWIND keys(labels) AS nodeLabel
    RETURN nodeLabel, labels[nodeLabel] AS nodeCount
''')
print(result)

query = """
CALL gds.betweenness.stream('default')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY name ASC
"""
result = gds.run_cypher(query)
print(result)
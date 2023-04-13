import networkx as nx
import dill
import heapq
import json
import math

print("loading graph")
with open('graph.pkl', 'rb') as f:
    nxG = dill.load(f)
print("graph loaded")

def find_fraudulent_nodes(g: nx.MultiDiGraph):
    fraudulent_nodes = []

    for node, attributes in g.nodes(data=True):
        if attributes["properties"].get('fraudMoneyTransfer') == 1:
            fraudulent_nodes.append(node)

    return fraudulent_nodes

def get_top_k_nodes(data_dict, metric, k=241):
    node_metric_tuples = [(node, data_dict[node][metric]) for node in data_dict if not math.isnan(data_dict[node][metric])]
    top_k_nodes = heapq.nlargest(k, node_metric_tuples, key=lambda x: x[1])
    return [node for node, _ in top_k_nodes]

def response_rate_overlap(data_dict, metric, fraudulent_nodes, k=241):
    top_k_nodes = get_top_k_nodes(data_dict, metric, k)
    overlap = set(top_k_nodes).intersection(set(fraudulent_nodes))
    return len(overlap)

fraudulent_nodes = find_fraudulent_nodes(nxG)
print("finished computing fraudulent nodes")
# Calculate overlap for each metric and graph
metrics = ["closeness", "degree", "betweenness", "eigenvector_centrality", "sis_infection_prob", "page_rank"]



with open('compute_metrics_results.json') as f:
    # Load the JSON object from the file as a dictionary
    data = json.load(f)
    print("finished loading JSON")
    for graph_name, data_dict in data.items():
        print(f"Graph: {graph_name}")
        for metric in metrics:
            overlap = response_rate_overlap(data_dict, metric, fraudulent_nodes)
            print(f"Overlap for {metric}: {overlap}")
        print()





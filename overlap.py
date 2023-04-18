from collections import defaultdict

import networkx as nx
import dill
import heapq
import json
import math
import scipy.stats
import numpy as np
import random

best_metrics = {}


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


def pick_random_nodes(g: nx.MultiDiGraph, num_nodes=241):
    # Get the non-fraudulent user nodes
    non_fraud_nodes = [node for node, attr in g.nodes(data=True) if attr['labels'] == {'User'} and attr['properties']['fraudMoneyTransfer'] == 0]

    # Pick random non-fraudulent nodes
    return random.sample(non_fraud_nodes, num_nodes)


def get_top_k_nodes(data_dict, metric, k=241):
    node_metric_tuples = [(node, data_dict[node][metric]) for node in data_dict if not math.isnan(data_dict[node][metric])]
    top_k_nodes = heapq.nlargest(k, node_metric_tuples, key=lambda x: x[1])
    return [node for node, _ in top_k_nodes]

def response_rate_overlap(data_dict, metric, fraudulent_nodes, k=241):
    top_k_nodes = get_top_k_nodes(data_dict, metric, k)
    overlap = set(top_k_nodes).intersection(set(fraudulent_nodes))
    return len(overlap)

def get_fraud_labels(graph, nodes):
    fraud_labels = []
    for node in nodes:
        fraud_labels.append(graph.nodes[node]["properties"]["fraudMoneyTransfer"])
    return fraud_labels

def compute_correlations(data_dict, nodes, fraud_labels):
    correlations = {}
    for metric in data_dict[nodes[0]].keys():
        metric_values = [data_dict[node][metric] for node in nodes]

        # Replace NaN values with 0
        metric_values_no_nan = np.nan_to_num(metric_values)
        # metric_values_normalized = normalize_data(metric_values_no_nan)
        correlation, _ = scipy.stats.pearsonr(fraud_labels, metric_values_no_nan)
        correlations[metric] = correlation
    return correlations

def normalize_data(data):
    min_value = np.min(data)
    max_value = np.max(data)
    return (data - min_value) / (max_value - min_value)

fraudulent_nodes = find_fraudulent_nodes(nxG)
print("finished computing fraudulent nodes")
# Calculate overlap for each metric and graph
metrics = ["closeness", "degree", "betweenness", "eigenvector_centrality", "sis_infection_prob", "page_rank"]

all_metrics = []

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

        # Calculate correlations for each metric
        nodes = list(data_dict.keys())
        user_nodes = [node for node in nodes if nxG.nodes[node]["labels"] == {"User"}]
        fraud_labels = get_fraud_labels(nxG, user_nodes)
        non_fraud_users = [node for node in nodes if
                           nxG.nodes[node]["labels"] == {"User"} and nxG.nodes[node]['properties'][
                               'fraudMoneyTransfer'] == 0]
        fraud_users = [node for node in nodes if
                       nxG.nodes[node]["labels"] == {"User"} and nxG.nodes[node]['properties'][
                           'fraudMoneyTransfer'] == 1]

        list0 = [0] * 241
        list1 = [1] * 241
        correlations = compute_correlations(data_dict, user_nodes, fraud_labels)
        print("\nCorrelations:")
        for metric, correlation in correlations.items():
            print(f"{metric}: {correlation}")

        correlations_list = defaultdict(list)
        for epoch in range(0, 1000):
            non_fraud_users_sample = random.sample(non_fraud_users, 241)
            correlations_new = compute_correlations(data_dict, fraud_users + non_fraud_users_sample, list1 + list0)

            for metric, correlation in correlations_new.items():
                correlations_list[metric].append(correlation)

        print("\nCorrelations (averaged over epochs and standard deviation):")
        for metric, correlation_values in correlations_list.items():
            avg_correlation = sum(correlation_values) / len(correlation_values)
            std_deviation = np.std(correlation_values)
            print(f"{metric}: {avg_correlation} (std: {std_deviation})")
            all_metrics.append((graph_name, metric, avg_correlation, std_deviation))

        print()

sorted_all_metrics = sorted(all_metrics, key=lambda x: x[2], reverse=True)
print("All metrics for each graph (sorted by highest correlation value):")
for graph, metric, avg_correlation, std_deviation in sorted_all_metrics:
    print(f"{graph}: {metric} (avg: {avg_correlation}, std: {std_deviation})")
# with open('compute_metrics_results_final_graph.json') as f:
#     # Load the JSON object from the file as a dictionary
#     data_dict = json.load(f)
#     print("finished loading JSON")
#     for metric in metrics:
#         overlap = response_rate_overlap(data_dict, metric, fraudulent_nodes)
#         print(f"Overlap for {metric}: {overlap}")
#         print()
#         # Calculate correlations for each metric
#         nodes = list(data_dict.keys())
#         user_nodes = [node for node in nodes if nxG.nodes[node]["labels"] == {"User"}]
#         fraud_labels = get_fraud_labels(nxG, user_nodes)
#         correlations = compute_correlations(data_dict, user_nodes, fraud_labels)
#         print("\nCorrelations:")
#         for metric, correlation in correlations.items():
#             print(f"{metric}: {correlation}")



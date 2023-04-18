import networkx as nx
import pandas as pd
import dill
from tqdm import tqdm
import graph_tool.all as gt
from tqdm import tqdm
import copy
from typing import Generic, TypeVar, cast
from typing import List, Tuple
from collections import defaultdict
import random, json
import networkx as nx
import datetime, os
from multiprocess import Pool
import polars as pl
from infection_sim_methods import infection_sim_fraud_metrics, infection_sim_effective_spreading_power
from infection_sim_methods import infection_simulations_alt
from infection_sim_methods import infection_simulations_agg
random.seed(1337)

# defining graph_tool methods

def convert_networkx_to_graphtool(nx_graph):
    # print("part_1")
    gt_graph = gt.Graph(directed=nx_graph.is_directed())
    # print("part_2")
    gt_graph.add_vertex(len(nx_graph))

    # Create a mapping between original labels and new integer labels
    # print("part_3")
    label_to_index = {label: index for index, label in enumerate(nx_graph.nodes)}
    index_to_label = {index: label for index, label in enumerate(nx_graph.nodes)}

    print("adding edges")
    for edge in tqdm(nx_graph.edges):
        gt_graph.add_edge(label_to_index[edge[0]], label_to_index[edge[1]])

    return gt_graph, label_to_index, index_to_label

def compute_betweenness(g):
    vb, eb = gt.betweenness(g)
    betweenness = {}
    for v in g.vertices():
        betweenness[int(v)] = vb[v]
    return betweenness

def compute_eigenvector_centrality(g):
    _, ev = gt.eigenvector(g)
    eigenvector_centrality = {}
    for v in g.vertices():
        eigenvector_centrality[int(v)] = ev[v]
    return eigenvector_centrality

def compute_closeness(g):
    closeness = gt.closeness(g)
    closeness_dict = {}
    for v in g.vertices():
        closeness_dict[int(v)] = closeness[v]
    return closeness_dict

def compute_degree(g):
    degree_dict = {}
    for v in g.vertices():
        degree_dict[int(v)] = v.out_degree() + v.in_degree()
    return degree_dict

def compute_SIS_infection_prob(g, beta, mu):
    N = g.num_vertices()
    infection_prob = [0] * N
    for i in range(N):
        infection_prob[i] = 1 - (1 - beta) ** (mu * g.vertex(i).out_degree())
    return infection_prob

def compute_page_rank(g, damping=0.85, epsilon=1e-6):
    pr = gt.pagerank(g, damping=damping, epsilon=epsilon)
    page_rank = {}
    for v in g.vertices():
        page_rank[int(v)] = pr[v]
    return page_rank

def compute_clustering_coeff(g):
    lc = gt.local_clustering(g)
    clustering_coeff = {}
    for v in g.vertices():
        clustering_coeff[int(v)] = lc[v]
    return clustering_coeff

def get_edge_time(edge_data):
    if 'transactionDateTime' in edge_data:
        return edge_data['transactionDateTime'].timestamp()
    elif 'cardDate' in edge_data:
        return datetime.datetime.combine(edge_data['cardDate'], datetime.datetime.min.time()).timestamp()
    elif 'deviceDate' in edge_data:
        return datetime.datetime.combine(edge_data['deviceDate'], datetime.datetime.min.time()).timestamp()
    elif 'ipDate' in edge_data:
        return datetime.datetime.combine(edge_data['ipDate'], datetime.datetime.min.time()).timestamp()
    else:
        return None

def _si_infection_step(args):
    graph, node, t, label_prob = args
    infected_nodes = set([node])
    new_infected_nodes = set([node])

    for _ in range(t):
        next_new_infected_nodes = set()
        for infected_node in new_infected_nodes:
            neighbors = list(graph.neighbors(infected_node))
            for neighbor in neighbors:
                if neighbor not in infected_nodes and random.random() <= label_prob[list(graph.nodes[infected_node]['labels'])[0]][list(graph.nodes[neighbor]['labels'])[0]]:
                    next_new_infected_nodes.add(neighbor)
                    infected_nodes.add(neighbor)
        new_infected_nodes = next_new_infected_nodes
    return len(infected_nodes)

def si_infection_simulation(graph, n=100, t=100, label_prob=None):
    if label_prob is None:
        label_prob = {
            # label1: {label2: 1.0 for label2 in graph.graph['Label']} for label1 in graph.graph['Label']
            "User": {
                "User": 1,
                "Device": 1,
                "Card": 1,
                "IP": 1,
            },
            "Device": {
                "User": 1,
                "Device": 1,
                "Card": 1,
                "IP": 1,
            },
            "Card": {
                "User": 1,
                "Device": 1,
                "Card": 1,
                "IP": 1,
            },
            "IP": {
                "User": 1,
                "Device": 1,
                "Card": 1,
                "IP": 1,
            },
        }

    nodes = list(graph.nodes)
    args = [(graph, node, t, label_prob) for node in nodes]

    with Pool() as pool:
        infection_counts = list(tqdm(pool.imap_unordered(_si_infection_step, args), total=len(args), desc="Infection simulation"))

    return infection_counts

# Example usage:
# G = ... # Load the graph
# n = 5
# t = 10
# label_prob = { ... } # Optional custom label probabilities
# infection_counts = si_infection_simulation(G, n, t, label_prob)
# print(infection_counts)

# Example usage:
# label_probabilities = {("A", "A"): 0.9, ("A", "B"): 0.5, ("B", "A"): 0.5, ("B", "B"): 0.9}
# infected_counts = SI_infection_simulation_timestamps(g, 10, 100, label_probabilities)


# computing metrics
def compute_metrics(G: gt.Graph, index_to_label: dict):

    g = G

    beta = 0.5
    mu = 0.3

    print("computing closeness")
    closeness = compute_closeness(g)

    print("computing degree")
    degree = compute_degree(g)

    print("computing betweenness")
    betweenness = compute_betweenness(g)

    print("computing eigenvector_centrality")
    eigenvector_centrality = compute_eigenvector_centrality(g)

    print("computing SIS_infection_prob")
    SIS_infection_prob = compute_SIS_infection_prob(g, beta, mu)

    print("computing pagerank")
    page_rank = compute_page_rank(g)

    print("computing clustering coefficient")
    clustering_coefficient = compute_clustering_coeff(g)

    print("computation done")

    # print(f"Node : Betweenness : Eigenvector Centrality : Closeness : Degree : SIS Infection Probability : Page Rank")
    # for node in list(g.vertices())[:10]:
    #     print(f"{int(node)} : {betweenness[int(node)]} : {eigenvector_centrality[int(node)]} : {closeness[int(node)]} : {degree[int(node)]} : {SIS_infection_prob[int(node)]} : {page_rank[int(node)]}")
    #     # print(f"Node {int(node)}: Betweenness: {betweenness[int(node)]}, Eigenvector Centrality: {eigenvector_centrality[int(node)]}, Closeness: {closeness[int(node)]}, Degree: {degree[int(node)]}, SIS Infection Probability: {SIS_infection_prob[int(node)]}")
    
    data_dict = {}
    for node in g.vertices():
        data_dict[index_to_label[int(node)]] = dict(
            closeness=closeness[int(node)],
            degree=degree[int(node)],
            betweenness=betweenness[int(node)],
            eigenvector_centrality=eigenvector_centrality[int(node)],
            sis_infection_prob=SIS_infection_prob[int(node)],
            page_rank=page_rank[int(node)],
            clustering_coefficient=clustering_coefficient[int(node)],
        )
    return data_dict



def infection_sim(df, node):
    df_n1, df_n2, df_t = df
    # this will return a list of nodes with times it took for them to
    # be infected when parameter "node" was the first infected node

    # We set a random seed
    start_infected_node = node

    infected = set()
    infection_nodes = []
    infection_times = []
    infected.add(start_infected_node)

    index = 0

    # row = df[index, :]
    index += 1
    infections = 0
    # INFECTION_THRESHOLD = 10000
    for t in sorted(set(df_t)):
        while(df_t[index] == t and index < len(df_t)):
            n1 = df_n1[index]
            n2 = df_n2[index]
            c1 = n1 in infected
            c2 = n2 in infected
            if c1 ^ c2: # xor
                infections += 1
                infection_times.append(t)
                if not c1:
                    infection_nodes.append(n1)
                    infected.add(n1)
                if not c2:
                    infection_nodes.append(n2)
                    infected.add(n2)
            index += 1
            if index >= len(df_t):
                break
        #     if infections > INFECTION_THRESHOLD or index >= len(df_t):
        #         break
        # if infections > INFECTION_THRESHOLD:
        #     break
    
    return list(zip(infection_times, infection_nodes))

def infection_simulations(nx_graph: nx.Graph, fraudulent_nodes: set, control_nodes: set, test=False):
    # data = {
    #     'n1': [],
    #     'n2': [],
    #     'time': []
    # }
    # print("loading edges into dict...")
    li_edges = []
    for v, u, keys, properties in tqdm(nx_graph.edges(keys=True, data='properties')):
        t = get_edge_time(properties)
        if t:
            li_edges.append((t, u, v))
    li_edges.sort()
    df_n1 = [e[1] for e in li_edges]
    df_n2 = [e[2] for e in li_edges]
    df_t = [e[0] for e in li_edges]
    df = (df_n1, df_n2, df_t)
    print("running simulations")
    infection_info = dict()
    
    fraud_group = list(fraudulent_nodes)
    control_group = random.sample(list(control_nodes.difference(fraudulent_nodes)), k=len(fraud_group))
    
    if test:
        from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(infection_sim)
        # for n in tqdm(list(nx_graph.nodes())[:10]):
        #     infection_info[n] = lp_wrapper(df, n)
        # lp.print_stats()
    else:
        # only compute infection sim for fraud and a same-size sample of non-fraud nodes
        for n in tqdm(fraud_group):
            infection_info[n] = infection_sim(df, n)
        for n in tqdm(control_group):
            infection_info[n] = infection_sim(df, n)

    final_info = {}
    for node_name, infectlist in infection_info.items():
        infection_duration, average_time, num_fraud_nodes_infected,\
            fraud_infection_speed = infection_sim_fraud_metrics(infectlist, fraud_group)
        effective_spreading_power = infection_sim_effective_spreading_power(infectlist, nx_graph)
        final_info[node_name] = {
            'SI_infection_duration': infection_duration,
            'SI_average_time': average_time,
            'SI_num_fraud_nodes_infected': num_fraud_nodes_infected,
            'SI_fraud_infection_speed': fraud_infection_speed,
            'SI_effective_spreading_power': effective_spreading_power,
        }

    return final_info


def save_graphs(nx_graph, gt_graph, graph_name, label_to_index, index_to_label):
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    nx_file = os.path.join('graphs', f'{graph_name}_networkx.dill')
    gt_file = os.path.join('graphs', f'{graph_name}_graphtool.dill')

    if not os.path.exists(nx_file):
        with open(nx_file, 'wb') as f:
            dill.dump(nx_graph, f)

    if not os.path.exists(gt_file):
        with open(gt_file, 'wb') as f:
            dill.dump((gt_graph, label_to_index, index_to_label), f)

def load_graphs(graph_name):
    nx_file = os.path.join('graphs', f'{graph_name}_networkx.dill')
    gt_file = os.path.join('graphs', f'{graph_name}_graphtool.dill')

    if os.path.exists(nx_file) and os.path.exists(gt_file):
        with open(nx_file, 'rb') as f:
            nx_graph = dill.load(f)

        with open(gt_file, 'rb') as f:
            gt_graph, label_to_index, index_to_label = dill.load(f)

        return nx_graph, gt_graph, label_to_index, index_to_label
    else:
        return None, None, None, None
    


if __name__ == "__main__":
    # the experiment then goes like this:
    # 1. instantiate the graphs you're interested in
    # 2. run analysis on each, only keep the data_dict
    # 3. put the dicts together in a dict with simple names

    graphs_we_want = {
        "only_users": dict(
            node_labels_to_keep=['User'],
            edges_to_keep=[],
        ),
        "users_and_cards": dict(
            node_labels_to_keep=['User', 'Card'],
            edges_to_keep=[],
        ),
        "users_and_devices": dict(
            node_labels_to_keep=['User', 'Device'],
            edges_to_keep=[],
        ),
        "users_and_ips": dict(
            node_labels_to_keep=['User', 'IP'],
            edges_to_keep=[],
        ),
        "users_and_cards_without_user_to_user_edges": dict(
            node_labels_to_keep=['User', 'Card'],
            edges_to_keep=[('User', 'Card')],
        ),
    }

    results = {}
    for graph_name, graph_info in graphs_we_want.items():
        print("\n\ncomputing graph ", graph_name)
        # 1. instantiate the graphs you're interested in
        nx_graph, gt_graph, label_to_index, index_to_label = load_graphs(graph_name)
        if nx_graph is None or gt_graph is None:
            nx_graph = filter_graph(nxG, graph_info['node_labels_to_keep'], graph_info['edges_to_keep'])
            gt_graph, label_to_index, index_to_label = convert_networkx_to_graphtool(nx_graph)
            save_graphs(nx_graph, gt_graph, graph_name, label_to_index, index_to_label)

        # 2. run analysis on each, only keep the data_dict
        result = compute_metrics(gt_graph, index_to_label)

        # 2.1 add the "fraudulent" label to the result dict
        attrs = nx.get_node_attributes(nx_graph, 'properties')
        fraud_nodes = set()
        control_nodes = set()
        for node_name in result.keys():
            if "fraudMoneyTransfer" in attrs[node_name]:
                if bool(attrs[node_name]['fraudMoneyTransfer']):
                    fraud_nodes.add(node_name)
                else:
                    control_nodes.add(node_name)
                result[node_name]["isFraudulent"] = bool(attrs[node_name]['fraudMoneyTransfer'])
                result[node_name]["isUser"] = True
            else:
                result[node_name]["isUser"] = False

        # 2.2 use the fraudulent nodes to run infection simulation experiments
        infections = infection_simulations(nx_graph, fraud_nodes, control_nodes, test=False)
        # returns this:
        # {
        # 'SI_infection_duration': infection_duration,
        # 'SI_average_time': average_time,
        # 'SI_num_fraud_nodes_infected': num_fraud_nodes_infected,
        # 'SI_fraud_infection_speed': fraud_infection_speed,
        # 'SI_effective_spreading_power': effective_spreading_power,
        # }
        for nodename, infection_stats in infections.items():
            result[nodename].update(infection_stats)
        
        # 2.3 run the special infection simulation experiments
        infections_alt_userheavy, infections_alt_otherheavy = infection_simulations_alt(nx_graph, fraud_nodes, control_nodes)
        for nodename, infection_stats in infections_alt_userheavy.items():
            result[nodename].update(infection_stats)
        for nodename, infection_stats in infections_alt_otherheavy.items():
            result[nodename].update(infection_stats)


        # 2.4 run the aggregated ones
        infections_alt_userheavy, infections_alt_otherheavy = infection_simulations_agg(nx_graph, fraud_nodes, control_nodes)
        for nodename, infection_stats in infections_alt_userheavy.items():
            result[nodename].update(infection_stats)
        for nodename, infection_stats in infections_alt_otherheavy.items():
            result[nodename].update(infection_stats)
        
        # 3. put the dicts together in a dict with simple names
        results[graph_name] = result
        # results = si_infection_simulation(nx_graph)
        # print(results)
    # nx_graph = filter_graph(nxG, graphs_we_want["users_and_cards"]['node_labels_to_keep'], graphs_we_want["users_and_cards"]['edges_to_keep'])
    # gt_graph, label_to_index, index_to_label = convert_networkx_to_graphtool(nx_graph)
    with open("compute_metrics_results.json", 'w') as f:
        json.dump(results, f, indent=4)
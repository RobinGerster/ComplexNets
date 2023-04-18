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
random.seed(1337)
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

SOFTENING = 0.2 # for 0.2, in user-heavy mode user-user infections happen with 0.8 probability

def user_user_check(n1, n2, attribute_thing, mode='user-heavy'):
    uu_check = list(attribute_thing[n1])[0] == "User" and list(attribute_thing[n2])[0] == "User"
    if mode == 'user-heavy':
        if uu_check:
            return 1 - SOFTENING
        else:
            return SOFTENING
    else:
        if uu_check:
            return SOFTENING
        else:
            return 1 - SOFTENING

# def infection_sim_alt(df, node, graph, mode='user-heavy'):
#     df_n1, df_n2, df_t = df
#     # this will return a list of nodes with times it took for them to
#     # be infected when parameter "node" was the first infected node

#     # We set a random seed
#     start_infected_node = node

#     # infect differently based on user node status
#     label_attr = nx.get_node_attributes(graph, 'Label')

#     infected = set()
#     infection_counts = []
#     infection_times = []
#     infected.add(start_infected_node)

#     index = 0

#     # row = df[index, :]
#     index += 1
#     infections = 0
#     INFECTION_THRESHOLD = 10000
#     for t in sorted(set(df_t)):
#         infection = False
#         while(df_t[index] == t and index < len(df_t)):
#             n1 = df_n1[index]
#             n2 = df_n2[index]
#             c1 = n1 in infected
#             c2 = n2 in infected
#             if c1 ^ c2: # xor
#                 test_coeff = user_user_check(n1, n2, label_attr, mode=mode)
#                 if test_coeff > random.random():
#                     if not infection:
#                         infection = True
#                         infection_counts.append(1)
#                         infection_times.append(df_t[index])
#                     else:
#                         infection_counts[-1] += 1
#                     infections += 1
#                     if not c1:
#                         infected.add(n1)
#                     if not c2:
#                         infected.add(n2)
#             index += 1
#             if infections > INFECTION_THRESHOLD or index >= len(df_t):
#                 break
#         if infections > INFECTION_THRESHOLD:
#             break
    
#     return list(zip(infection_times, infection_counts))

def infection_sim_alt(df, node, graph, mode='user-heavy'):
    df_n1, df_n2, df_t = df
    # this will return a list of nodes with times it took for them to
    # be infected when parameter "node" was the first infected node

    # We set a random seed
    start_infected_node = node


    # infect differently based on user node status
    label_attr = nx.get_node_attributes(graph, 'labels')

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
                test_coeff = user_user_check(n1, n2, label_attr, mode=mode)
                if test_coeff > random.random():
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
    
def infection_simulations_alt(nx_graph: nx.Graph, fraudulent_nodes: set, control_nodes: set, test=False):
    # this would be called Heterogeneous SI model
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
    infection_info_userheavy = dict()
    infection_info_otherheavy = dict()
    
    fraud_group = list(fraudulent_nodes)
    control_group = random.sample(list(control_nodes.difference(fraudulent_nodes)), k=len(fraud_group))
    
    if test:
        from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(infection_sim_alt)
        # for n in tqdm(list(nx_graph.nodes())[:10]):
        #     infection_info_userheavy[n] = lp_wrapper(df, n, nx_graph)
        # lp.print_stats()
    else:
        # df, node, graph, mode='user-heavy'
        # only compute infection sim for fraud and a same-size sample of non-fraud nodes
        for n in tqdm(fraud_group):
            infection_info_userheavy[n] = infection_sim_alt(df, n, nx_graph, mode='user-heavy')
        for n in tqdm(control_group):
            infection_info_userheavy[n] = infection_sim_alt(df, n, nx_graph, mode='user-heavy')
        for n in tqdm(fraud_group):
            infection_info_otherheavy[n] = infection_sim_alt(df, n, nx_graph)
        for n in tqdm(control_group):
            infection_info_otherheavy[n] = infection_sim_alt(df, n, nx_graph)
    
    # this should, for each infection_info, user_heavy and other_heavy, compute the statistics
    res = []
    for infection_info, title in zip([infection_info_userheavy, infection_info_otherheavy], ["userheavy_", "otherheavy_"]):
        final_info = {}
        for node_name, infectlist in infection_info.items():
            infection_duration, average_time, num_fraud_nodes_infected,\
                fraud_infection_speed = infection_sim_fraud_metrics(infectlist, fraud_group)
            effective_spreading_power = infection_sim_effective_spreading_power(infectlist, nx_graph)
            final_info[node_name] = {
                f'HSI_{title}infection_duration': infection_duration,
                f'HSI_{title}average_time': average_time,
                f'HSI_{title}num_fraud_nodes_infected': num_fraud_nodes_infected,
                f'HSI_{title}fraud_infection_speed': fraud_infection_speed,
                f'HSI_{title}effective_spreading_power': effective_spreading_power,
            }
        res.append(final_info)
    
    infection_info_userheavy, infection_info_otherheavy = res[0], res[1]

    return infection_info_userheavy, infection_info_otherheavy


# CUSTOM METRICS FOR INFECTION SIMULATION OUTPUTS

def infection_sim_fraud_metrics(infection_info: list, fraud_group: set):
    infection_info_filtered = [elem for elem in infection_info if elem[1] in fraud_group]
    if len(infection_info_filtered) == 1:
        return 0.0, 0.0, 1, 0.0
    if len(infection_info_filtered) > 0:
        first_time = infection_info_filtered[0][0] # takes the first time
        last_time = infection_info_filtered[-1][0]
        infection_duration = last_time - first_time
        average_time = (sum([e[0] for e in infection_info]) / len(infection_info)) - first_time
        num_fraud_nodes_infected = len(infection_info_filtered)
        fraud_infection_speed = num_fraud_nodes_infected / infection_duration
        return infection_duration, average_time, num_fraud_nodes_infected, fraud_infection_speed
    return 0.0, 0.0, 0, 0.0

def infection_sim_effective_spreading_power(infection_info: list, nx_graph):
    # filter the graph so it only has user nodes
    label_attr = nx.get_node_attributes(nx_graph, 'labels')
    infection_info_updated = []
    for time, node in infection_info:
        if list(label_attr[node])[0] == "User":
            infection_info_updated.append((time, node))
    
    ESP = len(infection_info_updated) # Effective Spreading Power
    return ESP

def infection_sim_agg(graph, node, mode='user-heavy'):
    label_attr = nx.get_node_attributes(graph, 'labels')

    infected = set()
    infection_nodes = []
    infection_times = []
    infected.add(node)

    t = 0
    new_infections = True

    while new_infections:
        new_infections = False
        t += 1
        newly_infected = set()

        for infected_node in infected:
            for neighbor in graph.neighbors(infected_node):
                if neighbor not in infected:
                    test_coeff = user_user_check(infected_node, neighbor, label_attr, mode=mode)
                    if test_coeff > random.random():
                        newly_infected.add(neighbor)
                        infection_nodes.append(neighbor)
                        infection_times.append(t)
                        new_infections = True

        infected.update(newly_infected)

    return list(zip(infection_times, infection_nodes))


def infection_simulations_agg(nx_graph: nx.Graph, fraudulent_nodes: set, control_nodes: set):
    infection_info_userheavy = dict()
    infection_info_otherheavy = dict()

    fraud_group = list(fraudulent_nodes)
    control_group = random.sample(list(control_nodes.difference(fraudulent_nodes)), k=len(fraud_group))

    for n in fraud_group:
        infection_info_userheavy[n] = infection_sim_agg(nx_graph, n, mode='user-heavy')
    for n in control_group:
        infection_info_userheavy[n] = infection_sim_agg(nx_graph, n, mode='user-heavy')
    for n in fraud_group:
        infection_info_otherheavy[n] = infection_sim_agg(nx_graph, n)
    for n in control_group:
        infection_info_otherheavy[n] = infection_sim_agg(nx_graph, n)

    res = []
    for infection_info, title in zip([infection_info_userheavy, infection_info_otherheavy], ["userheavy_", "otherheavy_"]):
        final_info = {}
        for node_name, infectlist in infection_info.items():
            infection_duration, average_time, num_fraud_nodes_infected,\
                fraud_infection_speed = infection_sim_fraud_metrics(infectlist, fraud_group)
            effective_spreading_power = infection_sim_effective_spreading_power(infectlist, nx_graph)
            final_info[node_name] = {
                f'HSI_agg_{title}infection_duration': infection_duration,
                f'HSI_agg_{title}average_time': average_time,
                f'HSI_agg_{title}num_fraud_nodes_infected': num_fraud_nodes_infected,
                f'HSI_agg_{title}fraud_infection_speed': fraud_infection_speed,
                f'HSI_agg_{title}effective_spreading_power': effective_spreading_power,
            }
        res.append(final_info)

    infection_info_userheavy, infection_info_otherheavy = res[0], res[1]

    return infection_info_userheavy, infection_info_otherheavy
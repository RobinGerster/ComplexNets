import networkx as nx
import pandas as pd
import dill
import datetime

def main():

    G = None
    with open('graph.pkl', 'rb') as f:
        G = dill.load(f)
    
    print(G)

    # THE FOLLOWING IS TO SHOW THE STRUCTURE OF THE GRAPH
    # IT SHOULD RETURN THE FOLLOWING:
    '''
    MultiDiGraph with 789856 nodes and 1776743 edges

    Node Attributes:
        labels: <class 'frozenset'>
        properties: <class 'dict'>

    Edge Attributes:
        type: <class 'str'>
        properties: <class 'dict'>
        
    Node Labels:
        User
        Device
        Card
        IP

    Edge Types:
        HAS_IP
        USED
        P2P
        HAS_CC
        REFERRED

    Node Property Keys:                                               (these are examples)
        level:                          Type <class 'str'>,           Values {'LEVEL_TYPE_85', 'LEVEL_TYPE_104'}
        cardType:                       Type <class 'str'>,           Values {'CARD_TYPE_19', 'CARD_TYPE_13'}
        guid:                           Type <class 'str'>,           Values {'ffa2872c-07ff-4aa0-8f76-b7e320194d48', 'e2e11709-7dec-4623-be34-d0a38213f2e5'}
        os:                             Type <class 'str'>,           Values {'OS_TYPE_3', 'OS_TYPE_2'}
        device:                         Type <class 'str'>,           Values {'DEVICE_MODEL_3812', 'DEVICE_MODEL_1120'}
        fraudMoneyTransfer:             Type <class 'int'>,           Values {0, 1} <- indicates a fraudulent node
        moneyTransferErrorCancelAmount: Type <class 'float'>,         Values {877.0, 718.0}
        firstChargebackMtDate:          Type <class 'datetime.date'>, Values {datetime.date(2018, 11, 4), datetime.date(1900, 1, 1)}

    Edge Property Keys:
        totalAmount:         Type <class 'float'>,             Values {600.0, 5.0}
        transactionDateTime: Type <class 'datetime.datetime'>, Values {datetime.datetime(2017, 4, 13, 9, 47, 6, tzinfo=<UTC>), datetime.datetime(2017, 4, 11, 14, 54, 23, tzinfo=<UTC>)}
        cardDate:            Type <class 'datetime.date'>,     Values {datetime.date(2017, 2, 16), datetime.date(2016, 11, 20)}
        deviceDate:          Type <class 'datetime.date'>,     Values {datetime.date(2017, 10, 7), datetime.date(2018, 9, 12)}
        ipDate:              Type <class 'datetime.date'>,     Values {datetime.date(2017, 10, 4), datetime.date(2017, 2, 18)}
    '''

    def filter(val):
        if isinstance(val, str):
            if "TYPE" in val or len(val.split("-")) == 5 or len(val) == len("8ecdfcbed0e061b4cb4cab475f17dd44"):
                return True
        if isinstance(val, datetime.date) or isinstance(val, datetime.datetime):
            return True
        if isinstance(val, float):
            return True
        return False

    def generalize(val):
        if isinstance(val, str):
            if "_TYPE_" in val or "_MODEL_" in val:
                return "_".join(val.split("_")[:-1] + ["#"])
        return val

    # Collect unique node attributes and their types
    node_attributes = {}
    node_property_keys = {}
    node_attributes_values = set()
    for node_id, node_data in G.nodes(data=True):
        for attr, value in node_data.items():
            if attr not in node_attributes:
                node_attributes[attr] = type(value)
            if attr == 'labels':
                for val in value:
                    node_attributes_values.add(val)
            if attr == 'properties':
                for key, val in value.items():
                    val = generalize(val)
                    if key not in node_property_keys:
                        node_property_keys[key] = {'type': type(val), 'values': set()}
                    if not filter(val) or len(node_property_keys[key]['values']) < 2:
                        node_property_keys[key]['values'].add(val)

    # Collect unique edge attributes and their types
    edge_attributes = {}
    edge_property_keys = {}
    edge_types = set()
    for _, _, edge_data in G.edges(data=True):
        for attr, value in edge_data.items():
            if attr not in edge_attributes:
                edge_attributes[attr] = type(value)
            if attr == 'properties':
                for key, val in value.items():
                    val = generalize(val)
                    if key not in edge_property_keys:
                        edge_property_keys[key] = {'type': type(val), 'values': set()}
                    if not filter(val) or len(edge_property_keys[key]['values']) < 2:
                        edge_property_keys[key]['values'].add(val)
            elif attr == 'type':
                edge_types.add(value)

    # Print node attributes and their types
    print("Node Attributes:")
    for attr, attr_type in node_attributes.items():
        print(f"  {attr}: {attr_type}")

    # Print edge attributes and their types
    print("\nEdge Attributes:")
    for attr, attr_type in edge_attributes.items():
        print(f"  {attr}: {attr_type}")

    # Print node labels
    print("\n Node Labels:")
    for node_label in list(node_attributes_values):
        print(f"  {node_label}")

    # Print edge types
    print("\nEdge Types:")
    for edge_type in edge_types:
        print(f"  {edge_type}")

    # Print node property keys, their types, and possible values
    print("\nNode Property Keys:")
    for key, key_data in node_property_keys.items():
        print(f"  {key}: Type {key_data['type']}, Values {key_data['values']}")

    # Print edge property keys, their types, and possible values
    print("\nEdge Property Keys:")
    for key, key_data in edge_property_keys.items():
        print(f"  {key}: Type {key_data['type']}, Values {key_data['values']}")

if __name__ == '__main__':
    main()
# chris_rasmus_graph_features.py
#
# This file contains graph features

import pandas as pd
import networkx as nx
import numpy as np


def build_bipartite_graph(complaint_df):

    # inititalize graph
    G = nx.Graph()

    # add edges
    for index, row in complaint_df.iterrows():
        crid = row["crid"]
        officer_id = row["officer_id"]
        edge_data = {'incident_date': row['incident_date'], 'beat': row['beat_2012_geocoded'],
                     'category': row['complaintcategory'], 'finalfinding':row['finalfinding']}
        G.add_edge(crid, officer_id, attr_dict=edge_data)

    return G

def number_of_neighbor_complaints(G, officer_ids, lag):

    # initialize  neighbor complaints dictionary
    neighbor_complaints = {}

    # for each officer
    for u in officer_ids:

        # initialize neighbor complaint set (divided into lags)
        neighbor_set = [set() for i in range(lag)]
        for c in G[u]:
            for v in G[c]:
                if v != u:

                    # add the neighbor to the correct lag bin
                    edge_data = G.get_edge_data(c,v)
                    if edge_data['incident_date'] <= lag:
                        neighbor_set[lag].add(v)


        # transform into array and store in dictionary
        neighbor_complaints[u] = np.array([len(a) for a in neighbor_set])

    return neighbor_complaints

def sum_neighbor_complaints(G, officer_ids, lag):

    # initialize  neighbor complaints dictionary
    ret_dict = {}

    # for each officer
    for u in officer_ids:

        # initialize neighbor complaint set (divided into lags)
        neighbor_complaints = np.zeros(lag)
        for c in G[u]:
            for v in G[c]:
                if v != u:
                    edge_data = G.get_edge_data(c, v)
                    neighbor_complaints[edge_data['incident_date']] += 1

        # transform into array and store in dictionary
        ret_dict[u] = neighbor_complaints

    return ret_dict

def number_high_offender_neighbors(G, officer_ids):

    # initialize
    return
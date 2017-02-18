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

def number_of_neighbor_complaints(G, officer_ids, lag, include_self=False):

    # initialize  neighbor complaints dictionary
    neighbor_complaints = {}

    # for each officer
    for u in officer_ids: # original officer

        # initialize neighbor complaint set (divided into lags)
        neighbor_set = [set() for i in range(lag)]
        for c1 in G[u]: # complaint
            for v in G[c1]: # co-complained officer
                if v != u:
                    for c2 in G[v]:
                        if not include_self and u in G[c2]:
                            continue
                        else:
                            # add the neighbor to the correct lag bin
                            edge_data = G.get_edge_data(c2,v)
                            t = edge_data['incident_date']
                            if t <= lag:
                                neighbor_set[t].add(v)


        # transform into array and store in dictionary
        neighbor_complaints[u] = np.array([len(a) for a in neighbor_set])

    return neighbor_complaints

def sum_neighbor_complaints(G, officer_ids, lag, include_self=False):

    # initialize  neighbor complaints dictionary
    ret_dict = {}

    # for each officer
    for u in officer_ids:

        # initialize neighbor complaint set (divided into lags)
        neighbor_complaints = np.zeros(lag)
        for c1 in G[u]:
            for v in G[c1]:
                if v != u:
                    for c2 in G[v]:
                        if not include_self and u in G[c2]:
                            continue
                        else:
                            edge_data = G.get_edge_data(c2, v)
                            t = edge_data['incident_date']
                            if t <= lag:
                                neighbor_complaints[edge_data['incident_date']] += 1
                               
        # store in dictionary
        ret_dict[u] = neighbor_complaints

    return ret_dict

def number_high_offender_neighbors(G, officer_ids, deg_thresh):

    # initialize number high neighbors dictionary
    ret_dict = {}
    for u in officer_ids:

        # go through each complaint
        high_offenders = set()
        for c in G[u]:

            # go through co-ocurring officers
            for v in G[c]:

                # add them if they are high offending
                if G.degree[v] >= deg_thresh:
                    high_offenders.add(v)
                    ret_dict[u] = len(high_offenders)

    return ret_dict
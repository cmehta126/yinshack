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
        edge_data = {'LAG': int(row['LAG']), 'beat': row['beat_2012_geocoded'],
                     'category': row['complaintcategory'], 'finalfinding':row['finalfinding']}
        G.add_edge(crid, officer_id, attr_dict=edge_data)

    return G

def num_of_nbr_complaints(G, officer_ids, lag, include_self=False):

	# df['LAG'] = df['LAG'].apply(pd.to_numeric)

    # initialize  nbr complaints dictionary
    nbr_complaints = {}

    # for each officer
    for u in officer_ids: # original officer

        # initialize nbr complaint set (divided into lags)
        nbr_set = [set() for i in range(lag)]
        if u not in G.nodes():
        	# print('Warning: Officer ID %d is not in graph' %  u)
        	nbr_complaints[u] = np.array([len(a) for a in nbr_set])
        	continue

        for c1 in G[u]: # complaint
            for v in G[c1]: # co-complained officer
                if v != u:
                    for c2 in G[v]:
                        if not include_self and u in G[c2]:
                            continue
                        else:
                            # # add the nbr to the correct lag bin
                            # # print(type(v))
                            # print(type(c2))
                            # assert type(v) is int
                            # assert type(c2) is int
                            edge_data = G.get_edge_data(c2,v)
                            t = edge_data['LAG']
                            if t <= lag:
                                nbr_set[t].add(v)

        # transform into array and store in dictionary
        nbr_complaints[u] = np.array([len(a) for a in nbr_set])

    return nbr_complaints

def doublecount_nbr_complaints(G, officer_ids, lag, include_self=False):

    # initialize  nbr complaints dictionary
    ret_dict = {}

    # for each officer
    for u in officer_ids:

        # initialize nbr complaint set (divided into lags)
        nbr_complaints = np.zeros(lag)
        for c1 in G[u]:
            for v in G[c1]:
                if v != u:
                    for c2 in G[v]:
                        if not include_self and u in G[c2]:
                            continue
                        else:
                            edge_data = G.get_edge_data(c2, v)
                            t = edge_data['LAG']
                            if t <= lag:
                                nbr_complaints[edge_data['LAG']] += 1
                               
        # store in dictionary
        ret_dict[u] = nbr_complaints

    return ret_dict

def num_high_offender_nbrs(G, officer_ids, deg_thresh):

    # initialize number high nbrs dictionary
    ret_dict = {}
    for u in officer_ids:

        # go through each complaint
        high_offenders = set()
        if u not in G.nodes():
            # print('Warning: Officer ID %d is not in graph' %  u)
            ret_dict[u] = len(high_offenders)
            continue
        for c in G[u]:

            # go through co-ocurring officers
            for v in G[c]:

                # add them if they are high offending
                if G.degree(v) >= deg_thresh:
                    high_offenders.add(v)
                    ret_dict[u] = len(high_offenders)

    return ret_dict


def doublecount_num_of_nbr_complaints_past_future(G, officer_ids, lag):

    # initialize number high nbrs dictionary
    ret_dict = {}
    for u in officer_ids:

        # initialize count array
        count_array = np.zeros(2 * lag)
        # go through each complaint
        for c1 in G[u]:

            t1 = G.get_edge_data(u, c1)['LAG']

            # go through co-ocurring officers
            for v in G[c1]:
                if v != u:
                    for c2 in G[v]:
                        if c1 == c2:
                            continue

                        t2 = G.get_edge_data(v, c2)['LAG']
                        if t1 > t2:
                            count_array[t2] += 1
                        else:
                            count_array[lag + t2] += 1

        # put array in dictionary
        ret_dict[u] = count_array

    return ret_dict

def num_of_nbr_complaints_past_future(G, officer_ids, lag, include_self=False):

    # initialize  nbr complaints dictionary
    ret_dict = {}

    # for each officer
    for u in officer_ids:  # original officer
        # initialize nbr complaint set (divided into lags)
        past = [set() for i in range(lag)]
        future = [set() for i in range(lag)]
        for c1 in G[u]:  # complaint
            t1 = G.get_edge_data(c1, u)['LAG']
            for v in G[c1]:  # co-complained officer
                if v != u:
                    for c2 in G[v]:
                        if (not include_self) and (u in G[c2]):
                            continue
                        else:
                            # add the nbr to the correct lag bin
                            edge_data = G.get_edge_data(c2, v)
                            t2 = edge_data['LAG']
                            if t2 <= lag:
                                if t1 >= t2: # TODO: should this be strict?
                                    past[t2].add(v)
                                else:
                                    future[t2].add(v)


        # transform into array and store in dictionary
        ret_dict[u] = np.array([len(a) for a in past] + [len(a) for a in future])

    # # initialize number high nbrs dictionary
    # ret_dict = {}
    # for u in officer_ids:
    #
    #     # initialize count array
    #     future_set = [set() for i in range(lag)]
    #     past_set = [set() for i in range(lag)]
    #
    #     # go through each complaint
    #     for c1 in G[u]:
    #
    #         t1 = G.get_edge_data(u, c1)['LAG']
    #
    #         # go through co-ocurring officers
    #         for v in G[c1]:
    #             if v != u:
    #                 for c2 in G[v]:
    #                     if u in G[c2] and not include_self:
    #                         continue
    #                     else:
    #                         t2 = G.get_edge_data(v, c2)['LAG']
    #                         if t2 <= lag:
    #                             past_set[t2].add(c2)
    #                             # if t1 > t2:
    #                             #     past_set[t2].add(c2)
    #                             # else:
    #                             #     future_set[t2].add(c2)
    #
    #     # put array in dictionary
    #     ret_dict[u] = np.array([len(a) for a in past_set] + [len(a) for a in future_set])

    return ret_dict

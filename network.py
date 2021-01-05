
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# network tools
import networkx as nx
import igraph as ig

def get_edgelist(edge_table, engine, loiter_time=2):
    """select all edges from the database and join them with the port info from wpi
    if the node is greater than 0 (not 0 which is open ocean or null)
    and has a time diff less than 2 hours.  That should also eliminate ports a
    ship transits through but does not actually stop at.
    df_stops is a table of all ports where a ship was within 5km for more than 2 hours.
    these are the "stops" we will use to build our edgelist."""

    df_stops = pd.read_sql_query(f"""select edge.node, edge.arrival_time, 
                                 edge.depart_time, edge.time_diff,
                                 edge.destination, edge.position_count, edge.uid, 
                                 sites.port_name, sites.latitude, sites.longitude
                                 from {edge_table} as edge, sites as sites
                                 where edge.node=sites.site_id and
                                 edge.node > 0 and
                                 time_diff > '{str(loiter_time)} hours';""", engine)
    df_stops.sort_values(['uid', 'arrival_time'], inplace=True)

    # to build the edge list, we will take the pieces from stops for the current node and the next node
    df_list = pd.concat([df_stops.node, df_stops.port_name,
                         df_stops.node.shift(-1), df_stops.port_name.shift(-1),
                         df_stops.uid, df_stops.uid.shift(-1),
                         df_stops.depart_time, df_stops.arrival_time.shift(-1),
                         df_stops.latitude, df_stops.longitude], axis=1)
    # rename the columns
    df_list.columns = ['Source_id', 'Source', 'Target_id', 'Target',
                       'uid', 'target_uid', 'source_depart', 'target_arrival', 'lat', 'lon']
    # drop any row where the uid is not the same.
    # this will leave only the rows with at least 2 nodes with valid stops, making one valid edge.
    # The resulting df is the full edge list
    df_list = (df_list[df_list['uid'] == df_list['target_uid']]
               .drop('target_uid', axis=1))
    print(f"{len(df_list)} edges and {len(df_list['Source'].unique())} source nodes.")
    return df_list

def site_check(row, dist):
    if row['dist_km'] <= dist:
        val = row['nearest_site_id']
    else:
        val = 0
    return val


def calc_static_seg(df_posits, df_nn, df_sites, dist_threshold_km, loiter_time_mins):
    """
    Given a df with a lat, lon, time, and id for each position, calculate static segementation
    based on the given df_sites, a distance threshold in km, and a loiter time in minutes. This will find
    clusters of positions within a max dist_threshold_km for a min of loiter_time_mins of any known site.
    This results in a low false positive rate as a cluster must form near a known site but needs good baseline
    site data to work well.  Can be helpful as a "ground truth" dataset.  The df_nn is required to find what
    positions are within the distance threshold of the sites.
    :param df_posits: a df with lat, lon, time, id
    :param df_nn: the output of gnact.clust.calc_nn
    :param df_sites: a df with a list of sites including a site_name and site_id
    :param dist_threshold_km: int, the maximum distance positions must be for a cluster to form near a site
    :param loiter_time_mins: int, the minimum time positions must be within the distance threshold to fom a cluster
    :return: df_stops, a df that contains the node id (site_id by default) of the stop, the next destination, the
    arrival time, the depart time, the time_diff, position_count at the stop, the site_id, the site_name, the lat and
    lon of the site, and the region (if applicable).
    """
    df = pd.merge(df_posits, df_nn, how='inner', left_on='id', right_on='id')
    # any duplicates will cause problems down the line.  catch them here.
    df.drop_duplicates(inplace=True)
    # site_check takes the dist to nearest port and if its less than dist, populates
    # site_id with the nearest port id.  If the nearest site is greater than dist,
    # site_id = 0.  0 will be used for activity "not at site"
    df['node'] = df.apply(site_check, args=(dist_threshold_km,), axis=1)
    # no longer need port_id and dist
    df.drop(['nearest_site_id', 'dist_km'], axis=1, inplace=True)
    # use shift to get the next node and previous node
    df['next_node'] = df['node'].shift(-1)
    df['prev_node'] = df['node'].shift(1)
    # reduce the dataframe down to only the positions where the previous node is
    # different from the next node.  These are the transitions between nodes
    df_reduced = (df[df['next_node'] != df['prev_node']]
                  .reset_index())
    # make a df of all the starts and all the ends.  When the node is the same as
    # the next node (but next node is different than previous node), its the start
    # of activity at a node.  Similarly, when the node is the same as the previous
    # node (but the next node is different than previous node), its the end of activity.
    df_starts = (df_reduced[df_reduced['node'] == df_reduced['next_node']]
                 .rename(columns={'time': 'arrival_time'})
                 .reset_index(drop=True))
    df_ends = (df_reduced[df_reduced['node'] == df_reduced['prev_node']]
               .rename(columns={'time': 'depart_time'})
               .reset_index(drop=True))
    # now take all the pieces which have their indices reset and concat
    df_final = (pd.concat([df_starts['node'], df_ends['next_node'], df_starts['arrival_time'],
                           df_ends['depart_time'], df_starts['index']], axis=1)
                .rename(columns={'next_node': 'destination'}))
    # add in a time difference column.  cast to str because postgres doesnt like
    # pandas time intervals
    df_final['time_diff'] = df_final['depart_time'] - df_final['arrival_time']

    # find the position count by subtracting the current index from the
    # shifted index of the next row
    df_final['position_count'] = df_final['index'].shift(-1) - df_final['index']
    df_final.drop('index', axis=1, inplace=True)

    # apply the loiter time filter
    df_final = df_final[df_final['time_diff'] > pd.to_timedelta(loiter_time_mins, 'minutes')]

    # sort by time, filter all stops in the "open"
    df_stops = df_final.sort_values('arrival_time')
    df_stops = df_stops[df_stops.node > 0]
    # get the
    df_stops = pd.merge(df_stops, df_sites, how='left', left_on='node', right_on='site_id')

    return df_stops


def get_weighted_edgelist(df_edgelist):
    # This produces a df that is the summarized edge list with weights
    # for the numbers of a time a ship goes from the source node to the target node.
    # The code executes groupby the source/target id/name, count all the rows, drop the time fields,
    # rename the remaining column from uid to weight, and reset the index
    df_edgelist_weighted = (df_edgelist.groupby(['Source_id', 'Source',
                                                 'Target_id', 'Target'])
                            .count()
                            .drop(['source_depart', 'target_arrival'], axis=1)
                            .rename(columns={'uid': 'weight'})
                            .reset_index())
    return df_edgelist_weighted


def plot_uid(uid, df_edgelist):
    """ This function will plot the path of a given uid across an edgelist df.
    :param uid: the str or int value of a uid within the df
    :param df_edgelist: a df_edgelist
    :return: a plot
    """
    """"""
    uid_edgelist = df_edgelist[df_edgelist['uid'] == uid].reset_index(drop=True)
    uid_edgelist = uid_edgelist[['Source', 'source_depart', 'Target', 'target_arrival']]
    # build the graph
    G = nx.from_pandas_edgelist(uid_edgelist, source='Source', target='Target',
                                edge_attr=True, create_using=nx.MultiDiGraph)
    # get positions for all nodes using circular layout
    pos = nx.circular_layout(G)
    # draw the network
    plt.figure(figsize=(6, 6))
    # draw nodes
    nx.draw_networkx_nodes(G, pos)
    # draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # add a buffer to the x margin to keep labels from being printed out of margin
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    # plot the title and turn off the axis
    plt.title(f'Network Plot for uid {str(uid).title()}')
    plt.axis('off')
    plt.show()
    print(uid_edgelist)


def plot_from_source(source, df):
    """This function will plot all the nodes visited from a given node."""
    # create the figure plot
    plt.figure(figsize=(8, 6))
    # get a df with just the source port as 'Source'
    df_g = df[df['Source'] == source.upper()]  # use upper to make sure fits df
    # build the network
    G = nx.from_pandas_edgelist(df_g, source='Source',
                                target='Target', edge_attr='weight',
                                create_using=nx.DiGraph)
    # get positions for all nodes
    pos = nx.spring_layout(G)
    # adjust the node lable position up by .1 so self loop labels are separate
    node_label_pos = {}
    for k, v in pos.items():
        node_label_pos[k] = np.array([v[0], v[1] + .1])
    # get edge weights as dictionary
    weights = [i['weight'] for i in dict(G.edges).values()]
    #  draw nodes
    nx.draw_networkx_nodes(G, pos)
    # draw node labels
    nx.draw_networkx_labels(G, node_label_pos, font_size=10, font_family='sans-serif')
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=scale_range(weights, .5, 5))
    # plot the title and turn off the axis
    plt.title('Weighted Network Plot for {} Port as Source'.format(source.title()),
              fontsize=16)
    plt.axis('off')
    # # make a test boc for the weights.  Can be plotted on edge, but crowded
    # box_str = 'Weights out from {}: \n'.format(source.title())
    # for neighbor, values in G[source.upper()].items():
    #     box_str = (box_str + neighbor.title() + ' - ' +
    #                str(values[0]['weight']) + '\n')

    plt.text(-1.2, -1.2, 'Edge Weights are scaled between 0.5 and 5 for visualization.', fontsize=12,
             verticalalignment='top', horizontalalignment='left')
    plt.show()  # display

    print(df_g[['Target', 'weight']].sort_values('weight', ascending=False).reset_index())


# community detection
from cdlib import viz
from cdlib import NodeClustering
from cdlib import algorithms

from sklearn import metrics


def get_truth_communities(df_truth, df_edges):
    """
    Returns a CDLib NodeClustering object gpt use in comupting and plotting communities
    :param df_truth: a dataframe with a "node" and "truth" column listing each nodes "true" community
    :param df_edges" a dataframe with the edges as "source" and "target"
    :return:
    """
    # make the nx graph
    nx_g = nx.from_edgelist(df_edges.values)
    # need to make a nodeclustering object for ground truth for later use
    # first get all the unique communities
    truth_coms = df_truth['truth'].unique()
    # define an empty list to catch a list of all nodes in each community
    communities_list = list()
    # iterate through the communities to get the nodes in each community and add them to the set as frozensets
    for com in truth_coms:
        com_nodes = df_truth[df_truth['truth'] == com].iloc[:, 0].values
        communities_list.append(com_nodes.tolist())
    # make the nodeclustering object
    ground_truth_com = NodeClustering(communities=communities_list, graph=nx_g, method_name="ground_truth")
    return ground_truth_com

def evaluate_comms(nx_g, df_nodes, name, algo):
    # get the predicted communities from the algo
    pred_coms = algo(nx_g)
    communities = pred_coms.communities

    # need to convert the community groups from list of lists to a dict of lists for ingest to df
    coms_dict = dict()
    for c in range(len(communities)):
        for i in communities[c]:
            coms_dict[i] = [c]

    # make a df with the results of the algo
    df_coms = pd.DataFrame.from_dict(coms_dict).T.reset_index()
    df_coms.columns = ['node', name]
    # merge this results with the df_nodes to keep track of all the nodes' clusters
    df_compare = pd.merge(df_nodes, df_coms, how='left', left_on='node', right_on='node')

    # We can then just adjusted mutual info to find similarity score
    ami_score = metrics.adjusted_mutual_info_score(df_compare[name], df_compare['truth'])
    print(f'The AMI for {name} algorithm is {round(ami_score, 3)}.')
    results = {'name': name,
               'AMI': round(ami_score, 3),
               'pred_coms': pred_coms,
               'numb_communities': len(communities),
               'truth_communities': len(df_compare['truth'].unique())}

    return results, pred_coms

def analyze_comms(df_edges, df_truth, graph_name, algo_dict, print_verbose=True):
    # need to read the edges df as a list of values into networkx
    nx_graph = nx.from_edgelist(df_edges.values)

    # most algos need the largest connected component (lcc) to find communities, so lets do that next.
    # use networkx to build the graph, find the nodes of the lcc, and build the subgraph of interest
    lcc = max(nx.connected_components(nx_graph), key=len)
    nx_g = nx_graph.subgraph(lcc)

    # define the positions here so all the cluster plots in the loop are the same structure
    pos = nx.spring_layout(nx_g)

    # adjust label pos
    label_pos = dict()
    for k, v in pos.items():
        label_pos[k] = (v[0], (v[1] + 0.05))

    # we need to only include nodes that are in the lcc, so lets filter down the df_truth
    # next reduce our ground truth df down to just those nodes in the lcc.
    df_truth = df_truth[df_truth['node'].isin(list(lcc))]

    # use gnact's network module to get the NodeClustering object from the truth and edges
    ground_truth_com = get_truth_communities(df_truth, df_edges)

    # plot the original network with the ground truth communities
    viz.plot_network_clusters(nx_g, ground_truth_com, pos, figsize=(10, 5))
    nx.draw_networkx_labels(nx_g, pos=label_pos)
    plt.title(f'Ground Truth of {graph_name}')
    plt.show()

    # make a dict to store results about each algo
    results_list = []
    # make a df with all the nodes.  will capture each model's clustering
    df_nodes = pd.DataFrame(list(nx_g.nodes))
    df_nodes.columns = ['node']
    df_nodes = pd.merge(df_nodes, df_truth, how='left', left_on='node', right_on='node')


    for name, algo in algo_dict.items():
        try:
            # use GNACT's abstraction of CDLib's library of algorithms
            results, pred_coms = evaluate_comms(nx_g, df_nodes, name, algo)
            results_list.append(results)

            if print_verbose==True:
                # plot the network clusters
                viz.plot_network_clusters(nx_g, pred_coms, pos, figsize=(10, 5))
                nx.draw_networkx_labels(nx_g, pos=label_pos)
                plt.title(f"Clusters for {name} algo of {graph_name}, \n AMI = {round(results['AMI'], 3)}")
                plt.show()

        except Exception as e:
            print(f'Error with algorithm {name}:')
            print(e)

    return results_list
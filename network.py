
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# network tools
import networkx as nx

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

def build_uid_lists(source_table, target_table, conn):
    """A function that build a list of all unique uids in the source table that are not already
     in the target table.  This can be used to resume processing if interrupted by only continuing
     to process the uids required."""
    print('Building the uid lists...')
    # This list is all of the uids in the table of interest.  It is the
    # total number of uids we will be iterating over.
    c = conn.cursor()
    c.execute(f"""SELECT DISTINCT(uid) FROM {source_table};""")
    uid_list_potential = c.fetchall()
    c.close()

    # if we have to stop the process, we can use the uids that are already completed
    # to build a new list of uids left to complete.  this will allow us to resume
    # processing without repeating any uids.
    c = conn.cursor()
    c.execute(f"""SELECT DISTINCT(uid) FROM {target_table};""")
    uid_list_completed = c.fetchall()
    c.close()

    # find the uids that are not in the edge table yet
    diff = lambda l1, l2: [x for x in l1 if x not in l2]
    uid_list = diff(uid_list_potential, uid_list_completed)
    print('UID lists built.')
    return uid_list


def plot_uid(uid, df_edgelist):
    """This function will plot the path of a given uid across an edgelist df."""
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

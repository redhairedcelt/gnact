from datetime import timedelta
from geopy.distance import great_circle

"""
INPUTS:
    df={o1,o2,...,on} Set of objects
    spatial_threshold = Maximum geographical coordinate (spatial) distance value in kilometers
    min_neighbors = Minimun number of points within Eps1 and Eps2 distance
OUTPUT:
    C = {c1,c2,...,ck} Set of clusters
"""


def DBSCAN(df, spatial_threshold, min_neighbors):
    cluster_label = 0
    NOISE = -1
    UNMARKED = 777777
    stack = []

    # initialize each point with unmarked
    df['clust_id'] = UNMARKED

    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['clust_id'] == UNMARKED:
            neighborhood = retrieve_neighbors(index, df, spatial_threshold)

            if len(neighborhood) < min_neighbors:
                df.at[index, 'clust_id'] = NOISE

            else:  # found a core point
                cluster_label = cluster_label + 1
                df.at[index, 'clust_id'] = cluster_label  # assign a label to core point

                for neig_index in neighborhood:  # assign core's label to its neighborhood
                    df.at[neig_index, 'clust_id'] = cluster_label
                    stack.append(neig_index)  # append neighborhood to stack

                while len(stack) > 0:  # find new neighbors from core point neighborhood
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(current_point_index, df, spatial_threshold)

                    if len(new_neighborhood) >= min_neighbors:  # current_point is a new core
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['clust_id']
                            if (neig_cluster != NOISE) & (neig_cluster == UNMARKED):
                                # TODO: verify cluster average before add new point
                                df.at[neig_index, 'clust_id'] = cluster_label
                                stack.append(neig_index)
    return df


def retrieve_neighbors(index_center, df, spatial_threshold):
    neigborhood = []

    center_point = df.loc[index_center]
    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle((center_point['lat'], center_point['lon']),
                                    (point['lat'], point['lon'])).kilometers
            if distance <= spatial_threshold:
                neigborhood.append(index)

    return neigborhood
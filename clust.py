# utility modules
import pandas as pd
import numpy as np
import datetime

import skmob
from sklearn.neighbors import BallTree
from skmob.preprocessing import detection
from gnact.cluster_methods import stdbscan

# sklearn tools
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import hdbscan
from sklearn.metrics.pairwise import haversine_distances

# %%

def get_uid_posits(uid, engine_pg, start_time='2017-01-01 00:00:00', end_time='2018-01-01 00:00:00'):
    """
    Given a UID, engine connection to a database, and a time range, this function will return all points
    in the database with the id ordered by time.
    :param uid: str
    :param engine_pg: SQL Alchemy Engine
    :param start_time: str formatted for PostGreSQL dates
    :param end_time: str formatted for PostGreSQL dates
    :return: a Pandas df with the id, time, lat, lon
    """
    read_sql = f"""SELECT id, lat, lon, time
                FROM uid_positions
                WHERE uid = '{uid[0]}'
                AND time between '{start_time}' and '{end_time}'
                ORDER by time"""
    df_posits = pd.read_sql_query(read_sql, con=engine_pg)
    return df_posits


def get_sites_wpi(engine):
    sites = pd.read_sql('sites', engine, columns=['site_id', 'port_name',
                                                'latitude', 'longitude', 'region'])
    df_sites = sites[sites['region'] != None]
    df_sites = df_sites.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'port_name': 'site_name'})
    return df_sites


def calc_clusts(df, method, eps_km=None, min_samp=None, eps_time=None):
    """
    Given a Pandas df with a lat, lon, time, and id named 'lat', 'lon', 'time', and 'id',
    this function will return another df with the results of a clustering algo.
    Options for clustering algorithm are
    :param eps_time:
    :param df: a df with a unique id, lat and lon in decimal degrees
    :param eps_km: The epsilon (or max_epsilon_ value used in the clustering algo.
    :param min_samp: The minimum samples for a cluster to form
    :param method: The clustering method used
    :return: a Pandas df with the id, lat, lon, and cluster id from the algo.
    """
    try:
        # need to add tests to ensure each column exists and is in the right format
        # round to the minute and drop duplicates
        df['time'] = df['time'].dt.floor('min')
        df.drop_duplicates('time', keep='first')
        # format data for clustering
        X = (np.radians(df.loc[:, ['lon', 'lat']].values))
        x_id = df.loc[:, 'id'].astype('int').values
        method = str(method)
    except Exception as e:
        print("Unable to convert 'lat', 'lon', 'time', or 'id' values.  Ensure columns are labeled correctly.")
        print(e)
    # execute the clustering method
    try:
        if method == 'dbscan':
            # execute sklearn's DBSCAN
            dbscan = DBSCAN(eps=eps_km / 6371, min_samples=min_samp, algorithm='kd_tree',
                            metric='euclidean', n_jobs=1)
            dbscan.fit(X)
            results_dict = {'id': x_id, 'clust_id': dbscan.labels_, 'time': df['time'].values,
                            'lat': df['lat'].values, 'lon': df['lon'].values}
            # gather the output as a dataframe
            df_clusts = pd.DataFrame(results_dict)
        elif method == 'optics':
            # execute sklearn's OPTICS
            # 5km in radians is max eps
            optics = OPTICS(max_eps=eps_km / 6371, min_samples=min_samp, metric='euclidean', cluster_method='xi',
                            algorithm='kd_tree', n_jobs=1)
            optics.fit(X)
            results_dict = {'id': x_id, 'clust_id': optics.labels_, 'time': df['time'].values,
                            'lat': df['lat'].values, 'lon': df['lon'].values}
            # gather the output as a dataframe
            df_clusts = pd.DataFrame(results_dict)
        elif method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=True,
                                        gen_min_span_tree=False, leaf_size=40,
                                        metric='euclidean', min_cluster_size=min_samp, min_samples=1)
            clusterer.fit(X)
            results_dict = {'id': x_id, 'clust_id': clusterer.labels_, 'time': df['time'].values,
                            'lat': df['lat'].values, 'lon': df['lon'].values}
            # gather the output as a dataframe
            df_clusts = pd.DataFrame(results_dict)
        elif method == 'stdbscan':
            # execute ST_DBSCAN. eps1 in km, eps2 in minutes
            df_clusts = stdbscan.ST_DBSCAN(df=df, spatial_threshold=eps_km, temporal_threshold=eps_time,
                                           min_neighbors=min_samp)
        elif method == 'dynamic':
            # make a new df to hold results
            df_clusts = pd.DataFrame()
            # turn the positions into a traj_df withink skmob package
            tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='time')
            # the stops_traj_df has one row per each stop
            stdf = detection.stops(tdf, minutes_for_a_stop=eps_time, spatial_radius_km=eps_km, leaving_time=True,
                                   no_data_for_minutes=360, min_speed_kmh=70)
            if len(stdf) > 0:
                # iterate through the stdf and make a df with all positions within each cluster
                # appropriately labeled with that cluster id
                for i in range(len(stdf)):
                    # get cluster start, stop, and the id for this cluster
                    cluster_start = stdf.datetime.loc[i, ]
                    cluster_end = stdf.leaving_datetime.loc[i, ]
                    cluster_id = i
                    # gather all the position reports in the timeframe of this cluster
                    cluster = tdf.loc[(tdf.datetime > cluster_start) & (tdf.datetime < cluster_end)].copy()
                    cluster['clust_id'] = cluster_id
                    cluster['time'] = cluster['datetime']
                    cluster['lon'] = cluster['lng']
                    cluster.drop(['lng', 'datetime'], inplace=True, axis=1)
                    df_clusts = df_clusts.append(pd.DataFrame(cluster), ignore_index=True)
        else:
            print("Error.  Method must be 'dbscan', 'optics', 'hdbscan', stdbscan', or 'dynamic'.")
            return None
    except Exception as e:
        print('UID error in clustering.')
        print(e)
        return None
    # drop all -1 clust_id, which are all points not in clusters
    if type(pd.DataFrame()) == type(df_clusts) and len(df_clusts) > 0:
        df_clusts = df_clusts[df_clusts['clust_id'] != -1]
    return df_clusts


def calc_nn(df_posits, df_sites, lat='lat', lon='lon', id='id'):
    """
    This function finds the nearest site_id in df_sites for each posit in df_posits.  Returns a df with the id of the
    posit from df_posits, the nearest site_id, and the distance in kilometers
    :param df_posits: a df with id, time, lat, and lon
    :param df_sites: a df with site_id, site_name, lat, and lon
    :return: a df with id, site_id, and distance in km
    """
    # build the BallTree using the sites as the candidates
    candidates = np.radians(df_sites.loc[:, ['lat', 'lon']].values)
    ball_tree = BallTree(candidates, leaf_size=40, metric='euclidean')
    # Now we are going to use sklearn's BallTree to find the nearest neighbor of
    # each position for the nearest port.
    points_of_int = np.radians(df_posits.loc[:, [lat, lon]].values)
    # query the tree
    dist, ind = ball_tree.query(points_of_int, k=1)
    # make the df from the results and original id values
    df_nn = pd.DataFrame(np.column_stack([df_posits[id].values,
                                          df_sites.iloc[ind.reshape(1, -1)[0], :].site_id.values,
                                          np.round((dist.reshape(1, -1)[0]) * 6371.0088, decimals=2)]),
                         columns=['id', 'nearest_site_id', 'dist_km'])
    df_nn['id'] = df_nn['id'].astype('int')
    df_nn['nearest_site_id'] = df_nn['nearest_site_id'].astype('int')
    return df_nn


def calc_centers(df_clusts, clust_id_value='clust_id'):
    """
    This function finds the center of a cluster from dbscan results (given lat, lon, time, and clust_id columns),
    and finds the average distance for each cluster point from its cluster center, as well as the min and max times.
    Returns a df.
    :param df_clusts: the results of calc_clusters
    :param clust_id_value: the column name for the clust_id column.
    :return: returns a df for each cluster with the cluster id, start time, end time, total cluster count,
    and location of the cluster.
    """
    # make a new df from the df_clusts grouped by cluster id
    # with an aggregation for min/max/count of times and the mean for lat and long
    df_centers = (df_clusts.groupby([clust_id_value])
                  .agg({'time': [min, max, 'count'],
                        'lat': 'mean',
                        'lon': 'mean'})
                  .reset_index(drop=False))
    df_centers.columns = ['clust_id', 'time_min', 'time_max', 'total_clust_count', 'average_lat', 'average_lon']
    # find the average distance from the centerpoint
    # We'll calculate this by finding all of the distances between each point in
    # df_clusts and the center of the cluster.  We'll then take the min and the mean.
    haver_list = []
    for i in df_centers[clust_id_value]:
        X = (np.radians(df_clusts[df_clusts[clust_id_value] == i]
                        .loc[:, ['lat', 'lon']].values))
        Y = (np.radians(df_centers[df_centers[clust_id_value] == i]
                        .loc[:, ['average_lat', 'average_lon']].values))
        haver_result = (haversine_distances(X, Y)) * 6371.0088  # km to radians
        haver_dict = {clust_id_value: i, 'average_dist_from_center': np.mean(haver_result)}
        haver_list.append(haver_dict)
    # merge the haver results back to df_centers
    haver_df = pd.DataFrame(haver_list)
    df_centers = pd.merge(df_centers, haver_df, how='left', on=clust_id_value)
    df_centers['time_diff'] = df_centers['time_max'] - df_centers['time_min']
    return df_centers


def calc_stats(df_clusts, df_stops, dist_threshold_km=5):
    """
    Takes a set of predicted clusters and "ground-truth" clusters and calculates the precision and recall.
    :param df_clusts: Predicted clusters.  Generated by clust.calc_clusts
    :param df_stops:  Ground Truth clusters.  Can be manually labled or generated by calc_clusts 'static trip'
    segmentation method.  Dataframe must have a column labeled 'site_id' and a row per cluster.
    :param dist_threshold_km: distance in km a predicted cluster must be within a groud truth cluster to be correct.
    :return: a dictionary with precision, recall, and f1 measure
    """
    # rollup the clusters to their center points
    df_centers = calc_centers(df_clusts)
    # calc the nearest stop to the clusters, including the distance.  We could also use df_sites and
    # filter down to just sites in df_stops, but this is faster.
    df_nearest_sites = calc_nn(df_centers, df_stops, lat='average_lat', lon='average_lon', id='clust_id')
    # correct clusters are within distance threshold of a visited point so we filter by dist_threshold
    df_clust_rollup_correct = df_nearest_sites[(df_nearest_sites.dist_km < dist_threshold_km)]

    # now group stops and clust_rollup by their site_ids, and select just one column.
    # the result is a series with site_id as index that can be used to calc precision, recall, and f1 measure
    df_stops_grouped = df_stops.groupby('site_id').agg('count').iloc[:, 0]
    df_rollup_grouped = df_clust_rollup_correct.groupby(['nearest_site_id']).agg('count').iloc[:, 0]

    # get the proportion of each site within stops to use for the recall
    total_prop_stops = df_stops_grouped / df_stops_grouped.sum()
    # get raw recall, which we calc by cluster.  therefore recall is number of clusters found at a site
    # divided by the total number of clusters at that site.  If the value is more than 1, set it to 1.
    recall_raw = (df_rollup_grouped / df_stops_grouped)
    recall_raw[recall_raw > 1] = 1
    # now multiply raw_recall by the total proportion to get a weighted value, and sum it.
    recall = (recall_raw * total_prop_stops).sum()

    # precision is the proportion of correct clusters to all clusters found.  since we are using df_stops,
    # correct clusters are any clusters with a calculated distance less than the distance threshold.
    precision = len(df_clust_rollup_correct) / len(df_centers)

    # now determine f1 measure
    f_measure = 2 * ((precision * recall) / (precision + recall))

    stats_dict = {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f_measure, 4)}
    return stats_dict

def get_df_stats(df_clusts, df_stops, dist_threshold_km=3):
    """
    Determines key metrics and stats for a df_clusts DataFrame from calc_clusts when compared to a "ground truth."
    The df_stops must have a node,
    :param df_clusts:
    :param df_stops: DataFrame.  List of ground truth clusters or sites used to determine the performance of df_clusts.
    Must include a lat and lon as a float and a site_id column with a float.
    :param dist_threshold_km:
    :return:
    """
    # need to add tests for required columns.
    # also can add in temporal filter similar to distance.
    # rollup the clusters to their center points
    df_centers = calc_centers(df_clusts)
    # find the closest ground truth cluster or site for each cluster in df_centers.
    df_nearest_sites = calc_nn(df_centers, df_stops, lat='average_lat', lon='average_lon', id='clust_id')
    # merge all the data together.  df_clust_rollup now includes all details about the cluster, the ground truth,
    #and the distance between each cluster and its nearest ground truth.
    df_clust_rollup = pd.merge(df_centers, df_nearest_sites, how='inner', left_on='clust_id', right_on='id')

    # false positives are clusters more than the threshold distance from a correct site in stops
    df_fp = df_clust_rollup.loc[(df_clust_rollup.dist_km >= dist_threshold_km)].copy()
    df_fp['results'] = 'False Positive'
    # true positices are within the threshold distance
    df_tp = df_clust_rollup.loc[(df_clust_rollup.dist_km < dist_threshold_km)].copy()
    df_tp['results'] = 'True Positive'
    # false negatives are techincally every cluster not within the threshold for each cluster found.
    # however, we will just plot the stops that were missed rather than match up each cluster.
    df_fn = df_stops.loc[~df_stops.site_id.isin(df_tp.nearest_site_id.tolist())].copy()
    df_fn['results'] = 'False Negative'

    # now concat all the pieces and rename to df_stats
    df_stats = pd.concat([df_fp, df_fn, df_tp]).reset_index(drop=True)
    df_stats = df_stats.drop(['node', 'destination', 'arrival_time', 'depart_time', 'region', 'id'], axis=1)
    return df_stats


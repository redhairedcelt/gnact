# utility modules
import pandas as pd
import numpy as np
import folium
import datetime

import skmob
from sklearn.neighbors.ball_tree import BallTree
from skmob.preprocessing import detection

from gnact import stdbscan

# sklearn tools
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import hdbscan
from sklearn.metrics.pairwise import haversine_distances


# %%

def get_uid_posits(uid, engine_pg, start_time='2017-01-01 00:00:00', end_time='2018-01-01 00:00:00'):
    """
    Given a UID and time range, this function will return all points in the database with the id ordered by time.
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


def calc_clusts(df, method, eps_km=None, min_samp=None, eps_time=None, Ceps=None):
    """
    Given a Pandas df with a lat and lon, this function will return another df with the results of a clustering algo.
    Currently limited to SciKit-Learn's DBSCAN and OPTICS, but more will be added
    :param Ceps:
    :param eps_time:
    :param df: a df with a unique id, lat and lon in decimal degrees
    :param eps_km: The epsilon (or max_epsilon_ value used in the clustering algo.
    :param min_samp: The minimum samples for a cluster to form
    :param method: The clustering method used
    :return: a Pandas df with the id, lat, lon, and cluster id from the algo.
    """
    try:
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
                                        metric='euclidean', min_cluster_size=5, min_samples=min_samp)
            clusterer.fit(X)
            results_dict = {'id': x_id, 'clust_id': clusterer.labels_, 'time': df['time'].values,
                            'lat': df['lat'].values, 'lon': df['lon'].values}
            # gather the output as a dataframe
            df_clusts = pd.DataFrame(results_dict)
        elif method == 'stdbscan':
            # execute ST_DBSCAN. eps1 in km, eps2 in minutes
            df_clusts = stdbscan.ST_DBSCAN(df=df, spatial_threshold=eps_km, temporal_threshold=eps_time,
                                           min_neighbors=min_samp)
        elif method == 'tdbscan':
            df_clusts = tdbscan.T_DBSCAN(df, Ceps, eps_km, min_samp)

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
                    cluster = tdf[(tdf.datetime > cluster_start) & (tdf.datetime < cluster_end)]
                    cluster['clust_id'] = cluster_id
                    cluster['time'] = cluster['datetime']
                    cluster['lon'] = cluster['lng']
                    cluster.drop(['lng', 'datetime'], inplace=True, axis=1)
                    df_clusts = df_clusts.append(pd.DataFrame(cluster), ignore_index=True)

        else:
            print("Error.  Method must be 'dbscan', 'optics', 'hdbscan', stdbscan', dynamic,"
                  "or static_segmentation.")
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
    """This function finds the center of a cluster from dbscan results (given lat, lon, time, and clust_id columns),
    and finds the average distance for each cluster point from its cluster center, as well as the min and max times.
    Returns a df."""
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


def plot_clusters(df_posits, df_centers):
    # plot the track
    m = folium.Map(location=[df_posits.lat.median(), df_posits.lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    points = list(zip(df_posits.lat, df_posits.lon))
    folium.PolyLine(points).add_to(m)
    # plot the clusters
    for row in df_centers.itertuples():
        popup = folium.Popup(f"Cluster: {row.clust_id}   Count: {row.total_clust_count}<BR>Average Dist from center "
                             f"{round(row.average_dist_from_center, 2)}<BR>Min Time:  {row.time_min}<BR>Max Time: "
                             f"{row.time_max}<BR>Time Diff: {row.time_diff}", max_width=220)
        folium.Marker(location=[row.average_lat, row.average_lon], icon=folium.Icon(color='blue'),
                      popup=popup).add_to(m)
    print(f'Plotted {len(df_centers)} total clusters.')
    return m


def get_sites_wpi(engine):
    sites = pd.read_sql('sites', engine, columns=['site_id', 'port_name',
                                                'latitude', 'longitude', 'region'])
    df_sites = sites[sites['region'] != None]
    df_sites = df_sites.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'port_name': 'site_name'})
    return df_sites

def plot_sites(df_sites):
    # build the map
    m = folium.Map(location=[df_sites.lat.median(), df_sites.lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    # plot the sites
    for row in df_sites.itertuples():
        popup = folium.Popup(f"Site ID: {row.site_id}<BR>Site Name: {row.site_name}"
                             f"<BR>Region: {row.region}", max_width=220)
        folium.Marker(location=[row.lat, row.lon], icon=folium.Icon(color='gray'),
                      popup=popup).add_to(m)
    print(f'Plotted {len(df_sites)} total sites.')
    return m


def calc_stats(df_clusts, df_stops, dist_threshold_km=3):
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
    # rollup the clusters to their center points
    df_centers = calc_centers(df_clusts)
    # assemble df_stats for plotting and review
    df_nearest_sites = calc_nn(df_centers, df_stops, lat='average_lat', lon='average_lon', id='clust_id')
    df_clust_rollup = pd.merge(df_centers, df_nearest_sites, how='inner', left_on='clust_id', right_on='id')

    # false positives are clusters more than the threshold distance from a correct site in stops
    df_fp = df_clust_rollup[(df_nearest_sites.dist_km >= dist_threshold_km)]
    df_fp['results'] = 'False Positive'
    # true positices are within the threshold distance
    df_tp = df_clust_rollup[(df_nearest_sites.dist_km < dist_threshold_km)]
    df_tp['results'] = 'True Positive'
    # false negatives are techincally every cluster not within the threshold for each cluster found.
    # however, we will just plot the stops that were missed rather than match up each cluster.
    df_fn = df_stops[~df_stops.site_id.isin(df_tp.nearest_site_id.tolist())]
    df_fn['results'] = 'False Negative'

    df_stats = pd.concat([df_fp, df_fn, df_tp]).reset_index(drop=True)
    df_stats = df_stats.drop(['node', 'destination', 'arrival_time', 'depart_time', 'region', 'id'], axis=1)
    return df_stats

def plot_stats(df_stats, df_posits):
    m = folium.Map(location=[df_stats.average_lat.median(), df_stats.average_lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    points = list(zip(df_posits.lat, df_posits.lon))
    folium.PolyLine(points).add_to(m)

    # plot the falase positive, false negatives, and true positives
    for idx, row in df_stats.iterrows():
        if row['results'] == 'False Positive':
            popup = folium.Popup(f"False Positive <BR> Cluster: {row.nearest_site_id}  Count: {row.total_clust_count}" +
                                 f"<BR> Site_id: {row.site_id}  Site_name: {row.site_name}",
                                 max_width=220)
            folium.Marker(location=[row.average_lat, row.average_lon], icon=folium.Icon(color='orange'),
                          popup=popup).add_to(m)
        elif row['results'] == 'False Negative':
            popup = folium.Popup(f"False Negative <BR> Site_id: {row.site_id}  Site_name: {row.site_name}",
                                 max_width=220)
            folium.Marker(location=[row.lat, row.lon], icon=folium.Icon(color='red'),
                          popup=popup).add_to(m)
        elif row['results'] == 'True Positive':
            popup = folium.Popup(f"True Positive <BR> Cluster: {row.clust_id}   Count: {row.total_clust_count}" +
                                 f"<BR> Site_id: {row.site_id}  Site_name: {row.site_name}",
                                 max_width=220)
            folium.Marker(location=[row.average_lat, row.average_lon], icon=folium.Icon(color='green'),
                          popup=popup).add_to(m)
    print(f'Plotted {len(df_stats)} total clusters.')
    return m



def calc_dist(df, unit='nm'):
    """
    Takes a df with id, lat, and lon and returns the distance  between
    the previous point to the current point as a series.
    :param df:
    :return:
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [df.lon.shift(1), df.lat.shift(1), df.lon, df.lat])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    r = 2 * np.arcsin(np.sqrt(a))
    if unit == 'mile':
        return 3958.748 * r
    if unit == 'km':
        return 6371 * r
    if unit == 'nm':
        return 3440.65 * r
    else:
        print("Unit is not valid.  Please use 'mile', 'km', or 'nm'.")
        return None


def calc_bearing(df):
    """
    Takes a df with id, lat, and lon and returns the computed bearing between
    the previous point to the current point as a series.
    :param df:
    :return:
    """
    lat1 = np.radians(df.lat.shift(1))
    lat2 = np.radians(df.lat)
    dlon = np.radians(df.lon - df.lon.shift(1))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return round(compass_bearing, 2)


def traj_enhance_df(df):
    """
    Takes a df with id, lat, lon, and time.  Returns a df with these same columns as well as
    time_rounded to the lowest minute, time difference from current point and previous group,
    time_diff_hours, course over ground, distance traveled since last point, and speed in knots
    :param df:
    :return:
    """
    # we want to round by minute and drop any duplicates
    df['time_rounded'] = df.time.apply(lambda x: x.floor('min'))
    df.drop_duplicates(['time'], keep='first', inplace=True)
    # calculate time diff between two points
    df['time_diff'] = df.time - df.time.shift(1)
    # time diff in hours needed for speed calc
    df['time_diff_hours'] = pd.to_timedelta(df.time_diff, errors='coerce').dt.total_seconds() / 3600
    # calculate bearing, distance, and speed
    df['cog'] = calc_bearing(df)
    df['dist_nm'] = calc_dist(df, unit='nm')
    df['speed_kts'] = df['dist_nm'] / df['time_diff_hours']
    return df


def postgres_dbscan_reworked(uid, eps_km, min_samp, method):
    """
    A function to conduct dbscan on the server for a global eps and min_samples value.
    Optimized for multiprocessing.
    :param min_samp:
    :param eps:
    :param uid:
    :return:
    """
    # iteration_start = datetime.datetime.now()
    params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
    # execute dbscan script
    dbscan_postgres_sql = f"""
    UPDATE clustering_results as c 
    SET {params_name} = t.clust_id
    FROM (SELECT id , ST_ClusterDBSCAN(geom, eps := {eps}, minpoints := {min_samp})
          over () as clust_id
          FROM uid_positions
          WHERE uid = '{uid[0]}'
          AND time between '{start_time}' and '{end_time}') as t
          WHERE t.id = c.id
          AND t.clust_id IS NOT NULL;"""
    conn_pg = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pg = conn_pg.cursor()
    c_pg.execute(dbscan_postgres_sql)
    conn_pg.commit()
    c_pg.close()
    # add the uid to the tracker and get current uid count from tracker
    uids_completed = utils.add_to_uid_tracker(uid, conn_pg)
    conn_pg.close()

    print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
    percentage = (uids_completed / len(uid_list)) * 100
    print(f'Approximately {round(percentage, 3)} complete.')

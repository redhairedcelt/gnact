# utility modules
import pandas as pd
import numpy as np
import folium
import datetime


# sklearn tools
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import hdbscan
from sklearn.metrics.pairwise import haversine_distances

from gnact import stdbscan
from gnact import tdbscan


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


def get_clusters(df, method, eps_km=None, min_samp=None, time_window=None, Ceps=None):
    """
    Given a Pandas df with a lat and lon, this function will return another df with the results of a clustering algo.
    Currently limited to SciKit-Learn's DBSCAN and OPTICS, but more will be added
    :param Ceps:
    :param time_window:
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
            df_results = pd.DataFrame(results_dict)
        elif method == 'optics':
            # execute sklearn's OPTICS
            # 5km in radians is max eps
            optics = OPTICS(max_eps=eps_km / 6371, min_samples=min_samp, metric='euclidean', cluster_method='xi',
                            algorithm='kd_tree', n_jobs=1)
            optics.fit(X)
            results_dict = {'id': x_id, 'clust_id': optics.labels_, 'time': df['time'].values,
                            'lat': df['lat'].values, 'lon': df['lon'].values}
            # gather the output as a dataframe
            df_results = pd.DataFrame(results_dict)
        elif method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=True,
                                        gen_min_span_tree=False, leaf_size=40,
                                        metric='euclidean', min_cluster_size=5, min_samples=min_samp)
            clusterer.fit(X)
            results_dict = {'id': x_id, 'clust_id': clusterer.labels_, 'time': df['time'].values,
                            'lat': df['lat'].values, 'lon': df['lon'].values}
            # gather the output as a dataframe
            df_results = pd.DataFrame(results_dict)
        elif method == 'stdbscan':
            # execute ST_DBSCAN. eps1 in km, eps2 in minutes
            df_results = stdbscan.ST_DBSCAN(df=df, spatial_threshold=eps_km, temporal_threshold=time_window,
                                            min_neighbors=min_samp)
        elif method == 'tdbscan':
            df_results = tdbscan.T_DBSCAN(df, Ceps, eps_km, min_samp)
        else:
            print("Error.  Method must be 'dbscan', 'optics', 'hdbscan', or stdbscan'.")
            return None
    except Exception as e:
        print('UID error in clustering.')
        print(e)
        return None
    # drop all -1 clust_id, which are all points not in clusters
    df_results = df_results[df_results['clust_id'] != -1]
    return df_results


def calc_centers(df_results, clust_id_value='clust_id'):
    """This function finds the center of a cluster from dbscan results,
    and finds the average distance for each cluster point from its cluster center, as well as the min and max times.
    Returns a df."""
    # make a new df from the df_results grouped by cluster id
    # with an aggregation for min/max/count of times and the mean for lat and long
    df_centers = (df_results.groupby([clust_id_value])
                  .agg({'time': [min, max, 'count'],
                        'lat': 'mean',
                        'lon': 'mean'})
                  .reset_index(drop=False))
    df_centers.columns = ['clust_id', 'time_min', 'time_max', 'total_clust_count', 'average_lat', 'average_lon']
    # find the average distance from the centerpoint
    # We'll calculate this by finding all of the distances between each point in
    # df_results and the center of the cluster.  We'll then take the min and the mean.
    haver_list = []
    for i in df_centers[clust_id_value]:
        X = (np.radians(df_results[df_results[clust_id_value] == i]
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
        pp = folium.Html(f"Cluster: {row.clust_id} \n Count: {row.total_clust_count}\n" +
                          f"Average Dist from center {round(row.average_dist_from_center, 2)}\n" +
                          f"Min Time: {row.time_min}\n Max Time: {row.time_max}, Time Diff: {row.time_diff}")
        popup = folium.Popup(pp, max_width=150)
        folium.Marker(location=[row.average_lat, row.average_lon],
                      popup=popup).add_to(m)
    print(f'Plotted {len(df_centers)} total clusters.')
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
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    r = 2 * np.arcsin(np.sqrt(a))
    if unit =='mile':
        return 3958.748 * r
    if unit =='km':
        return 6371 * r
    if unit =='nm':
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

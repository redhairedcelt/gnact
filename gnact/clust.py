# utility modules
import pandas as pd
import numpy as np
import folium
import datetime

import gnact
from importlib import reload
reload(gnact)

# sklearn tools
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import haversine_distances


def get_uid_posits(uid, engine_pg, start_time='2017-01-01 00:00:00', end_time='2017-02-01 00:00:00'):
    """
    Given a UID and time range, this function will return all points in the database with the id ordered by time.
    :param uid: str
    :param engine_pg: SQL Alchemy Engine
    :param start_time: str formatted for PostGreSQL dates
    :param end_time: str formatted for PostGreSQL dates
    :return: a Pandas df with the id, time, lat, lon
    """
    read_sql = f"""SELECT id, lat, lon
                FROM uid_positions
                WHERE uid = '{uid[0]}'
                AND time between '{start_time}' and '{end_time}'
                ORDER by time"""
    df_posits = pd.read_sql_query(read_sql, con=engine_pg)
    return df_posits


def get_clusters(df, eps_km, min_samp, method):
    """
    Given a Pandas df with a lat and lon, this function will return another df with the results of a clustering algo.
    Currently limited to SciKit-Learn's DBSCAN and OPTICS, but more will be added
    :param df: a df with a unique id, lat and lon in decimal degrees
    :param eps_km: The epsilon (or max_epsilon_ value used in the clustering algo.
    :param min_samp: The minimum samples for a cluster to form
    :param method: The clustering method used
    :return: a Pandas df with the id, lat, lon, and cluster id from the algo.
    """
    # format data for dbscan
    X = (np.radians(df.loc[:, ['lon', 'lat']].values))
    x_id = df.loc[:, 'id'].values
    try:
        if method == 'dbscan':
            # execute sklearn's DBSCAN
            dbscan = DBSCAN(eps=eps_km/6371, min_samples=min_samp, algorithm='ball_tree',
                            metric='haversine', n_jobs=1)
            dbscan.fit(X)
            results_dict = {'id': x_id, 'clust_id': dbscan.labels_, 'lat': df['lat'], 'lon': df['lon']}
        if method == 'optics':
            # execute sklearn's OPTICS
            # 5km in radians is max eps
            optics = OPTICS(max_eps=eps_km/6371, min_samples=min_samp, metric='euclidean', cluster_method='xi',
                            algorithm='kd_tree', n_jobs=1)
            optics.fit(X)
            results_dict = {'id': x_id, 'clust_id': optics.labels_, 'lat': df['lat'], 'lon': df['lon']}
        if method not in ['optics', 'dbscan']:
            print("Error.  Method must be 'dbscan' or 'optics'.")
            return None
    except Exception as e:
        print('UID error in clustering.')
        print(e)
        return None
    # gather the output as a dataframe
    df_results = pd.DataFrame(results_dict)
    # drop all -1 clust_id, which are all points not in clusters
    df_results = df_results[df_results['clust_id'] != -1]
    return df_results

def pooled_clustering(uid, eps_km, min_samp, method, print_verbose=False):
    iteration_start = datetime.datetime.now()
    dest_column = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
    temp_table_name = f'temp_{str(uid[0])}'

    engine_pg = gnact.utils.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
    conn_pg = gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pg = conn_pg.cursor()

    df_posits = get_uid_posits(uid, engine_pg)
    df_results = get_clusters(df_posits, eps_km=eps_km, min_samp=min_samp, method=method)
    df_results = df_results[['id', 'clust_id']]
    try:
        # write results to database in a temp table with the uid in the name
        sql_drop_table = f"""DROP TABLE IF EXISTS {temp_table_name};"""
        c_pg.execute(sql_drop_table)
        conn_pg.commit()
        sql_create_table = f"""CREATE TABLE {temp_table_name}
                           (id int, 
                           clust_id int);"""
        c_pg.execute(sql_create_table)
        conn_pg.commit()
        df_results.to_sql(name=temp_table_name, con=engine_pg,
                          if_exists='append', method='multi', index=False)
        # take the clust_ids from the temp table and insert them into the temp table
        sql_update = f"UPDATE clustering_results AS c " \
                     f"SET {dest_column} = clust_id " \
                     f"FROM {temp_table_name} AS t WHERE t.id = c.id"
        c_pg.execute(sql_update)
        conn_pg.commit()
    except Exception as e:
        print(f'UID {uid[0]} error in writing clustering results to the database.')
        print(e)
    # delete the temp table
    c_pg.execute(sql_drop_table)
    conn_pg.commit()
    c_pg.close()
    # close the connections
    engine_pg.dispose()
    conn_pg.close()
    if print_verbose == True:
        print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
        uids_completed = add_to_uid_tracker(uid, conn_pg)
        percentage = (uids_completed / len(uid_list)) * 100
        print(f'Approximately {round(percentage, 3)} complete this run.')


def calc_centers(df_results, clust_id_value='clust_id'):
    """This function finds the center of a cluster from dbscan results,
    and finds the average distance for each cluster point from its cluster center.
    Returns a df."""
    # make a new df from the df_results grouped by cluster id
    # with the mean for lat and long
    df_centers = (df_results[[clust_id_value, 'lat', 'lon']]
                  .groupby(clust_id_value)
                  .mean()
                  .rename({'lat': 'average_lat', 'lon': 'average_lon'}, axis=1)
                  .reset_index())
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
    # create "total cluster count" column through groupby
    clust_size = (df_results[['lat', clust_id_value]]
                  .groupby(clust_id_value)
                  .count()
                  .reset_index()
                  .rename({'lat': 'total_clust_count'}, axis=1))
    # merge results back to df_Centers
    df_centers = pd.merge(df_centers, clust_size, how='left', on=clust_id_value)
    return df_centers


def plot_clusters(uid, eps_km, min_samp, method, pg_engine):
    df_posits = gnact.clust.get_uid_posits(uid, pg_engine, end_time='2018-01-01')

    # plot the track of the ship
    m = folium.Map(location=[df_posits.lat.median(), df_posits.lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    points = list(zip(df_posits.lat, df_posits.lon))
    folium.PolyLine(points).add_to(m)

    # plot the clusters
    df_results = gnact.clust.get_clusters(df_posits, eps_km, min_samp, method)
    df_centers = gnact.clust.calc_centers(df_results)
    for row in df_centers.itertuples():
        folium.Marker(location=[row.average_lat, row.average_lon],
                      popup=[f"Cluster: {row.clust_id} \n"
                             f"Count: {row.total_clust_count}\n"
                             f"Average Dist from center {round(row.average_dist_from_center,2)}"]
                      ).add_to(m)
    print(f'{method.upper()} for UID {uid[0]} with {eps_km} km epsilon '
          f'and {min_samp} minimum samples '
          f'found {len(df_centers)} total clusters.')
    return m


def postgres_dbscan_reworked(uid, eps_km, min_samp):
    """
    A function to conduct dbscan on the server for a global eps and min_samples value.
    Optimized for multiprocessing.
    :param min_samp:
    :param eps:
    :param uid:
    :return:
    """
    # iteration_start = datetime.datetime.now()
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
    uids_completed = gnact.utils.add_to_uid_tracker(uid, conn_pg)
    conn_pg.close()

    print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
    percentage = (uids_completed / len(uid_list)) * 100
    print(f'Approximately {round(percentage, 3)} complete.')
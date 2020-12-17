import pandas as pd
import folium


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


def plot_stops(df_stops, df_posits):
    # build the map
    m = folium.Map(location=[df_stops.lat.median(), df_stops.lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    # plot the stops
    for row in df_stops.itertuples():
        popup = folium.Popup(f"Site ID: {row.node}<BR>Site Name: {row.site_name}"
                             f"<BR>Count: {row.position_count}"
                             f"<BR>Min Time:  {row.arrival_time}"
                             f"<BR>Time Diff: {row.time_diff}", max_width=220)
        folium.Marker(location=[row.lat, row.lon], icon=folium.Icon(color='gray'),
                      popup=popup).add_to(m)
    points = list(zip(df_posits.lat, df_posits.lon))
    folium.PolyLine(points).add_to(m)
    print(f'Plotted {len(df_stops)} total sites.')
    return m


def plot_stats(df_stats, df_stops, df_posits):
    m = folium.Map(location=[df_stats.average_lat.median(), df_stats.average_lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    points = list(zip(df_posits.lat, df_posits.lon))
    folium.PolyLine(points).add_to(m)

    # plot the stops
    for row in df_stops.itertuples():
        popup = folium.Popup(f"Site ID: {row.node}<BR>Site Name: {row.site_name}"
                             f"<BR>Count: {row.position_count}"
                             f"<BR>Min Time:  {row.arrival_time}"
                             f"<BR>Time Diff: {row.time_diff}", max_width=220)
        folium.Marker(location=[row.lat, row.lon], icon=folium.Icon(color='gray'),
                      popup=popup).add_to(m)

    # plot the false positive, false negatives, and true positives
    for idx, row in df_stats.iterrows():
        if row['results'] == 'False Positive':
            popup = folium.Popup(f"False Positive"
                                 f"<BR>Nearest Site ID: {row.nearest_site_id}" 
                                 f"<BR>Dist to nearest site (km): {row.dist_km}"
                                 f"<BR>Count: {row.total_clust_count}"
                                 f"<BR>Cluster Duration: {row.time_diff}",
                                 max_width=220)
            folium.Marker(location=[row.average_lat, row.average_lon], icon=folium.Icon(color='orange'),
                          popup=popup).add_to(m)
        elif row['results'] == 'False Negative':
            popup = folium.Popup(f"False Negative"
                                 f"<BR>Site_id: {row.site_id}"
                                 f"<BR>Site_name: {row.site_name}"
                                 f"<BR>Count: {row.position_count}"
                                 f"<BR>Cluster Duration: {row.time_diff}",
                                 max_width=220)
            folium.Marker(location=[row.lat, row.lon], icon=folium.Icon(color='red'),
                          popup=popup).add_to(m)
        elif row['results'] == 'True Positive':
            popup = folium.Popup(f"True Positive"
                                 f"<BR>Dist to nearest site (km): {row.dist_km}"
                                 f"<BR>Nearest Site ID: {row.nearest_site_id}"
                                 f"<BR>Count: {row.total_clust_count}"
                                 f"<BR>Cluster Duration: {row.time_diff}",
                                 max_width=220)
            folium.Marker(location=[row.average_lat, row.average_lon], icon=folium.Icon(color='green'),
                          popup=popup).add_to(m)

    print(f'Plotted {len(df_stats)} predicted clusters and {len(df_stops)} ground truth clusters.')
    return m

from gnact.clust import calc_stats, get_df_stats
def analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km):
    """
    Roll up function to get metrics and produce plots for presentations and visualizations
    :param df_posits:
    :param df_clusts:
    :param df_stops:
    :param df_stats:
    :param dist_threshold_km:
    :return:
    """
    # get precision, recall, f1 metrics
    print(calc_stats(df_clusts, df_stops, dist_threshold_km))
    # get the stats df
    df_stats = get_df_stats(df_clusts, df_stops, dist_threshold_km)
    # plot the results
    m = plot_stats(df_stats, df_stops, df_posits)
    return m
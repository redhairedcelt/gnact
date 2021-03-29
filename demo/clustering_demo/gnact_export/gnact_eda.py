#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import folium
import skmob

from gnact import utils, clust, plotting, network

import warnings 
warnings.filterwarnings("ignore")
# warnings in scikit-learn optics in reachability graph with dividing by zero
# and in skmob's use of pandas' soon to be depreciated use of pd.datetime


# In[2]:


import os
os.chdir('/Users/patrickmaus/PycharmProjects/gnact/demo/clustering_demo')


# ## Demonstration of Plotting and Density Based Clustering Methods Using GNACT

# ### Plot of Raw Data

# In[3]:


os.chdir('/Users/patrickmaus/PycharmProjects/gnact/demo/clustering_demo')
# load the position data
df_posits = pd.read_csv('df_posits_636016432.csv', parse_dates=['time'])
df_posits['uid'] = '636016432'
df_posits.head()


# In[4]:


os.chdir('/Users/patrickmaus/PycharmProjects/AIS_project/data')
df_posits = pd.read_csv('uid_positions.csv', parse_dates=['time'], usecols=['id', 'lat', 'lon', 'time', 'uid'])
df_posits.info()


# In[6]:


# make a list of all uids
list_uids = df_posits['uid'].unique()
sample_uids = list_uids[:10]


# In[7]:


#Now from all the data, select just the following uids
df_posits_sample = df_posits[df_posits['uid'].isin(sample_uids[np.array([2,4,5,6,9])])]
df_posits_sample.groupby('uid').agg('count')


# In[8]:


def plot_posits(df_posits):
    # plot with Folium
    m = folium.Map(location=[df_posits.lat.median(), df_posits.lon.median()],
                   zoom_start=4, tiles='OpenStreetMap')
    points = list(zip(df_posits.lat, df_posits.lon))
    folium.PolyLine(points).add_to(m)
    return m


# In[21]:


plot_posits(df_posits_sample[df_posits_sample['uid'] == 316029000])


# ### Use the World Port Index as Reference Sites
# This list is not exhaustive, but its a good example of a real-world reference dataset where most of the major sites are known but many smaller sites are not.

# In[9]:


os.chdir('/Users/patrickmaus/PycharmProjects/gnact/demo/clustering_demo')
df_sites = pd.read_csv('wpi_clean.csv')
df_sites.head()


# ### Calculate Nearest Site for Each Position and Create a List of Stops
# 
# The first step in static trip segmentation is to calculate the nearest known site for each position.  Here we will apply the approach against a DataFrame using a custom function.  
# 
# We will use a distance threshold of 5km and loiter time of 6 hours (360 minutes).  This means a cargo ship must spend at least 6 hours within 5km of a known port to be counted as making a stop.  

# In[14]:


df_nn = clust.calc_nn(df_posits_sample, df_sites)
df_nn.head()


# ### Finding Activity at Known Sites

# Now we can apply static trip segmentation against this data and find activity at only known sites that meet the min loiter time within the max distanct threshold.  Each time the uid is near a known site within those thresholds, a new cluster is formed.

# In[15]:


dist_threshold_km = 5
loiter_time_mins = 360

# determine the "ground truth" for this sample
df_stops = network.calc_static_seg(df_posits_sample, df_nn, df_sites, 
                                   dist_threshold_km, loiter_time_mins)
df_stops.head()


# ### Review Activity at Known Sites

# In[12]:


df_stops.head()


# In[ ]:


df_stops.groupby('site_name').agg('count').iloc[:,0]


# In[ ]:


df_stops.drop(['lat','lon','region','destination', 'position_count', 'node'], axis=1).head(5)


# In[ ]:


#df_stops[['node', ]] ['Source', 'source_depart', 'Target', 'target_arrival']


# In[ ]:


#network.plot_uid('636016432', df_posits)


# ## Apply Other Clustering Algorithms and Compare to Activity at Known Sites

# ### Plot of DBSCAN with Low Parameters

# In this scenario, we choose parameters with DBSCAN that were far too low, causing numerous false positive clusters.  By zooming in to the activity off the coasts of Delaware, New Jersey, and Virginia, we can see numerous clusters where the ship is conducting normal sailing operations.  This suggests we need higher thresholds.

# In[ ]:


# execute clustering algo with hyperparameters
df_clusts = clust.calc_clusts(df_posits, eps_km=5, eps_time=360, method='dynamic')
df_centers = clust.calc_centers(df_clusts)
plotting.plot_clusters(df_posits, df_centers)


# In[ ]:


df_centers[['nearest_site_id', 'dist_km']] = clust.calc_nn(df_centers, df_sites, lat='average_lat', lon='average_lon', id='clust_id').drop('id', axis=1)
df_centers.head()


# In[ ]:


df_unid_sites = df_centers[df_centers['dist_km'] > 5]
plotting.plot_clusters(df_posits, df_unid_sites)


# In[ ]:


df_unid_sites


# In[ ]:


df_unid_clusts = clust.calc_clusts(unid_sites, lat='average_lat', lon='average_lon', id='clust_id', time='time_min',
                                   eps_km=5, min_samp=2, method='dbscan')
df_unid_clusts


# In[ ]:


df_unid_centers = clust.calc_centers(df_unid_clusts)
df_unid_centers


# In[ ]:



plotting.plot_clusters(df_posits, df_unid_centers)


# ## DBSCAN With Speed Filter

# We can try to eliminate false positives by applying subject matter insight and establish a maximum speed.  A cluster therefroe could only occur if enough points below the speed threshold clustered together.

# In[ ]:


# enhance the df with speed, course and additional trajecotry info
df_traj_enhanced = utils.traj_enhance_df(df_posits)
# filter down to points below certain speed
df_slow_posits = df_traj_enhanced.loc[df_traj_enhanced['speed_kts'] < 1].copy()
# cluster only the slow points
df_clusts = clust.calc_clusts(df_slow_posits, eps_km=2, min_samp=50, method='dbscan')

plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ### Experimentation with Double Clustering Approaches

# By clustering raw positions to centerpoints, and then clustering those centerpoints, we can use lower thresholds to cluster positions by ships.  Then we can cluster the resulting centerpoints with a minimum sample size above a certain noise threshold.

# In[ ]:


from sklearn.cluster import DBSCAN

df_clusts = clust.calc_clusts(df_posits, eps_km=3, min_samp=2000, method='dbscan')
#df_clusts = clust.calc_clusts(df_posits, eps_km=5, min_samp=200, method='optics')

# need new unique cluster ids across each uid.
clust_count = 0
# will hold results of second round temporal clustering
df_second_round = pd.DataFrame()

# begin iteration.  Look at each cluster in turn from first round results and cluster across time
clusters = df_clusts['clust_id'].unique()
for c in clusters:
    df_c = df_clusts[df_clusts['clust_id'] == c]
    X = ((df_c['time'].astype('int').values) / ((10**9)*60)).reshape(-1,1) #converts time to mins
    x_id = df_c.loc[:, 'id'].astype('int').values
    # cluster again using DBSCAN with a temportal epsilon (minutes) in one dimension
    dbscan = DBSCAN(eps=600, min_samples=2, algorithm='kd_tree',
                    metric='euclidean', n_jobs=1)
    dbscan.fit(X)
    results2_dict = {'id': x_id, 'clust_id': dbscan.labels_}
    # gather the output as a dataframe
    df_clusts2 = pd.DataFrame(results2_dict)
    df_clusts2 = df_clusts2.loc[df_clusts2['clust_id'] != -1].copy()
    clusters2 = df_clusts2['clust_id'].unique()
    for c2 in clusters2:
        df_c2 = df_clusts2.loc[df_clusts2['clust_id'] == int(c2)].copy() # need int rather than numpy.int64
        # need to assign a new cluster id
        df_c2['clust_id'] = clust_count
        # add each iteration result to the df_clusts2 DataFrame
        df_second_round = df_second_round.append(df_c2)
        # iterate the cluster count
        clust_count +=1

df_second_results = pd.merge(df_second_round, df_clusts.drop('clust_id', axis=1), how='left', left_on='id', right_on='id')


plotting.analyze_clusters(df_posits, df_second_results, df_stops, dist_threshold_km)


# ## Integration of Scikit-Mobility and Dynamic Segmentation of Trips
# 
# Scikit-Mobility provides additional plotting and packaging tools to parse a geospatial dataset into "trips" based on "stops" along each UID's path.

# In[ ]:


import skmob


# In[ ]:


#df_posits = clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')
#df_posits['uid'] = '636016432'
tdf = skmob.TrajDataFrame(df_posits, latitude='lat', longitude='lon', datetime='time')
tdf.plot_trajectory(tiles='OpenStreetMap', zoom=4)


# ## SKMOB has a "stop detection" algorithm to identify stops based on distance, duration, max speed, and includes an escape clause when there is a gap in data.

# In[ ]:


from skmob.preprocessing import detection
stdf = detection.stops(tdf, minutes_for_a_stop=360, spatial_radius_km=2, leaving_time=True, 
                       no_data_for_minutes=360, min_speed_kmh=70)

print('Points of the original trajectory:\t%s'%len(tdf))
print('Points of stops:\t\t\t%s'%len(stdf))

m = stdf.plot_trajectory(max_users=1, start_end_markers=False, tiles='OpenStreetMap', zoom=4)
stdf.plot_stops(max_users=1, map_f=m)


# In[ ]:


stdf.head()


# ### We can then use DBSCAN to cluster the stops to find "destinations" frequently visited.

# In[ ]:


from skmob.preprocessing import detection, clustering
cstdf = clustering.cluster(stdf, cluster_radius_km=1, min_samples=1)
cstdf.head()


# In[ ]:


m = cstdf.plot_trajectory(max_users=1, start_end_markers=False, tiles='OpenStreetMap', zoom=4)
cstdf.plot_stops(max_users=1, map_f=m)


# ### Now we can add it to our existing function wrappers and determine the statistics

# In[ ]:


df_clusts = clust.calc_clusts(df_posits, eps_km=3, eps_time=360, method='dynamic')

plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


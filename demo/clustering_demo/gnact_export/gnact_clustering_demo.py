#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import folium

get_ipython().system('conda develop PycharmProjects')
from gnact import utils, clust, plotting, network

import warnings 
warnings.filterwarnings("ignore")
# warnings in scikit-learn optics in reachability graph with dividing by zero
# and in skmob's use of pandas' soon to be depreciated use of pd.datetime


# ## Demonstration of Plotting and Density Based Clustering Methods Using GNACT

# ### Plot of Raw Data

# In[2]:


# load the position data
df_posits = pd.read_csv('df_posits_636016432.csv', parse_dates=['time'])
df_posits.head()


# In[3]:


# plot with Folium
m = folium.Map(location=[df_posits.lat.median(), df_posits.lon.median()],
               zoom_start=4, tiles='OpenStreetMap')
points = list(zip(df_posits.lat, df_posits.lon))
folium.PolyLine(points).add_to(m)
m


# ### Use the World Port Index as Reference Sites
# This list is not exhaustive, but its a good example of a real-world reference dataset where most of the major sites are known but many smaller sites are not.

# In[4]:


df_sites = pd.read_csv('wpi_clean.csv')
df_sites.head()


# ## Finding Stops/Clusters at Known Ports Using Static Trip Segmentation
# We will generate clusters for all positions that spend a minimum amount of time within a certain distance of any known site.  This is known as static trip segmentation in the literature, and will have the lowest false positive rate of any method because all positions clustered must be within a certain distance of a known port.
# 
# I previously developed a static trip segmentation methodology during a Directed Stuides on Network Analysis.  By applying this methodology against a geospatial dataset with a known set of ports, I generated a network map  with each stop repersenting a node and travel between ports as edges.

# ### Calculate Nearest Site for Each Position and Create a List of Stops
# 
# The first step in static trip segmentation is to calculate the nearest known site for each position.  Here we will apply the approach against a DataFrame using a custom function.  
# 
# We will use a distance threshold of 5km and loiter time of 6 hours (360 minutes).  This means a cargo ship must spend at least 6 hours within 5km of a known port to be counted as making a stop.  

# In[5]:


dist_threshold_km = 5
loiter_time_mins = 360

df_nn = clust.calc_nn(df_posits, df_sites)
df_nn.head()


# ### Building "Ground Truth" From Static Trip Segmentation Results

# Now we can apply static trip segmentation against this data and use it as our "ground truth".
# 

# In[6]:


# determine the "ground truth" for this sample
df_stops = network.calc_static_seg(df_posits, df_nn, df_sites, 
                                   dist_threshold_km, loiter_time_mins)
# plot results 
plotting.plot_stops(df_stops, df_posits)


# In[7]:


df_stops.head()


# After plotting the ports visited, its clear that there was activity near Savannah that was not recorded.  Turns out the port near Savannah where the MSC ARUSHI stopped was not in the database.  We can add the port manually, re-generate our nearest_neighbor df, and recompute our statistics.

# In[8]:


# manually create the site
savannah_site = {'site_id':3, 'site_name': 'SAVANNAH_MANUAL_1', 'lat': 32.121167, 'lon':-81.130085, 
               'region':'East_Coast'}
# add the site to the df_sites
df_sites = df_sites.append(savannah_site, ignore_index=True) # add savannah
# recompute the nearest neighbors
df_nn = clust.calc_nn(df_posits, df_sites)


# In[9]:


# determine the "ground truth" for this sample
df_stops = network.calc_static_seg(df_posits, df_nn, df_sites, 
                                   dist_threshold_km, loiter_time_mins)
# plot results 
plotting.plot_stops(df_stops, df_posits)


# ### Review "Ground Truth"

# Now our new Savannah Port is correctly identifed as a ground truth cluster.  We can next use our code to generate clusters and compare them to the ground truth.  We can see from aggregating site_names in our df_stops DataFrame, there were 8 stops in Boston, 2 in Freeport, 11 in Gloucester, 8 in Newark, and 2 at our manually added Savannah site.

# In[10]:


df_stops.groupby('site_name').agg('count').iloc[:,0]


# In[11]:


df_stops.drop(['lat','lon','region','destination', 'position_count', 'node'], axis=1).head(5)


# ## Apply Algoithms, Get Clusters, Compare to Ground Truth, and Plot Results

# With our ground truth and tools to generate clusters using different methods and various hyperparameters, we can now plot and compare a number of different clustering methods against our ground truth.  Let's look at a few examples.

# ### Plot of DBSCAN with Low Parameters

# In this scenario, we choose parameters iwth DBSCAN that were far too low, causing numerous false positive clusters.  By zooming in to the activity off the coasts of Delaware, New Jersey, and Virginia, we can see numerous clusters where the ship is conducting normal sailing operations.  This suggests we need higher thresholds.

# In[12]:


# execute clustering algo with hyperparameters
df_clusts = clust.calc_clusts(df_posits, eps_km=1, min_samp=50, method='dbscan')
plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ### Plot of DBSCAN with High Parameters

# In this case, the parameters are too high, and only sites with very high levels of activity across the time period are identified as clusters.  It is also a good example of why a high precision score is not always idicitive of an effective clustering method.

# In[13]:


# execute clustering algo with hyperparameters
df_clusts = clust.calc_clusts(df_posits, eps_km=3, min_samp=2000, method='dbscan')
plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ### DBSCAN with Tuned Parameters

# In this example, well-tuned parameters successfully identify numerous clusters, but still have several false positives.  DBSCAN is also penalized heavily in that clusters in the same area tend to be grouped together with a low enough epsilon value. Therefore, multiple visits to the same site are only counted as one correct answer.  

# In[14]:


df_clusts = clust.calc_clusts(df_posits, eps_km=1, min_samp=250, method='dbscan')
plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ## DBSCAN With Speed Filter

# We can try to eliminate false positives by applying subject matter insight and establish a maximum speed.  A cluster therefroe could only occur if enough points below the speed threshold clustered together.

# In[15]:


# enhance the df with speed, course and additional trajecotry info
df_traj_enhanced = utils.traj_enhance_df(df_posits)
# filter down to points below certain speed
df_slow_posits = df_traj_enhanced.loc[df_traj_enhanced['speed_kts'] < 1].copy()
# cluster only the slow points
df_clusts = clust.calc_clusts(df_slow_posits, eps_km=2, min_samp=50, method='dbscan')

plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ### OPTICS with Tuned Parameters

# OPTICS uses a more finely-tuned parsing approach for finding clusters, and therefore can separate clusters very close to each other in space into unique clusters.

# In[31]:


df_clusts = clust.calc_clusts(df_posits, eps_km=5, min_samp=200, method='optics')
plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ## ST_DBSCAN with Tuned Parameters

# ST_DBSCAN adds an additional epsilon distance for time.  A cluster must be within the spatial and temporal thresholds for a cluster to form in this method.  The non-optimized approach I implemented takes a long time to process, so for this demo we will load a csv.

# In[32]:


# Processing is upwards of an hour...
#df_clusts = stdbscan.ST_DBSCAN(df_posits, spatial_threshold=3, temporal_threshold=600, min_neighbors=100)
#df_clusts.to_csv('st_dbscan_results.csv', index=False)


# In[33]:


df_clusts = pd.read_csv('st_dbscan_results.csv', parse_dates=['time'])
df_clusts = df_clusts[df_clusts['clust_id'] != -1]

plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# ### Experimentation with Double Clustering Approaches

# By clustering raw positions to centerpoints, and then clustering those centerpoints, we can use lower thresholds to cluster positions by ships.  Then we can cluster the resulting centerpoints with a minimum sample size above a certain noise threshold.

# In[34]:


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

# In[4]:


import skmob


# In[5]:


#df_posits = clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')
#df_posits['uid'] = '636016432'
tdf = skmob.TrajDataFrame(df_posits, latitude='lat', longitude='lon', datetime='time')
tdf.plot_trajectory(tiles='OpenStreetMap', zoom=4)


# ## SKMOB has a "stop detection" algorithm to identify stops based on distance, duration, max speed, and includes an escape clause when there is a gap in data.

# In[6]:


from skmob.preprocessing import detection
stdf = detection.stops(tdf, minutes_for_a_stop=360, spatial_radius_km=2, leaving_time=True, 
                       no_data_for_minutes=360, min_speed_kmh=70)

print('Points of the original trajectory:\t%s'%len(tdf))
print('Points of stops:\t\t\t%s'%len(stdf))

m = stdf.plot_trajectory(max_users=1, start_end_markers=False, tiles='OpenStreetMap', zoom=4)
stdf.plot_stops(max_users=1, map_f=m)


# In[24]:


stdf


# ### We can then use DBSCAN to cluster the stops to find "destinations" frequently visited.

# In[27]:


from skmob.preprocessing import detection, clustering
cstdf = clustering.cluster(stdf, cluster_radius_km=10, min_samples=2)
cstdf


# In[40]:


m = cstdf.plot_trajectory(max_users=1, start_end_markers=False, tiles='OpenStreetMap', zoom=4)
cstdf.plot_stops(max_users=1, map_f=m)


# ### Now we can add it to our existing function wrappers and determine the statistics

# In[28]:


df_clusts = clust.calc_clusts(df_posits, eps_km=3, eps_time=360, method='dynamic')

plotting.analyze_clusters(df_posits, df_clusts, df_stops, dist_threshold_km)


# In[ ]:





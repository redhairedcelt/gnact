#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import folium
import skmob
import os

from gnact import utils, clust, plotting, network


# In[9]:


os.chdir('/Users/patrickmaus/PycharmProjects/AIS_project/data')
df_posits = pd.read_csv('uid_positions.csv', parse_dates=['time'], usecols=['id', 'lat', 'lon', 'time', 'uid'])
list_uids = df_posits['uid'].unique()
sample_uids = list_uids[:10]
df_posits.info()


# In[10]:


df_posits_sample = df_posits[df_posits['uid'].isin(sample_uids[np.array([2,4,5,6,9])])]
df_posits_sample.groupby('uid').agg('count')


# In[11]:


#df_posits = clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')
#df_posits['uid'] = '636016432'
tdf = skmob.TrajDataFrame(df_posits_sample, latitude='lat', longitude='lon', datetime='time')
tdf.plot_trajectory(tiles='OpenStreetMap', zoom=4)


# In[12]:


tdf.head()


# In[14]:


from skmob.preprocessing import detection, clustering
stdf = detection.stops(tdf, minutes_for_a_stop=360, spatial_radius_km=2, leaving_time=True, 
                       no_data_for_minutes=10000)

print('Points of the original trajectory:\t%s'%len(tdf))
print('Points of stops:\t\t\t%s'%len(stdf))


# In[15]:


m = stdf.plot_trajectory(max_users=10, start_end_markers=True, tiles='OpenStreetMap', zoom=4)
stdf.plot_stops(max_users=10, map_f=m)


# In[16]:


stdf.head()


# In[17]:


# look at a single UID


# In[18]:


m = stdf[stdf['uid']==366938780].plot_trajectory(start_end_markers=True, 
                          tiles='OpenStreetMap', zoom=4)
stdf[stdf['uid']==366938780].plot_stops(max_users=10, map_f=m)


# In[19]:


cstdf = clustering.cluster(stdf, cluster_radius_km=1, min_samples=1)

print(len(cstdf.cluster.unique()))
cstdf.head()


# In[20]:


cstdf.plot_diary(user=366938780, legend=True)


# In[21]:


df_posits_sample.info()


# In[ ]:


# execute clustering algo with hyperparameters
df_clusts = clust.calc_clusts(df_posits_sample[df_posits_sample['uid']==366938780], 
                              eps_km=1, min_samp=10, method='optics')
df_centers = clust.calc_centers(df_clusts)
plotting.plot_clusters(df_posits, df_centers)


# In[ ]:


df_posits_sample[df_posits_sample['uid']==366938780]


# In[ ]:


# network plot


# In[ ]:


cstdf


# In[ ]:


# sample


# In[ ]:


import skmob
from skmob.preprocessing import detection, clustering
import pandas as pd
# read the trajectory data (GeoLife, Beijing, China)
url = skmob.utils.constants.GEOLIFE_SAMPLE
df = pd.read_csv(url, sep=',', compression='gzip')
tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
print(tdf.head())


# In[ ]:


# detect stops
stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
print(stdf.head())


# In[ ]:


m = stdf.plot_trajectory(max_users=10, start_end_markers=True, tiles='OpenStreetMap', zoom=4)
stdf.plot_stops(max_users=10, map_f=m)


# In[ ]:


# cluster stops
cstdf = clustering.cluster(stdf, cluster_radius_km=1, min_samples=1)
print(cstdf)
print(len(cstdf.cluster.unique()))


# In[ ]:


# plot the diary of one individual
user = 1
start_datetime = pd.to_datetime('2008-10-23 030000')
end_datetime = pd.to_datetime('2008-10-30 030000')
ax = cstdf.plot_diary(user, start_datetime=start_datetime, end_datetime=end_datetime)


# In[ ]:





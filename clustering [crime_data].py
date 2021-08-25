#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


crime_data=pd.read_csv("\\Users\\piyus\\Downloads\\crime_data.csv")
crime_data.head()


# In[3]:


crime_data.shape


# In[4]:


data=crime_data.copy()
data=data.iloc[:,1:]
data.head(2)


# In[5]:


data.isna().sum()


# In[6]:


data.dtypes


# In[7]:


#scaling the data
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scale.fit(data.iloc[:,0:])
data.iloc[:,0:]=scale.transform(data.iloc[:,0:])

data.head()


# In[8]:


# elbow method to get best k in kmeans
k_rng=range(1,10)
sse= []
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(data)
    sse.append(km.inertia_)


# In[9]:


sse


# In[10]:


from matplotlib import pyplot as plt
plt.xlabel('k')
plt.ylabel('sum of square error')
plt.plot(k_rng,sse)


# # KMeans clustering

# In[11]:


model=KMeans(n_clusters=4)
y_pred=model.fit_predict(data)
y_pred


# In[12]:


data['Cluster']=y_pred
data.head()


# In[13]:


crime_data.groupby(data.Cluster).mean()


# In[14]:


data_df=data.drop('Cluster',axis=1)


# # DBSCAN

# In[15]:


model_=DBSCAN(eps=0.3,min_samples=5)
y_predict=model_.fit_predict(data_df)

y_predict


# In[16]:


data_df['clust']=y_predict
data_df.head()


# In[17]:


crime_data.groupby(data_df.clust).mean()


# # Hierarchical clustering 

# In[18]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import seaborn as sns


# In[19]:


#creating dendrogram
dendrogram=sch.dendrogram(sch.linkage(data,method='single'))


# In[20]:


HC=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')
y_hc=HC.fit_predict(data)
clusters=pd.DataFrame(y_hc,columns=['Clusters'])
clusters


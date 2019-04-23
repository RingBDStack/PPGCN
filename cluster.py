
# coding: utf-8

# In[98]:

from sklearn.cluster import spectral_clustering, dbscan, affinity_propagation
import numpy as np
import time

#start = time.time()

# In[5]:
'''
adjacentM = [
    [0, 5, 3, 8, 2, 9],
    [5, 0, 1, 7, 6, 5],
    [3, 1, 0, 4, 3, 1],
    [8, 7, 4, 0, 9, 2],
    [2, 6, 3, 9, 0, 7],
    [9, 5, 1, 2, 7, 0]
]
adjacentM = np.array(adjacentM)


# In[133]:

similarities = [
    [1.0, 0.5, 0.7, 0.2, 0.8, 0.1],
    [0.5, 1.0, 0.9, 0.3, 0.4, 0.5],
    [0.7, 0.9, 1.0, 0.6, 0.7, 0.9],
    [0.2, 0.3, 0.6, 1.0, 0.1, 0.8],
    [0.8, 0.4, 0.7, 0.1, 1.0, 0.3],
    [0.1, 0.5, 0.9, 0.8, 0.3, 1.0]
]
similarities = np.array(similarities)
'''
print('Loading similarities ...')
similarities = np.load('ring_adj_data.npy')#[:, 0:10000, 0:10000]
similarities = np.sum(similarities, axis = 0)/13
print('Reversing similarities to distance ...')
adjacentM = 1 - similarities

start = time.time()
print('Start clustering ...')
# In[26]:
'''
spectral_labels = spectral_clustering(adjacentM, n_clusters=2084, n_components=None, eigen_solver=None)
spectral_labels = np.array(spectral_labels)
np.save('spectral_event_labels.npy', spectral_labels)

spectral_labels = spectral_clustering(adjacentM, n_clusters=954, n_components=None, eigen_solver=None)
spectral_labels = np.array(spectral_labels)
np.save('spectral_story_labels.npy', spectral_labels)
'''
# In[27]:

#spectral_labels


# In[96]:

core_samples, dbscan_labels = dbscan(adjacentM, eps=0.1, min_samples=1, 
                                     metric='minkowski', metric_params=None, algorithm='auto', p=2, sample_weight=None)


# In[97]:

#dbscan_labels


# In[134]:

# not adjacentM, use similarities
#cluster_centers_indices, ap_labels = affinity_propagation(similarities, preference=None, 
#                                                          convergence_iter=15, max_iter=200, damping=0.5)


# In[135]:

#ap_labels = np.array(ap_labels)
#np.save('ap_labels.npy', ap_labels)
dbscan_labels = np.array(dbscan_labels)
np.save('dbscan_labels.npy', dbscan_labels)
end = time.time()
print('Time: %.4f minutes.'%((end-start)/60))

# In[ ]:





# coding: utf-8

# In[1]:

import numpy as np
import math


# In[11]:

def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


# In[18]:

#A = np.array([1,1,1,2,1,1,2,2,3,3])
#B = np.array([1,1,1,0,1,1,0,0,2,2])

A = np.load('ring_story_label.npy')[0:5000]
B = np.load('dbscan_labels.npy')
#B = np.load('spectral_event_labels.npy')
#B_ = np.load('spectral_story_labels.npy')
#B = np.load('dbscan_labels.npy')
C = np.load('ring_event_label.npy')[0:5000]

# In[19]:

print('Story NMI Calculating Result Is: ', NMI(A, B))
print('Event NMI Calculating Result Is: ', NMI(C, B))

# In[ ]:




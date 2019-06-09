
# coding: utf-8

# In[1]:

# import sys
import numpy as np
import time
import json
# In[ ]:

# raw_entity
# relations
# data
# kb
start_time = time.time()

with open('events.json', 'r') as json_file:
    events = json.load(json_file)

print('Build relations, data dict finished.')

# create single matrix
def M_single_step(s, d):
    src = list(s.keys())
    dst = list(d.keys())
    M_l = np.zeros((len(src), len(dst)))
    for i in range(len(src)):
        for j in range(len(dst)):
            if dst[j] in s[src[i]]:
                M_l[i][j] = 1
    return M_l

# create matrix according to meta-path
def cal_M(meta_path):
    for i in range(len(meta_path) - 1):
        with open(meta_path[i]+'.json', 'r') as json_file:
            src = json.load(json_file)
        with open(meta_path[i+1]+'.json', 'r') as json_file:
            dst = json.load(json_file)
        M_l = M_single_step(src, dst)
        print('Single step matrix generated.')
        if i == 0:
            M = M_l
        else:
            M = np.matmul(M, M_l)
        del M_l
        print('Single step multiply process finished.')
        del src, dst
    return M

# calculate sim between two nodes
def know_sim(M, i, j):
    broadness = M[i][i] + M[j][j]
    overlap = 2*M[i][j]
    if broadness == 0:
        return 0
    else:
        return overlap/broadness

# create similarity matrix according to meta-path
#data['entity'] = data['Name'] + data['Site'] + data['Institution'] + data['IP']
event_num = len(events)

# we show 3 meta-paths as examples.
meta_paths = [
    ['events', 'topics', 'keywords', 'topics', 'events'],
    ['events', 'entities', 'keywords', 'entities', 'events'],
    ['events', 'entities', 'topics', 'entities', 'events']
]

print('Start creating matrix ...')

for k in range(len(meta_paths)):
    path = meta_paths[k]
    M = cal_M(path)
    sim = np.zeros((event_num, event_num))
    print('path'+str(k)+' M calculation finished.')
    for i in range(event_num):
        sim[i][i] = 1
    for i in range(event_num - 1):
        for j in range(i+1, event_num):
            sim[i][j] = know_sim(M, i, j)
            sim[j][i] = sim[i][j]
    del M
    np.save('sim_'+str(path)+'.npy', sim)
    del sim
    print('path'+str(k)+' sim matrix finished.')    


# In[ ]:
with open('time.txt', 'a') as f:
    f.write('Create Similarity matrix takes %.3f minutes\n'%((time.time()-start_time)/60))






























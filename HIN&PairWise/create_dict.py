
# coding: utf-8

# In[1]:

import json
import os
import time


def events_dict(events, num_files):
    for i in range(num_files):
        events['event'+str(i+1000000)] = ['event'+str(i+1000000)]
    with open('events.json', 'w') as json_file:
        json.dump(events, json_file)
    return events


def keywords_dict(keywords, events, num_files, fp):
    for i in range(num_files):
        print(i)
        f = open(fp+'event'+str(i+18778+1000000)+'.txt', 'r')
        full = f.readlines()
        f.close()
        keys = full[0].strip().split(' ')
        events['event'+str(i+1000000)] += keys
        for k in keys:
            try:
                keywords[k].append('event'+str(i+1000000))
            except:
                keywords[k] = [k, 'event'+str(i+1000000)]
#        if i % 10000 == 0:
#            print('Saving dictionary ...')
#            with open('events.json', 'w') as json_file:
#                json.dump(events, json_file)
#            with open('keywords.json', 'w') as json_file:
#                json.dump(keywords, json_file)
#            print('Saved.')
        del full
        del keys
    with open('events.json', 'w') as json_file:
        json.dump(events, json_file)
    with open('keywords.json', 'w') as json_file:
        json.dump(keywords, json_file)


if __name__ == "__main__": 
    start_time = time.time()
    events = {}
    keywords = {}

    fp = 'news/'
#    fp = 'Data/'
    num_files = 9000000
    print('Totally ', num_files)
    print('Initializing events dictionary ...')
    events = events_dict(events, num_files)
    print('Initializing keywords dictionary and Adding keywords&events relations ...')
    keywords_dict(keywords, events, num_files, fp)

    print('Done!!!')
    with open('time.txt', 'a') as f:
        f.write('Initializing dict totally takes %.3f minutes\n'%((time.time()-start_time)/60))













# coding: utf-8

# In[1]:

import json
import time

def add_key_entity(entities, keywords, full_entities, full_keywords, desc):
    num_keywords = len(full_keywords)
    num_entities = len(full_entities)
    print('Totally ', num_keywords)
    for i in range(num_keywords):
        print(i)
        key = full_keywords[i]
        for j in range(num_entities):
            e = full_entities[j]
            try:
                if key in desc[e]:
                    keywords[key].append(e)
                    entities[e].append(key)
            except:
                continue
#        if i % 1000 == 0:
#            print('Saving dictionaries ...')
#            with open('keywords.json', 'w') as json_file:
#                json.dump(keywords, json_file)
#            with open('entities.json', 'w') as json_file:
#                json.dump(entities, json_file)
#            print('Saved.')
    for k in full_keywords:
        keywords[k] = list(set(keywords[k]))
    for e in full_entities:
        entities[e] = list(set(entities[e]))
    with open('keywords.json', 'w') as json_file:
        json.dump(keywords, json_file)
    with open('entities.json', 'w') as json_file:
        json.dump(entities, json_file)


if __name__ == "__main__":
    start_time = time.time()
    with open('keywords.json', 'r') as json_file:
        keywords = json.load(json_file)

    with open('entities.json', 'r') as json_file:
        entities = json.load(json_file)

    desc = {}
    for i in range(10):
        f = open('Desc/card'+str(i+1)+'.txt', 'r')
        desc_tmp = f.read()
        desc_tmp = eval(desc_tmp)
        f.close()
        desc = dict(desc, **desc_tmp)

    full_entities = list(entities.keys())
    full_keywords = list(keywords.keys())
    print('Adding key entity relations ...')
    add_key_entity(entities, keywords, full_entities, full_keywords, desc)

    print('Done!!!')
    with open('time.txt', 'a') as f:
        f.write('Adding key entity relations takes %.3f minutes\n'%((time.time()-start_time)/60))















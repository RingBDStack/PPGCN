
# coding: utf-8

# In[1]:

import json
import sys
#import synonyms
#from multiprocessing import Pool

'''
def remove_ambiguity(context, possible_entities, kb):
    if len(possible_entities) == 1:
        return possible_entities[0]
    score = []
    for e in possible_entities:
        try:
            possible_context = str(kb[e]).replace('[', '').replace('\'', '').replace(']', '')
        except:
            score.append(0)
            continue
        score.append(synonyms.compare(context, possible_context))
    return possible_entities[score.index(max(score))]
'''


def add_entities(entities, full_entities, raw_entities, kb, r_e, start, end):
#    tmp_entities = {}
#    for i in range(start, end):
#        tmp_entities[full_entities[i]] = []

    for i in range(start, end):
#        print(i)
        entity = full_entities[i]
        try:
            related_entities = kb[entity]
#            context = str(kb[entity]).replace('[', '').replace('\'', '').replace(']', '')
        except:
            continue
        num_related = len(related_entities)
        for j in range(num_related):
            r = related_entities[j]
            if r in raw_entities:
                r = r_e[r]
#                tmp_entities[entity].append(r)
                try:
                    entities[r].append(entity)
                    entities[entity].append(r)
                except:
                    continue
            elif r in full_entities:
#                tmp_entities[entity].append(r)
                try:
                    entities[r].append(entity)
                    entities[entity].append(r)
                except:
                    continue
#                    tmp_entities[r] = [entity]
    '''
    for k in list(entities.keys()):
        entities[k] = list(set(entities[k]))
    '''
    with open('entities/entities{0}_{1}.json'.format(start, end), 'w') as json_file:
        json.dump(entities, json_file)



if __name__ == "__main__":
    import time
    start_time = time.time()
    with open('entities.json', 'r') as json_file:
        entities = json.load(json_file)
    print(len(entities))
    with open('r_e.json', 'r') as json_file:
        r_e = json.load(json_file)
    
#    print('Start building kb dict ...')
    kb = {}
    for i in range(10):
        f = open('Kb/nocard'+str(i+1)+'.txt', 'r')
        kb_tmp = f.read()
        kb_tmp = eval(kb_tmp)
        f.close()
        kb = dict(kb, **kb_tmp)
        del kb_tmp
#        print('kb'+str(i+1)+' finished.')
#    print('Build kb dict finished.')

    full_entities = list(entities.keys())
    raw_entities = list(r_e.keys())
#    num_entities = len(full_entities)
#    print('Adding entities relations ...')

#    lnums = [(0,1000)]
#    p = Pool(5)
#    results = []
#    for i in range(len(lnums)):
#    start,end = lnums[0]
    start = int(sys.argv[1])
    end = int(sys.argv[2])
#        print("process{0} start. Range({1},{2})".format(i,start,end))
    add_entities(entities, full_entities, raw_entities, kb, r_e, start, end)
#        print("process{0} end".format(i))
#    p.close()
#    p.join()
#    for r in results:
#        print(r.get())

    print(str(start)+'~'+str(end)+' Done!!!')
    with open('time.txt', 'a') as f:
        f.write('Adding relations between entities takes %.3f munites\n'%((time.time()-start_time)/60))














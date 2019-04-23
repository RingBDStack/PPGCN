
# coding: utf-8

# In[1]:

import os
import synonyms
import json
#from multiprocessing import Pool
import time


def remove_ambiguity(context, possible_entities, kb):
    if len(possible_entities) == 1:
        return possible_entities[0]
    score = []
#    print(possible_entities)
    for e in possible_entities:
        score.append(synonyms.compare(context, e, ignore=True))
#    print(score)
    return possible_entities[score.index(max(score))]


def extract_entities(fp, m2e, kb, start, end, r_e):
    tmp_entities = {}
    tmp_events = {}
    for i in range(start, end):
        tmp_events['event'+str(i)] = []
    for i in range(start, end):
        print(i)
        f = open(fp+'event'+str(i)+'.txt', 'r')
        content = f.readlines()
        keywords = content[0].strip().replace(' ', ',')
        context = content[1].strip()
        try:
            raw_entities = content[2].strip().split(' ')
        except:
            continue
#        print(context)
        for r in raw_entities:
            try:
                e = r_e[r]
            except:
                try:
                    possible_entities = m2e[r]
                except:
                    continue
                e = remove_ambiguity(context+keywords, possible_entities, kb)
#            print('Choose:', e)
                r_e[r] = e
            tmp_events['event'+str(i)].append(e)
            try:
                tmp_entities[e].append('event'+str(i))
            except:
                tmp_entities[e] = [e, 'event'+str(i)]

        del content, context, raw_entities, keywords
        f.close()
    with open('TEST/entities/entities{0}_{1}.json'.format(start, end), 'w') as json_file:
        json.dump(tmp_entities, json_file)
    with open('TEST/events/events{0}_{1}.json'.format(start, end), 'w') as json_file:
        json.dump(tmp_events, json_file)
    with open('TEST/r_e/r_e{0}_{1}.json'.format(start, end), 'w') as json_file:
        json.dump(r_e, json_file)



if __name__ == "__main__":
    import sys
    start_time = time.time()
    fp = 'Data/'
    entities = {}
    with open('events.json', 'r') as json_file:
        events = json.load(json_file)
    print('Start building m2e dict ...')
    f = open('m2e.txt', 'r')
    m2e = f.read()
    m2e = eval(m2e)
    f.close()
    print('Build m2e dict finished.')
    print('Start building kb dict ...')
    kb = {}
    for i in range(10):
        f = open('Kb/nocard'+str(i+1)+'.txt', 'r')
        kb_tmp = f.read()
        kb_tmp = eval(kb_tmp)
        f.close()
        kb = dict(kb, **kb_tmp)
        del kb_tmp
        print('kb'+str(i+1)+' finished.')
    print('Build kb dict finished.')
    print('Extracting entities ...')
    with open('r_e.json', 'r') as json_file:
        r_e = json.load(json_file)
#    f = open('raw_entity.txt', 'r')
#    r_e = f.read()
#    r_e = eval(r_e)
#    f.close()
#   320, 100000
#    lnums = [(0, 100)]
#    p = Pool(35)
#    results = []
#    for i in range(len(lnums)):
#    start,end = lnums[0]
    start = int(sys.argv[1])
    end = int(sys.argv[2])
#        print("process{0} start. Range({1},{2})".format(i,start,end))
    extract_entities(fp, m2e, kb, start, end, r_e)
#        print("process{0} end".format(i))
#    p.close()
#    p.join()
#    for r in results:
#        print(r.get())

    print('Done!!!')
    with open('time.txt', 'a') as f:
        f.write('Remove ambiguous entities partly takes %.3f minutes\n'%((time.time()-start_time)/60))














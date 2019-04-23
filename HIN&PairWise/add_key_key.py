
# coding: utf-8

# In[1]:

import synonyms
import json
from multiprocessing import Pool
import time

def add_keys_relations(keywords, start, end):
    tmp_keywords = {}
    keys = list(keywords.keys())
    for i in range(start, end):
        tmp_keywords[keys[i]] = []
    for i in range(start, end):
        print(i)
        k = keys[i]
        neighbors = synonyms.nearby(k)[0]
        for n in neighbors:
            if n == k:
                continue
            if n in keys:
                try:
                    tmp_keywords[n].append(k)
                except:
                    tmp_keywords[n] = [k]
                tmp_keywords[k].append(n)
#        if i % 10000 == 0:
#            print('Saving dictionary ...')
#            with open('keywords/keywords{0}_{1}.json'.format(start, end), 'w') as json_file:
#                json.dump(tmp_keywords, json_file)
#            print('Saved.')
    for k in list(tmp_keywords.keys()):
        tmp_keywords[k] = list(set(tmp_keywords[k]))
    with open('keywords/keywords{0}_{1}.json'.format(start, end), 'w') as json_file:
        json.dump(tmp_keywords, json_file)
    


if __name__ == "__main__":
    start_time = time.time()
    with open('million-1/keywords.json', 'r') as json_file:
        keywords = json.load(json_file)
    print('Adding relations between keywords ...')
    num_keywords = len(keywords)
    print('Totally ', num_keywords)
    lnums = [(i*int(num_keywords/30), (i+1)*int(num_keywords/30)) for i in range(0, 30)]
    p = Pool(30)
    results = []
    for i in range(len(lnums)):
        start,end = lnums[i]
        print("process{0} start. Range({1},{2})".format(i,start,end))
        results.append(p.apply_async(add_keys_relations,args=(keywords,start,end)))
        print("process{0} end".format(i))
    p.close()
    p.join()
    for r in results:
        print(r.get())

    print('Done!!!')
    '''
    with open('time.txt', 'a') as f:
        f.write('Keys relation totally takes %.3f minutes\n'%((time.time()-start_time)/60))
    '''









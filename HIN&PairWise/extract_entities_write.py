import os
import pynlpir
import time

pynlpir.open()

def find_raw_entity(text):
    raw_entities = []
    segments = pynlpir.segment(text, pos_names = 'all')
    for segment in segments:
        try:
            if (segment[1] == 'noun:other proper noun')|(segment[1] == 'noun:organization/group name'):
                raw_entities.append(segment[0])
            elif segment[1].startswith('noun:personal name'):
                raw_entities.append(segment[0])
            elif segment[1].startswith('noun:toponym'):
                raw_entities.append(segment[0])
        except:
            continue
    # ret list
    return list(set(raw_entities))

start_time = time.time()

#files = os.listdir('news/')
for i in range(9000000):
    print(i)
    f = open('news/event'+str(i+18778+1000000)+'.txt', 'r')
    content = f.readlines()
    f.close()
    text = content[1].strip()
    try:
        raw_entities = find_raw_entity(text)
    except:
        raw_entities = []
    with open('news/event'+str(i+18778+1000000)+'.txt', 'a') as f:
        f.write(str(raw_entities).replace('\'', '').replace('[', '').replace(']', '').replace(',', ''))

print('Done!!!')
with open('time.txt', 'a') as f:
    f.write('Extracting entities takes %.3f minutes\n'%((time.time()-start_time)/60))

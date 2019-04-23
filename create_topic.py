import json
import time
start_time = time.time()

with open('t.json', 'r') as f:
    t = json.load(f)
with open('events.json', 'r') as json_file:
	events = json.load(json_file)
with open('keywords.json', 'r') as json_file:
	keywords = json.load(json_file)
with open('entities.json', 'r') as json_file:
	entities = json.load(json_file)
with open('r_e.json', 'r') as json_file:
	r_e = json.load(json_file)

topics = {}
for i in range(10004):
	try:
		topics[t[str(i)]] = topics[t[str(i)]]+events['event'+str(i)]+[t[str(i)]]
	except:
		topics[t[str(i)]] = events['event'+str(i)]+[t[str(i)]]
	events['event'+str(i)].append(t[str(i)])

for i in range(10004):
	print(i)
	f = open('Data/event'+str(i)+'.txt', 'r')
	content = f.readlines()
	tmp_k = content[0].strip().split(' ')
	for k in tmp_k:
		try:
			keywords[k].append(t[str(i)])
		except:
			continue
	try:
		tmp_e = content[3].strip().split(' ')
	except:
		continue
	for r in tmp_e:
		try:
			entities[r_e[r]].append(t[str(i)])
		except:
			continue

with open('events.json', 'w') as json_file:
	json.dump(events, json_file)
with open('keywords.json', 'w') as json_file:
	json.dump(keywords, json_file)
with open('entities.json', 'w') as json_file:
	json.dump(entities, json_file)
with open('topics.json', 'w') as json_file:
	json.dump(topics, json_file)
f = open('time.txt', 'a')
f.write('Creating topics takes %.3f minutes\n'%((time.time()-start_time)/60))
f.close()

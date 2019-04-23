from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
with open("content.txt","r",encoding='utf-8') as contentfile:
    content = eval(contentfile.read())
    contents = list()
    for key in content.keys():
        contents.append(content[key].replace(","," "))


with open("event.txt","r",encoding="utf-8") as eventfile:
    event = eval(eventfile.read())
    s = set()
    index = {}
    eventlabel = list()
    n = 0
    l = len(event.keys())
    for i in range(l):
        if event[i] not in s:
            index[event[i]] = n
            n = n + 1
        s.add(event[i])
    print(n)
    for i in range(l):
        eventlabel.append(index[event[i]])
    eventlabel = np.array(eventlabel)
    print(eventlabel.shape)

tfidf = TfidfVectorizer()
re = tfidf.fit_transform(contents).toarray()

np.save("tf_idf_xdata.npy",re)
print(re.shape)


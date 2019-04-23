import numpy as np

events = np.load("event_label.npy")
storys = np.load("story_label.npy")
event = []
story = []
event_set = set()
story_set = set()
event_num = 0
story_num = 0
l = events.shape[0]
print(l)
for i in range(5000):
   story_temp = events[i]
   event_temp = storys[i]
   if story_temp not in story_set:
      story_num += 1
      story_set.add(story_temp)
   if event_temp not in event_set:
      event_num += 1
      event_set.add(event_temp)
   event.append(event_num-1)
   story.append(story_num-1)

np.save("ring_story_label.npy",np.array(story))
np.save("ring_event_label.npy",np.array(event))
print(event_num)
print(story_num)

xdata = np.load("ring_xdata.npy")
xdata = xdata[0:5000]
np.save("ring_xdata_new.npy",np.array(xdata))


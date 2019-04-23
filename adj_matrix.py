import numpy as np
import os


mete_path_size = 13

# adj_data = np.zeros([mete_path_size,node_size,node_size]).astype(float)
# for i in range(mete_path_size):
def d_matrix_pro(adj_data):
    d_matrix = np.sum(adj_data,1)
    d_h = np.power(d_matrix,-0.5)
    t = []
    for i in range(node_size):
    	t.append(d_h)
    t = np.array(t)

    t = t*np.eye(node_size)

    return np.dot(np.dot(t,adj_data),t)

l = []

n  = 0
for filename in os.listdir("Sim"):
    f = np.load("Sim/" + filename)

    
    #if n == 4:
    #	print(f)
    #n = n + 1
    #temp = abs(-np.log(1-(1-np.exp(-37))*f))
    #temp = d_matrix_pro(temp)

    l.append(f)
adj_data = np.array(l)

print(adj_data[4])
print(adj_data.shape)
np.save("adj_data.npy",adj_data)

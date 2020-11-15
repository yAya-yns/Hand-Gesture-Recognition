import networkx as nx

'''
Experiment on Graph AE

Ignore this for now (not implemented)
'''
G = nx.Graph()

num_keypoints = 21
num_fingers = 5

for i in range(num_fingers):

    G.add_edge(0, 4*i+1)
    for j in range(4*i+1, 4*i+4):
        G.add_edge(j, j+1)

matrix = nx.adjacency_matrix(G)


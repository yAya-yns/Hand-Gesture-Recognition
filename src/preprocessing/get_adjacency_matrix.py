import networkx as nx

G = nx.Graph()

num_keypoints = 21
num_fingers = 5

for i in range(num_keypoints*2):

    G.add_node(i)

offsets = [0, num_keypoints]

for i in range(num_keypoints):
    G.add_edge(i, i+offsets[1])
for offset in offsets:

    for i in range(num_fingers):

        G.add_edge(0+offset, 4*i+1+offset)
        for j in range(4*i+1+offset, 4*i+4+offset):
            G.add_edge(j+offset, j+1+offset)

matrix = nx.adjacency_matrix(G)


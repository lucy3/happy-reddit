import networkx as nx
#from networkx import edge_betweenness_centrality as betweenness
#from networkx.algorithms import community
import community
import itertools
from collections import defaultdict

USER_EDGES = "../logs/user_edges.tsv"
TOPIC_EDGES = "../logs/topic_edges.tsv"
USER_NODES = "../logs/user_nodes.tsv"
TOPIC_NODES = "../logs/topic_nodes.tsv"
OUT = "../results/communities.txt"

def communities(out, com_type):
    if com_type == "user":
        edges_path = USER_EDGES
        nodes_path = USER_NODES
    else:
        edges_path = TOPIC_EDGES
        nodes_path = TOPIC_NODES
    G = nx.Graph()
    with open(edges_path, 'r') as edge_file:
        for line in edge_file:
            contents = line.split('\t')
            if contents[0] != 'Source':
                start = int(contents[0])
                end = int(contents[1])
                weight = float(contents[2])
                G.add_edge(start, end, weight=weight)
    node_dict = {}
    with open(nodes_path, 'r') as node_file:
        for line in node_file:
            contents = line.split('\t')
            if contents[0] != 'Id':
                node_dict[int(contents[0])] = contents[1]
    
    partition = community.best_partition(G)
    community_nodes = defaultdict(set)
    for node in partition:
        community_nodes[partition[node]].add(node)
    print com_type + " communities"
    i = 0
    for comm in community_nodes:
        i += 1
        print >> out, com_type + str(i),
        for node in community_nodes[comm]:
            print >> out, node_dict[node],
        print >> out
    print community.modularity(partition, G)
    '''
    # Girvan-Newman
    def most_central_edge(G):
        centrality = betweenness(G, weight='weight')
        return max(centrality, key=centrality.get)
    coms = community.girvan_newman(G, most_valuable_edge=most_central_edge)
    i = 0
    for tup in itertools.islice(coms, 50):
        i += 1
        communities = list(tup)
        print i, community.modularity(G, communities, weight='weight')
    '''
        
out = open(OUT, 'w')
communities(out, "user")
communities(out, "topic")
out.close()
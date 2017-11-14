"""
Generates subreddit graph
"""
import json
from collections import defaultdict
import itertools

INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
SUBREDDIT_USERS = "../data/subreddit_users.json"
EDGES = "../logs/user_edges.tsv"
NODES = "../logs/user_nodes.tsv"
TOP_100 = "../logs/top_100subreddits.txt"

def generate_user_sets(): 
    """
    @return: 
    - subreddit: list of commenters
    """
    d = defaultdict(set)
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            comment = json.loads(line)
            d[comment['subreddit']].add(comment['author'])
    new_d = {}
    for sub in d: 
        new_d[sub] = list(d[sub])
    with open(SUBREDDIT_USERS, 'w') as out: 
        json.dump(new_d, out)

def user_graph():
    """
    Subreddits by user
    data
    """
    with open(SUBREDDIT_USERS, 'r') as inp: 
        data = json.load(inp)
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    d = {}
    for sub in data:
        if sub in subreddits:
            d[sub] = set(data[sub])
    # assign node IDs to subs
    sorted_subs = sorted(subreddits)
    sub_id = {} 
    for i, sub in enumerate(sorted_subs):
        sub_id[sub] = i
    # write node file
    nodes_out = open(NODES, 'w')
    nodes_out.write('Id\tLabel\tWeight\n')
    for sub in d:
        nodes_out.write(str(sub_id[sub]) + '\t' + sub + '\t' \
                        + str(len(d[sub])) + '\n')
    nodes_out.close()  
    # write edges file
    pairs = itertools.combinations(d.keys(), 2)
    edges_out = open(EDGES, 'w')
    edges_out.write('Source\tTarget\tWeight\n')
    for p in pairs: 
        weight = len(d[p[0]]&d[p[1]])/float(len(d[p[0]]|d[p[1]]))
        edges_out.write(str(sub_id[p[0]]) + '\t' + \
                        str(sub_id[p[1]]) + '\t' + \
                        str(weight) + '\n')
    edges_out.close()

def topic_graph(): 
    """
    Subreddits by topic similarity 
    Maybe post similarity, 
    post title similarity, 
    or description similarity
    """
    pass

def main():
    user_graph()

if __name__ == "__main__":
    main()
"""
Getting post IDs
from May 2015
"""
import json

INPUT = "/dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05"
OUTPUT = "../data/post_names.txt"

ids = []
with open(INPUT, 'r') as input_file: 
    for line in input_file: 
        post = json.loads(line)
        ids.append(post['name'])
with open(OUTPUT, 'w') as out_file: 
    out_file.write(' '.join(ids))

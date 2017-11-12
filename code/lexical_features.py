"""
Creates lexical features

Should be parallelized:
xargs -a /dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05 -0 -d '\n' -P 144 -n 1 python lexical_features.py

Output is named
linkid_commentid.npy
"""
import json
import numpy
import sys

POST_IDs = "../data/post_IDs.txt"
INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
OUTPUT = "../logs/lexical_vectors"

def get_text():
    '''
    @input
        - 
    @return
        - 
    '''
    with open(POST_IDs, 'r') as post_id_file: 
        post_ids = set(post_id_file.read().split())
    i = 0
    with open(INPUT, 'r') as input_file: 
        for line in input_file: 
            comment = json.loads(line)
            post = comment['link_id'].split('_')[-1]
            if post in post_ids: 
                i += 1
    print i
            #comment['body']
    

def get_lexical_features(text, post_id, comment_id):
    '''
    @input
        - 
    @return
        - 
    '''
    pass

def main():
    with open(POST_IDs, 'r') as post_id_file: 
        post_ids = set(post_id_file.read().split())
    comment = json.loads(sys.argv[1])
    post = comment['link_id'].split('_')[-1]
    if post in post_ids: 
        get_lexical_features(comment['body'], post, comment['id'])
    #get_text()

if __name__ == "__main__":
    main()
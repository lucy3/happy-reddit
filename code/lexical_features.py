"""
Creates lexical features

Should be parallelized:
xargs -a /dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05 -0 -d '\n' -P 144 -n 1 python lexical_features.py

Output is named
subreddit_linkid_commentid.npy
"""
import json
import numpy
import sys
import re
from collections import defaultdict, Counter
import numpy as np

POST_IDs = "../data/post_IDs.txt"
INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
OUTPUT = "../logs/lexical_vectors/"
LIWC = "/dfs/scratch1/lucy3/twitter-relationships/data/en_liwc.txt"
TOP_100 = "../logs/top_100subreddits_comments.txt"
GILDS = "../logs/comment_gilds_balanced.json"

def get_liwc_groups():
    """
    Parses LIWC file
    return:
    - words with suffixed endings, other words
    - list of liwc group names
    """
    liwc = open(LIWC, 'r')
    star_words = defaultdict(list) # prefix : categories
    all_words = defaultdict(set) # category : words 
    liwc_names = [] 
    for line in liwc:
        l = line.split()
        g = l[0]
        liwc_names.append(g)
        words = l[1:]
        for w in words:
            w = w.lower()
            if w.endswith('*'):
                w = w[:-1]
                star_words[w].append(g)
            else:
                all_words[g].add(w) 
    liwc.close()
    return star_words, all_words, liwc_names

def get_liwc_features(line, star_words, all_words, liwc_names):
    """
    in: 
    - line: str
    return:
    - augmented feature list of counts of
    words in each LIWC group in tw
    """
    tok_counts = Counter(line.split())
    new_features = [0]*len(liwc_names)
    indices = {k: v for v, k in enumerate(liwc_names)}
    for g in all_words:
        new_features[indices[g]] += sum([c for x,c in tok_counts.iteritems() if x in all_words[g]])
    for tok in tok_counts:
        for w in star_words:
            if tok.startswith(w): 
                groups = star_words[w]
                for g in groups:
                    new_features[indices[g]] += tok_counts[tok]
    total = float(len(line.split()))
    if total != 0:
        new_features = [t/total for t in new_features]
    return new_features

def get_lexical_features(text, post_id, comment_id, subreddit, edited, \
                         star_words, all_words, liwc_names):
    '''
    @input
        - 
    @return
        - 
    '''
    if edited: 
        # remove edits, since they may thank the gilder after the fact
        text_lines = text.split('\n')
        text_lines = [line for line in text_lines if 'Edit' not in line and \
                              'ETA' not in line and 'edit' not in line \
                              and 'eta' not in line and 'gold' not in line and \
                                  'Gold' not in line and 'gild' not in line]
        text = '\n'.join(text_lines)
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower().encode('utf-8')
    liwc_feats = get_liwc_features(text, star_words, all_words, liwc_names)
    length = [len(text.split())]
    all_feats = np.array(length + liwc_feats)
    np.save(OUTPUT + subreddit + '_' + post_id + '_' + comment_id + '.npy', all_feats)

def main():
    with open(GILDS, 'r') as gilds_file:
        gilds = json.load(gilds_file)
    gilded_dataset = set(gilds.keys())
    '''
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    with open(POST_IDs, 'r') as post_id_file: 
        post_ids = set(post_id_file.read().split())
    '''
    star_words, all_words, liwc_names = get_liwc_groups()
    comment = json.loads(sys.argv[1])
    post = comment['link_id'].split('_')[-1]
    subreddit = comment['subreddit']
    comment_id = comment['id']
    if subreddit+'_'+post+'_'+comment_id in gilded_dataset: 
        get_lexical_features(comment['body'], post, comment_id, \
                             subreddit, comment['edited'], \
                            star_words, all_words, liwc_names)

if __name__ == "__main__":
    main()
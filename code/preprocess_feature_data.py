"""
Preprocessing for lexical features. 
For example: 
    Get bodies of comments we want in a separate file
    Get bodies of parents and posts as dictionaries
    Brown corpus word likelihoods 
"""
import pprint
import json
import numpy
import sys
import re
from collections import defaultdict, Counter
import numpy as np
import time
from nltk import ngrams
from nltk.corpus import brown
import string

DATA = "RANK"
if DATA == "GILDS": 
    GILDS = "../logs/comment_gilds_classifier.json"
    OUTPUT = "../data/gilded_classifier_comments"
    POST_OUTPUT = "../data/gilded_classifier_posts"
    PARENT_OUTPUT = "../data/gilded_classifier_parents"
elif DATA == "RANK": 
    GILDS = "../logs/comment_rank_classifier.json"
    OUTPUT = "../data/rank_classifier_comments"
    POST_OUTPUT = "../data/rank_classifier_posts"
    PARENT_OUTPUT = "../data/rank_classifier_parents"
INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
POST_INPUT = "/dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05"
BROWN_COUNTS = "../logs/brown_counts.json"
TOP_100 = "../logs/top_100subreddits_comments.txt"
SUB_PROBS = "../logs/subreddit_word_probs.json"

import json
from collections import defaultdict
import time

def nested_dict():
    return defaultdict(dict)

def comment_bodies():
    with open(GILDS, 'r') as gilds_file:
        gilds = json.load(gilds_file)
    gilded_dataset = set(gilds.keys())
    out = open(OUTPUT, 'w')
    res = defaultdict(nested_dict)
    i = 0
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            if i % 5000000 == 0: print time.time()
            i += 1
            comment = json.loads(line)
            post_id = comment['link_id'].split('_')[-1]
            subreddit = comment['subreddit']
            comment_id = comment['id']
            name = subreddit+'_'+post_id+'_'+comment_id
            if name in gilded_dataset: 
                comment_dict = {}
                comment_dict['body'] = comment['body']
                comment_dict['edited'] = comment['edited']
                comment_dict['parent_id'] = comment['parent_id']
                res[subreddit][post_id][comment_id] = comment_dict
    with open(OUTPUT, 'w') as outfile: 
        json.dump(res, outfile)
        
def post_bodies():
    with open(OUTPUT, 'r') as input_file:
        comments = json.load(input_file)
    posts = set()
    for subreddit in comments:
        posts.update(comments[subreddit].keys())
    res = defaultdict(dict)
    with open(POST_INPUT, 'r') as input_file: 
        for line in input_file: 
            post = json.loads(line)
            if post['id'] in posts: 
                post_dict = {}
                subreddit = post['subreddit']
                post_dict['title'] = post['title']
                if 'selftext' in post:
                    post_dict['text'] = post['selftext'] 
                else:
                    post_dict['text'] = ''
                res[subreddit][post['id']] = post_dict
    with open(POST_OUTPUT, 'w') as outfile:
        json.dump(res, outfile)
    
def parent_bodies():
    with open(OUTPUT, 'r') as input_file:
        comments = json.load(input_file)
    parent_comments = set()
    for subreddit in comments:
        posts = comments[subreddit]
        for post in posts: 
            for comment in posts[post]:
                parent_id = posts[post][comment]['parent_id'].split('_')[-1]
                if parent_id != post:
                    parent_comments.add(parent_id)
    res = defaultdict(nested_dict)
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            comment = json.loads(line)
            comment_id = comment['id']
            subreddit = comment['subreddit']
            post_id = comment['link_id'].split('_')[-1]
            if comment_id in parent_comments: 
                res[subreddit][post_id][comment_id] = comment['body']
    with open(PARENT_OUTPUT, 'w') as outfile:
        json.dump(res, outfile)
        
def count_brown():
    words = brown.words()
    words = [re.sub(r'[^\w\d\'\s]','',w).lower() for w \
             in words if w not in string.punctuation]
    word_counts = Counter(words)
    sentences = brown.sents()
    bigrams = []
    for s in sentences: 
        s = [re.sub(r'[^\w\d\'\s]','',w).lower() for w \
             in s if w not in string.punctuation]
        s = [w for w in s if w != '']
        bigrams.extend(ngrams(s, 2))
    bigram_counts = Counter(bigrams)
    fixed_bigram_counts = Counter()
    for bigram in bigram_counts:
        fixed_bigram_counts[bigram[0] \
            + ' ' + bigram[1]] = bigram_counts[bigram]
    res = {}
    res['words'] = word_counts
    res['bigrams'] = fixed_bigram_counts
    with open(BROWN_COUNTS, 'w') as outfile: 
        json.dump(res, outfile)
        
def count_subreddit():
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    res = defaultdict(Counter)
    subreddit_comments = Counter() # number of comments in each subreddit
    i = 0
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            comment = json.loads(line)
            subreddit = comment['subreddit']
            if subreddit in subreddits: 
                i += 1
                if i % 5000000 == 0: print time.time()
                subreddit_comments[subreddit] += 1
                text = comment['body']
                text = re.sub(r'[^\w\d\'\s]','',text).lower()
                words = set(text.split())
                res[subreddit].update(words)
    ret = {k:{} for k in subreddits}
    for subreddit in res: 
        for word in res[subreddit]:
            prob = res[subreddit][word]/float(subreddit_comments[subreddit])
            ret[subreddit][word] = prob
    with open(SUB_PROBS, 'w') as outfile: 
        json.dump(ret, outfile)

def main():
    comment_bodies()
    post_bodies()
    parent_bodies()
    #count_brown()
    #count_subreddit()
        
if __name__ == "__main__":
    main()

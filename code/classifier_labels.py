"""
Create classifier labels

outputs three jsons where the keys are
subreddit_linkid_commentid

And the values are either
gilded (0/1), score (int), or rank
where rank 1 if a comment is in the top half of
scores on that post or 0 if it is in the lower half
"""
import json
from collections import defaultdict
import random

INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
SCORES = "../logs/comment_scores.json"
RANK = "../logs/comment_rank.json"
GILDS = "../logs/comment_gilds.json"
GILDS_BALANCED = "../logs/comment_gilds_balanced.json"
RANK_SUBSET = "../logs/comment_rank_subset.json"
POST_IDs = "../data/post_IDs.txt"
TOP_100 = "../logs/top_100subreddits_comments.txt"

def get_rank():
    '''
    For each post, rank its comments
    '''
    with open(SCORES, 'r') as scores_file:
        scores = json.load(scores_file)
    # dictionary from posts to tuples of comment, score
    posts_comments = defaultdict(list)
    for name in scores:
        items = name.split('_')
        subreddit_post = '_'.join(items[:-1])
        comment_id = items[-1]
        score = scores[name]
        posts_comments[subreddit_post].append((comment_id, score))
    rank = {}
    for pc in posts_comments:
        sorted_posts = sorted(posts_comments[pc], key=lambda tup: tup[1])
        halfway = len(sorted_posts)/2
        for i, com in enumerate(sorted_posts):
            if i >= halfway:
                rank[pc + '_' + com[0]] = 1
            else:
                rank[pc + '_' + com[0]] = 0
    assert len(set(scores.keys())) == len(set(rank.keys()))
    with open(RANK, 'w') as rank_file:
        json.dump(rank, rank_file)

def get_gilds_scores():
    '''
    Iterate over all comments and save
    a json of subreddit_linkid_commentid
    to gild, and another one from 
    subreddit_linkid_commentid to score
    '''
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    with open(POST_IDs, 'r') as post_id_file: 
        # all posts in May 2015
        post_ids = set(post_id_file.read().split())
    scores = {}
    gilds = {}
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            comment = json.loads(line)
            post_id = comment['link_id'].split('_')[-1]
            subreddit = comment['subreddit']
            if post_id in post_ids and subreddit in subreddits: 
                # only get comments of May 2015 posts
                # in the top 100 subreddits
                comment_id = comment['id']
                name = subreddit+'_'+post_id+'_'+comment_id
                scores[name] = comment['score']
                gilds[name] = comment['gilded']
    with open(SCORES, 'w') as scores_file:
        json.dump(scores, scores_file)
    with open(GILDS, 'w') as gilds_file:
        json.dump(gilds, gilds_file)
        
def balance_gilds():
    '''
    Balance the dataset for gilds. 
    Grab all gilded comments and then random subset 
    of non-gilded comments. 
    '''
    random.seed(0)
    with open(GILDS, 'r') as gilds_file:
        gilds = json.load(gilds_file)
    gilds_balanced = {}
    non_gilded = []
    for name in gilds: 
        if gilds[name] == 1: 
            gilds_balanced[name] = 1
        else:
            non_gilded.append(name)
    num_gilds = len(gilds_balanced)
    non_gilded_samp = random.sample(non_gilded, num_gilds)
    for samp in non_gilded_samp: 
        gilds_balanced[samp] = 0
    with open(GILDS_BALANCED, 'w') as gilds_balanced_file:
        json.dump(gilds_balanced, gilds_balanced_file) 
        
def subset_rank():
    random.seed(0)
    with open(RANK, 'r') as rank_file:
        rank = json.load(rank_file)
    rank_subset = {}
    total_top = 10000
    total_bottom = 10000
    rank_keys = rank.keys()
    random.shuffle(rank_keys)
    for name in rank_keys: 
        if rank[name] == 1 and total_top > 0:
            rank_subset[name] = 1
            total_top -= 1
        elif rank[name] == 0 and total_bottom > 0:
            rank_subset[name] = 0
            total_bottom -= 1
        if total_top <= 0 and total_bottom <= 0:
            break
    with open(RANK_SUBSET, 'w') as rank_subset_file:
        json.dump(rank_subset, rank_subset_file)

def main():
    #get_gilds_scores()
    #get_rank()
    #balance_gilds()
    subset_rank()

if __name__ == "__main__":
    main()
"""
Create classifier labels

outputs three jsons where the keys are
subreddit_linkid_commentid

And the values are either
gilded (0/1), score (int), or rank
where rank is dependent on quartile: 1 (top), 2, 3, 4 (bottom)
"""
import json
from collections import defaultdict
import random

INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
SCORES = "../logs/comment_scores.json"
RANK = "../logs/comment_rank.json"
RANK_ALL = "../logs/comment_rank_all.json"
GILDS = "../logs/comment_gilds.json"
GILDS_CLASSIFIER = "../logs/comment_gilds_classifier.json"
RANK_CLASSIFIER = "../logs/comment_rank_classifier.json"
POST_IDs = "../data/post_IDs.txt"
TOP_100 = "../logs/top_100subreddits_comments.txt"
COMMUNITY = "../results/communities.txt"

def get_rank():
    '''
    For each post, rank its comments into quartile
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
        if len(sorted_posts) >= 4:
        # check each post has at least 4 comments
            upper = 3*len(sorted_posts)/4.0
            lower = len(sorted_posts)/4.0
            for i, com in enumerate(sorted_posts):
                if i >= upper:
                    rank[pc + '_' + com[0]] = 1
                elif i < lower:
                    rank[pc + '_' + com[0]] = 0
    with open(RANK, 'w') as rank_file:
        json.dump(rank, rank_file)
        
def get_rank_all():
    '''
    For each post, rank its comments into quartile
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
        upper = 3*len(sorted_posts)/4.0
        mid = len(sorted_posts)/2.0
        lower = len(sorted_posts)/4.0
        for i, com in enumerate(sorted_posts):
            if i >= upper:
                rank[pc + '_' + com[0]] = 1
            elif i >= mid: 
                rank[pc + '_' + com[0]] = 2
            elif i >= lower: 
                rank[pc + '_' + com[0]] = 3
            else:
                rank[pc + '_' + com[0]] = 4
    with open(RANK_ALL, 'w') as rank_file:
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
    gilds_data = {}
    non_gilded = []
    for name in gilds: 
        if gilds[name] >= 1: 
            gilds_data[name] = 1
        else:
            non_gilded.append(name)
    num_gilds = len(gilds_data)
    print "Number of gilded samples", num_gilds
    train_size = 8842
    test_size = num_gilds - train_size
    non_gilded_samp = random.sample(non_gilded, train_size + test_size*5)
    for samp in non_gilded_samp: 
        gilds_data[samp] = 0
    print "Total samples:", len(gilds_data.keys())
    with open(GILDS_CLASSIFIER, 'w') as gilds_data_file:
        json.dump(gilds_data, gilds_data_file) 
        
def subset_rank():
    # 12000 for each 
    random.seed(0)
    with open(RANK, 'r') as rank_file:
        rank = json.load(rank_file)
    rank_subset = {}
    one = 12500
    zero = 12500
    rank_keys = rank.keys()
    random.shuffle(rank_keys)
    for name in rank_keys: 
        if rank[name] == 1 and one > 0:
            rank_subset[name] = 1
            one -= 1
        elif rank[name] == 0 and zero > 0:
            rank_subset[name] = 0
            zero -= 1
        if one <= 0 and zero <= 0:
            break
    print "Total samples:", len(rank_subset.keys())
    with open(RANK_CLASSIFIER, 'w') as rank_data_file:
        json.dump(rank_subset, rank_data_file)
        
        
def count_community_gilds():
    communities = {}
    with open(COMMUNITY, 'r') as community_file:
        for line in community_file:
            contents = line.split()
            communities[contents[0]] = set(contents[1:])
    with open(GILDS_CLASSIFIER, 'r') as gilds_file:
        gilds = json.load(gilds_file)
    subreddit_gilds = defaultdict(int) # subreddit: # of gilds
    subreddit_total = defaultdict(int)
    for name in gilds:
        items = name.split('_')
        subreddit = '_'.join(items[:-2])
        if gilds[name] == 1:
            subreddit_gilds[subreddit] += 1
        subreddit_total[subreddit] += 1
    for com in communities: 
        print com, 
        total_total = 0
        total = 0
        for subred in communities[com]:
            total += subreddit_gilds[subred]
            total_total += subreddit_total[subred]
        print total, total_total

def main():
    #get_gilds_scores()
    #get_rank()
    #balance_gilds()
    #subset_rank()
    #count_community_gilds()
    get_rank_all()

if __name__ == "__main__":
    main()

import json 
import numpy as np

INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
POST_IDs = "../data/post_IDs.txt"
TOP_100 = "../logs/top_100subreddits_comments.txt"

def process_posts(filename):
	deleted_count = 0
	total_count = 0


	with open(filename,'r') as f:
		for post_json in f:
			post = json.loads(post_json)
			try:
				subreddit = post["subreddit"]
			except KeyError:
				continue

			author = post['author']
			if author == '[deleted]':
				deleted_count += 1
			total_count += 1

	
    
	print deleted_count,total_count

def count_deleted_gilds():
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    with open(POST_IDs, 'r') as post_id_file: 
        # all posts in May 2015
        post_ids = set(post_id_file.read().split())
    scores = {}
    gilds = {}
    count = 0
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            comment = json.loads(line)
            post_id = comment['link_id'].split('_')[-1]
            subreddit = comment['subreddit']
            if post_id in post_ids and subreddit in subreddits: 
                # only get comments of May 2015 posts
                # in the top 100 subreddits
                if comment['gilded'] == 1 and comment['author'] == "[deleted]":
                    count += 1
    print count

def main():
    '''
	# /dfs/scratch1/jmendels/happy-reddit/logs/first_5000_posts
	f = '/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05'
	#f = '../logs/first_5000_posts'
	#f = 'first_5000_posts'
	process_posts(f)
    '''
    count_deleted_gilds()
	
	


if __name__ == "__main__":
	main()
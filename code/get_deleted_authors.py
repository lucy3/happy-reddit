import json 
import numpy as np




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



def main():
	# /dfs/scratch1/jmendels/happy-reddit/logs/first_5000_posts
	f = '/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05'
	#f = '../logs/first_5000_posts'
	#f = 'first_5000_posts'
	process_posts(f)
	
	


if __name__ == "__main__":
	main()
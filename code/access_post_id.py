import json 
from collections import defaultdict,Counter
import numpy as np




def make_dict(filename,outfile):

	#top_file = 'top_100subreddits_comments.txt'
	top_file = '../logs/top_100subreddits_comments.txt'
	with open(top_file,'r') as infile:
		subreddit_list = infile.readlines()
	subreddit_list = [s.strip('\n') for s in subreddit_list]
	top_subreddits = set(subreddit_list)

	post_dict = {}

	with open(filename,'r') as f:
		for post_json in f:
			post = json.loads(post_json)
			post_id = post['name']

			post_dict[post_id] = {}
			post_dict[post_id]['score'] = post['score']
			post_dict[post_id]['created_utc'] = post['created_utc']
			post_dict[post_id]['author'] = post['author']
			post_dict[post_id]['author_flair_text'] = post['author_flair_text']
	
	



	with open(outfile,'w') as fp:
		json.dump(post_dict,fp)



def main():
	
	f = '/dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05'
	#f = '../logs/first_5000_comments'
	#f = 'first_5000_posts'
	outfile = '../logs/post_dict.json'
	make_dict(f,outfile)
	


if __name__ == "__main__":
	main()
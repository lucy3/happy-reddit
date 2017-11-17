import json 
from collections import defaultdict,Counter
import numpy as np




def make_dict(filename):


	top_file = '../logs/top_100subreddits_comments.txt'

	with open(top_file,'r') as infile:
		subreddit_list = infile.readlines()
	subreddit_list = [s.strip('\n') for s in subreddit_list]
	top_subreddits = set(subreddit_list)


	with open('../data/post_IDs.txt','r') as fp:
		post_list = fp.readlines()
	post_list = [s.strip('\n') for s in post_list]
	post_set = set(post_list)


	comment_dict = {}

	with open(filename,'r') as f:
		for comment_json in f:
			comment = json.loads(comment_json)
			if "subreddit" in comment and comment["subreddit"] in top_subreddits:
				subreddit = comment['subreddit']
				link_id = comment["link_id"]
				comment_id = comment['id']


				if subreddit not in comment_dict:
					comment_dict[subreddit] = {}
				if link_id not in comment_dict[subreddit]:
					comment_dict[subreddit][link_id] = {}
				if comment_id not in comment_dict[subreddit][link_id]:
					comment_dict[subreddit][link_id][comment_id] = {}
				



				comment_dict[subreddit][link_id][comment_id]['score'] = comment['score']
				comment_dict[subreddit][link_id][comment_id]['created_utc'] = comment['created_utc']
				comment_dict[subreddit][link_id][comment_id]['author_flair_text'] = comment['author_flair_text']
				comment_dict[subreddit][link_id][comment_id]['controversiality'] = comment['controversiality']
				comment_dict[subreddit][link_id][comment_id]['author'] = comment['author']
				comment_dict[subreddit][link_id][comment_id]['parent_id'] = comment['parent_id']
				if 'can_gild' in comment:
					comment_dict[subreddit][link_id][comment_id]['can_gild'] = comment['can_gild']
	
	


	for subreddit in comment_dict:
		f = '../../comment_dicts/' + subreddit + '_comment_dict.json'
		#f = 'comment_dicts/' + subreddit + '_comment_dict.json'
		with open(f,'w') as outfile:
			json.dump(comment_dict[subreddit],outfile)



def main():
	
	f = '/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05'
	#f = '../logs/first_5000_comments'
	#f = 'first_5000_comments'
	make_dict(f)
	


if __name__ == "__main__":
	main()
import json
import glob



filenames = "comment_dicts/*"
subreddit_files = glob.glob(filenames)
print subreddit_files
count_post = 0
count_comments = 0
for subreddit_file in subreddit_files:
	with open(subreddit_file,'r') as jsonin:
		subreddit_dict = json.load(jsonin)
		for link_id in subreddit_dict:
			count_post += 1
			for comment_id in subreddit_dict[link_id]:
				count_comments +=1
print count_post
print count_comments
					
					
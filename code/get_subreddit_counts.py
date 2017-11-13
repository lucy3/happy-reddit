import json 
from collections import defaultdict,Counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np




def process_posts(filename):
	subreddit_counts = Counter()
	user_lookup_dict = {}
	subreddit_lookup_dict = {}


	with open(filename,'r') as f:
		for post_json in f:
			post = json.loads(post_json)
			try:
				subreddit = post["subreddit"]
				subreddit_counts[subreddit] += 1

			except KeyError:
				continue


			user = post["author"]
			if user not in user_lookup_dict:
				user_lookup_dict[user] = Counter()
			if subreddit not in subreddit_lookup_dict:
				subreddit_lookup_dict[subreddit] = Counter()
			# if subreddit not in user_lookup_dict[user]:
			# 	user_lookup_dict[user][subreddit] = 0
			# if user not in subreddit_lookup_dict[subreddit]:
			# 	subreddit_lookup_dict[subreddit][user] = 0
			user_lookup_dict[user][subreddit] += 1
			subreddit_lookup_dict[subreddit][user] += 1 
	

	with open('../logs/user_lookup_dict.json', 'w') as outfile:
		json.dump(user_lookup_dict, outfile)
	with open('../logs/subreddit_lookup_dict.json', 'w') as outfile:
		json.dump(subreddit_lookup_dict, outfile)
	with open('../logs/subreddit_counts.json', 'w') as outfile:
		json.dump(subreddit_counts, outfile)
    
	return subreddit_counts 

def plot_histogram(subreddit_counts):
	post_counts = []
	num_subreddits= []
	highest_post_count = max(subreddit_counts.values())
	for i in range(1,highest_post_count):
		post_counts.append(i)
		num_subreddits.append(subreddit_counts.values().count(i))

	print num_subreddits
	plt.scatter(post_counts,num_subreddits)
	plt.xlabel('Number of Posts')
	plt.ylabel('Number of Subreddits')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim((10**0,10**4))
	plt.ylim((10**0,10**4))
	plt.title('Post Counts for Subreddits')
	plt.savefig('../logs/subreddit_counts.png')

def get_top_subreddits(subreddit_counts,n):
	outfile = "../logs/top_" + str(n) + "subreddits.txt"
	tops = [s[0] + "\n" for s in subreddit_counts.most_common(n)]
	with open(outfile,'w') as f:
		f.writelines(tops)


def main():
	# /dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05
	# /dfs/scratch1/jmendels/happy-reddit/logs/first_5000_posts
	#subreddit_counts = process_posts('/dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05')
	with open('../logs/subreddit_counts.json','r') as f:
		subreddit_counts = Counter(json.load(f))
	get_top_subreddits(subreddit_counts,50)
	get_top_subreddits(subreddit_counts,100)

	#plot_histogram(subreddit_counts)



if __name__ == "__main__":
	main()
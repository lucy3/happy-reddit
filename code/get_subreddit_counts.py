import json 
from collections import defaultdict,Counter
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np




def process_posts(filename):
	subreddit_counts = Counter()
	# user_lookup_dict = {}
	# subreddit_lookup_dict = {}
	# scores = defaultdict(int)


	with open(filename,'r') as f:
		for post_json in f:
			post = json.loads(post_json)
			try:
				subreddit = post["subreddit"]
				subreddit_counts[subreddit] += 1

			except KeyError:
				continue


			# user = post["author"]
			
			# if user not in user_lookup_dict:
			# 	user_lookup_dict[user] = Counter()
			# if subreddit not in subreddit_lookup_dict:
			# 	subreddit_lookup_dict[subreddit] = Counter()
			
			# scores[user] += post["score"]
			# user_lookup_dict[user][subreddit] += 1
			# subreddit_lookup_dict[subreddit][user] += 1 
	


	# with open('../logs/user_lookup_dict_comments.json', 'w') as outfile:
	# 	json.dump(user_lookup_dict, outfile)
	# with open('../logs/subreddit_lookup_dict_comments.json', 'w') as outfile:
	# 	json.dump(subreddit_lookup_dict, outfile)
	with open('../logs/subreddit_counts_comments.json', 'w') as outfile:
		json.dump(subreddit_counts, outfile)
	# with open('../logs/user_scores_comments.json', 'w') as outfile:
	# 	json.dump(scores, outfile)
    
	return subreddit_counts 

def plot_histogram(subreddit_counts):
	post_counts = []
	num_subreddits= []
	# print center,hist
	plt.xlabel('Number of Posts')
	plt.ylabel('Number of Subreddits')
	plt.title('Number of posts in subreddits')
	counts = Counter(subreddit_counts.values())
	post_counts = [c[0] for c in counts.items()]
	num_subreddits = [c[1] for c in counts.items()]
	
	plt.scatter(post_counts,num_subreddits)
	plt.xscale('log')
	plt.yscale('log')
	plt.savefig('subreddit_counts.png')
	plt.show()



	# counts = Counter(subreddit_counts.values())
	# hist,bins = np.histogram(counts,bins=100)
	# print counts
	# post_counts = counts[0]
	# #print post_counts
	# num_subreddits = counts[1]
	# plt.scatter(post_counts,num_subreddits)
	plt.xlabel('Number of Posts')
	plt.ylabel('Number of Subreddits')
	plt.xscale('log')
	plt.yscale('log')
	#plt.xlim((10**0,10**6))
	#plt.ylim((10**0,10**6))
	# plt.title('Post Counts for Subreddits')
	# #plt.savefig('../logs/subreddit_counts.png')
	# plt.savefig('subreddit_counts.png')
	# plt.show()

def get_top_subreddits(subreddit_counts,n):
	outfile = "../logs/top_" + str(n) + "subreddits_comments.txt"
	tops = [s[0] + "\n" for s in subreddit_counts.most_common(n)]
	with open(outfile,'w') as f:
		f.writelines(tops)


def main():
	# /dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05
	# /dfs/scratch1/jmendels/happy-reddit/logs/first_5000_posts
	f = '/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05'
	#f = '../logs/first_5000_posts'
	subreddit_counts = process_posts(f)
	
	get_top_subreddits(subreddit_counts,100)

	# with open('subreddit_counts.json','r') as f:
	# 	subreddit_counts = json.load(f)
	#plot_histogram(subreddit_counts)



if __name__ == "__main__":
	main()
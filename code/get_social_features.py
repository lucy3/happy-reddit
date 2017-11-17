import json
import glob
import numpy as np 



# with open('../logs/post_dict.json','r') as f:
# 	post_dict = json.load(f)
# with open('../logs/user_lookup_dict_comments.json','r') as f:
# 	user_lookup_dict = json.load(f)
# with open('../logs/subreddit_lookup_dict_comments.json','r') as f:
# 	subreddit_lookup_dict = json.load(f)
# with open('../logs/user_scores_comments.json','r') as f:
# 	score_dict = json.load(f)



def loop_through_subreddits(comment_dict_directory):
	filenames = comment_dict_directory + "*"
	subreddit_files = glob.glob(filenames)
	for subreddit_file in subreddit_files:
		with open(subreddit_file,'r') as jsonin:
			subreddit_dict = json.load(jsonin)
			items = subreddit_file.split('_')
			subreddit_name = '_'.join(items[:-2])

			for link_id in subreddit_dict:
				for comment_id in subreddit_dict[link_id]:
					get_all_features(comment_id,subreddit_dict[link_id],link_id,subreddit_name)


def loop_through_samples(comment_dict_directory,sample_json,out_directory):
	with open(sample_json,'r') as f:
		sample_dict = json.load(f)
	samples = set(sample_dict.keys())
	found_samples  = set()
	print len(samples)
	num_found = 0
	filenames = comment_dict_directory + "*"
	subreddit_files = glob.glob(filenames)
	print len(subreddit_files)
	for subreddit_file in subreddit_files:
		with open(subreddit_file,'r') as jsonin:
			subreddit_dict = json.load(jsonin)
			full_name = subreddit_file.split('/')
			items = full_name[-1].split('_')
			subreddit_name = '_'.join(items[:-2])
			for link_id in subreddit_dict:
				for comment_id in subreddit_dict[link_id]:
					link_name = link_id.split('_')[-1]
					name = subreddit_name + '_' + link_name + '_' + comment_id
					feature_name = out_directory + name
					if name in samples:
						found_samples.add(name)
						num_found += 1
					else:
						continue
						#print name
						#feature_vector = get_all_features(comment_id,subreddit_dict[link_id],link_id,subreddit_name)
						#np.save(feature_name,feature_vector)
	diff = samples.difference(found_samples)
	for i in diff:
		print i
	print num_found




def get_all_features(comment_id,link_dict,link_id,subreddit_name):
	status = get_status(link_dict,comment_id,score_dict)
	parent_pop = get_parent_popularity(link_dict,comment_id,link_id)
	sub_loyalty = get_loyalty_from_subreddit(link_dict,comment_id,subreddit_name)
	user_loyalty = get_loyalty_from_user(link_dict,comment_id,subreddit_name)
	time_past = get_time_from_post(link_dict,comment_id,post_dict,link_id)
	#num_replies = get_num_replies_to_comment(link_dict,comment_id)
	feature_vector = [status,parent_pop,sub_loyalty,user_loyalty,time_past]
	return feature_vector
	

def get_status(link_dict,comment_id,score_dict):
	author = link_dict[comment_id]['author']
	total_score = score_dict[author]
	return total_score - link_dict[comment_id]['score']


def get_loyalty_from_subreddit(link_dict,comment_id,subreddit_name):
	author = link_dict[comment_id]['author']
	num_comments_from_user = subreddit_lookup_dict[subreddit_name][author]
	total_comments_in_sub = sum(subreddit_lookup_dict[subreddit_name].values())

	return float(num_comments_from_user/total_comments_in_sub)

def get_loyalty_from_user(link_dict,comment_id,subreddit_name):
	author = link_dict[comment_id]['author']
	num_comments_to_subreddit = user_lookup_dict[author][subreddit_name]
	total_comments_by_user = sum(user_lookup_dict[author].values())
	return float(num_comments_to_subreddit) / total_comments_by_user

def get_parent_popularity(link_dict,comment_id,link_id):
	parent_id = link_dict[comment_id]['parent_id']

	if parent_id in post_dict:
		parent_popularity =  post_dict[parent_id]['score']

	elif parent_id in link_dict:
		parent_popularity = link_dict[parent_id]['score']
	else:
		parent_popularity = 0
	return parent_popularity

def get_num_replies_to_comment(link_dict,comment_id):
	num_replies = 0
	for c_id in link_dict:
		if link_dict[c_id]['parent_id'] == comment_id:
			num_replies += 1
	return num_replies

def get_time_from_post(link_dict,comment_id,post_lookup_dict,link_id):
	if link_id not in post_dict:
		return 0
	post_time = post_dict[link_id]['created_utc']
	comment_time = link_dict[comment_id]['created_utc']
	return float(int(comment_time)-int(post_time))/1000

def get_distance_to_post(comment_id):
	return 0



def main():
	comment_dicts = '../../comment_dicts/'
	out_directory = '../logs/rank_samples_features/social_features/'
	sample_json = '../logs/comment_rank_subset.json'
	#out_directory = 'ranked_samples_features/social_features'
	#loop_through_subreddits(comment_dicts)
	loop_through_samples(comment_dicts,sample_json,out_directory)



if __name__ == "__main__":
	main() 
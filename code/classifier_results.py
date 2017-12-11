"""
Outputs a tsv of 

name text y_test y_pred

for random forest
"""
import json
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

GILDS = "../logs/comment_gilds_classifier.json"
INPUT = "../data/gilded_classifier_comments"
OUT = "../results/text_and_labels.tsv"

with open(GILDS, 'r') as input_file:
    gilds = json.load(input_file)
sorted_gilds = sorted(gilds.keys())

with open(INPUT, 'r') as input_file:
    # all comments in dataset
    all_comments = json.load(input_file)

out = open(OUT, 'w')
out.write('subreddit' + '\t' + 'post_id' + '\t' + 'comment_id' + \
          '\t' + 'gild' + '\t' + 'text' + '\n')

for i, comment in enumerate(sorted_gilds):
    items = comment.split('_')
    subreddit = '_'.join(items[:-2])
    post_id = items[-2]
    comment_id = items[-1]
    c = all_comments[subreddit][post_id][comment_id]
    body = c['body']
    label = gilds[comment]
    out.write(subreddit + '\t' + post_id + '\t' + comment_id + \
              '\t' + str(label) + '\t' + body + '\n')
        
out.close()

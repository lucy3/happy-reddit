from collections import Counter, defaultdict
import json
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import os
import liwc_features
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import string
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = "../data/gilded_classifier_comments"
GILDS = "../logs/comment_gilds_classifier.json"
LOG_ODDS = "../logs/log_odds/"
GLOVE = "/dfs/scratch0/jayebird/glove/glove.840B.300d.txt"
VECTORS = "../logs/log_odds_vectors.json"

def count_ngrams(texts, n):
    """
    input: 
    - texts: list of comments
    - n: n-gram size
    return: 
    - Counter of ngrams
    """
    grams = []
    for words in texts: 
        sentences = sent_tokenize(words.decode('utf-8'))
        for s in sentences: 
            s = [re.sub(r'[^\w\d\'\s]','',w).lower() for w \
                 in word_tokenize(s) if w not in string.punctuation]
            s = [w for w in s if w != '']
            grams.extend(ngrams(s, n))
    return Counter(grams)

def write_frequency_file(file_name, countr):
    """
    input: 
    - file_name: str of file name
    - countr: Counter of n-grams and their frequencies
    out: 
    - files that act as inputs to log-odds program
    """
    o = open(file_name, 'w')
    for key in countr:
        if countr[key] > 10:
            gram = '$'.join(list(key))
            o.write(str(countr[key]) + ' ' + gram.encode('utf-8') + '\n')
    o.close()
    
def do_log_odds():
    with open(GILDS, 'r') as gilds_file: 
        gilds = json.load(gilds_file)
    gilded_lines = []
    non_gilded_lines = []
    with open(INPUT, 'r') as input_file:
        # all comments in dataset
        all_comments = json.load(input_file)
    for name in gilds: 
        items = name.split('_')
        subreddit = '_'.join(items[:-2])
        post_id = items[-2]
        comment_ID = items[-1]
        comment = all_comments[subreddit][post_id][comment_ID] 
        text = comment['body']
        edited = comment['edited']
        text = text.lower().encode('utf-8')
        if edited: 
            # remove edits, since they may thank the gilder after the fact
            text_lines = text.split('\n')
            text_lines = [line for line in text_lines if 'Edit' not in line and \
                      'ETA' not in line and 'edit' not in line \
                      and 'eta' not in line and 'gold' not in line and \
                          'Gold' not in line and 'gild' not in line]
            text = '\n'.join(text_lines)
        if gilds[name] == 1:
            gilded_lines.append(text)
        else:
            non_gilded_lines.append(text)
    all_lines = gilded_lines + non_gilded_lines
    for i in range(1, 4):
        prefix = LOG_ODDS
        close_count = count_ngrams(gilded_lines, i)
        write_frequency_file(prefix+str(i)+'gilded.out', close_count)
        far_count = count_ngrams(non_gilded_lines, i)
        write_frequency_file(prefix+str(i)+'non_gilded.out', far_count)
        prior_count = close_count + far_count
        write_frequency_file(prefix+str(i)+'prior.out', prior_count)
        os.system('python bayesequal.py -f '+prefix+str(i)+\
                'gilded.out -s '+prefix+str(i)+\
                'non_gilded.out -p '+prefix+str(i)+\
                'prior.out > '+prefix+str(i)+\
                'gram_log_odds.out')
        
def load_glove(all_words):
    res = {}
    with open(GLOVE, 'r') as glove_file: 
        for line in glove_file: 
            contents = line.strip().split()
            word = contents[0]
            if word in all_words:
                vector = [float(i) for i in contents[1:]]
                res[word] = np.array(vector)
    return res

def get_vector_reps(glove, all_grams_dict):
    """
    Representations of n-grams are average of
    GloVe vectors. 
    """
    vector_dict = {}
    for g in all_grams_dict: 
        reps_dict = {}
        for gram in all_grams_dict[g]:
            words = gram.split('$')
            can_get = True
            for w in words:
                if w not in glove or w.isdigit() or w == '\'' \
                    or '_' in w or '\'\'' in w:
                    can_get = False
            if can_get: 
                rep = sum([glove[w] for w in words])/float(len(words))
                reps_dict[gram] = rep
        vector_dict[g] = reps_dict
    return vector_dict

def cluster_reps(vector_reps):
    for g in vector_reps:
        if g == '1': 
            cluster_num = 20
        elif g == '2': 
            cluster_num = 15
        else:
            cluster_num = 10
        sorted_grams = sorted(vector_reps[g].keys())
        sorted_reps = [value for (key, value) in sorted(vector_reps[g].items())]
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_jobs=-1).fit(sorted_reps)
        labels = kmeans.labels_
        label_dict = defaultdict(list) # label: [grams]
        for i in range(len(labels)):
            label_dict[labels[i]].append(sorted_grams[i])
        with open(LOG_ODDS + g + 'clusters', 'w') as cluster_file:
            for label in label_dict:
                if len(label_dict[label]) > 5: 
                    cluster_file.write(str(label) + '\t')
                    for gram in label_dict[label]:
                        cluster_file.write(gram + ' ')
                    cluster_file.write('\n')
        
def cluster_log_odds():
    grams = ['1', '2', '3'] 
    all_grams_dict = {}
    all_words = set()
    for g in grams:
        all_grams = []
        file_name = LOG_ODDS + g+'gram_log_odds.out'
        with open(file_name, 'r') as odds_file: 
            for line in odds_file: 
                contents = line.split()
                gram = contents[0]
                value = float(contents[1])
                if value > 0.95: 
                    all_grams.append(gram)
                    words = gram.split('$')
                    all_words.update(words)
        all_grams_dict[g] = all_grams
    print 'Finished reading log odds files...'
    glove = load_glove(all_words)
    print 'Finished loading glove...'
    vector_dict = get_vector_reps(glove, all_grams_dict)
    print 'Got vector representations of ngrams...'
    cluster_reps(vector_dict)
    print 'Finished kmeans clustering...'
        
def main():
    do_log_odds()
    #cluster_log_odds()
        
if __name__ == '__main__':
    main()

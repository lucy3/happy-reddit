from collections import Counter
import json
from nltk import ngrams
import re
import os

INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
GILDS = "../logs/comment_gilds_balanced.json"
LOG_ODDS = "../logs/log_odds/"

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
        w = words.strip().split()
        grams.extend(ngrams(w, n))
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
        for line in input_file:
            comment = json.loads(line)
            post_id = comment['link_id'].split('_')[-1]
            subreddit = comment['subreddit']
            comment_id = comment['id']
            name = subreddit+'_'+post_id+'_'+comment_id
            if name in gilds:
                text = comment['body']
                edited = comment['edited']
                if edited: 
                    # remove edits, since they may thank the gilder after the fact
                    text_lines = text.split('\n')
                    text_lines = [line for line in text_lines if 'Edit' not in line and \
                              'ETA' not in line and 'edit' not in line \
                              and 'eta' not in line and 'gold' not in line and \
                                  'Gold' not in line and 'gild' not in line]
                    text = '\n'.join(text_lines)
                text = re.sub(r'[^\w\s]','',text)
                text = text.lower().encode('utf-8')
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
        
def main():
    do_log_odds()
        
if __name__ == '__main__':
    main()
"""
Generates subreddit graph
"""
import json
from collections import defaultdict
import itertools
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
import time
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

POSTS = "/dfs/dataset/infolab/Reddit/submissions/2015/RS_2015-05"
INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"
SUBREDDIT_USERS = "../data/subreddit_users.json"
EDGES = "../logs/user_edges.tsv"
NODES = "../logs/user_nodes.tsv"
TOP_100 = "../logs/top_100subreddits_comments.txt"
SUBREDDIT_MODEL = "../logs/subreddit_model.doc2vec"
SUBREDDIT_TITLES = "../logs/subreddit_titles.json"
EDGES2 = "../logs/topic_edges.tsv"
NODES2 = "../logs/topic_nodes.tsv"
TF_IDF = "../logs/tf_idf.json"

def generate_user_sets(): 
    """
    @return: 
    - subreddit: list of commenters
    """
    d = defaultdict(set)
    with open(INPUT, 'r') as input_file:
        for line in input_file:
            comment = json.loads(line)
            d[comment['subreddit']].add(comment['author'])
    new_d = {}
    for sub in d: 
        new_d[sub] = list(d[sub])
    with open(SUBREDDIT_USERS, 'w') as out: 
        json.dump(new_d, out)

def user_graph():
    """
    Subreddits by user
    data
    """
    with open(SUBREDDIT_USERS, 'r') as inp: 
        data = json.load(inp)
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    d = {}
    for sub in data:
        if sub in subreddits:
            d[sub] = set(data[sub])
    # assign node IDs to subs
    sorted_subs = sorted(subreddits)
    sub_id = {} 
    for i, sub in enumerate(sorted_subs):
        sub_id[sub] = i
    # write node file
    nodes_out = open(NODES, 'w')
    nodes_out.write('Id\tLabel\tWeight\n')
    for sub in d:
        nodes_out.write(str(sub_id[sub]) + '\t' + sub + '\t' \
                        + str(len(d[sub])) + '\n')
    nodes_out.close()  
    # write edges file
    pairs = itertools.combinations(d.keys(), 2)
    edges_out = open(EDGES, 'w')
    edges_out.write('Source\tTarget\tWeight\n')
    for p in pairs: 
        weight = len(d[p[0]]&d[p[1]])/float(len(d[p[0]]|d[p[1]]))
        edges_out.write(str(sub_id[p[0]]) + '\t' + \
                        str(sub_id[p[1]]) + '\t' + \
                        str(weight) + '\n')
    edges_out.close()

def subreddit_docs():
    '''
    '''
    start = time.time()
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    subreddit_contents = defaultdict(list)
    with open(POSTS, 'r') as post_file:
        for line in post_file:
            post = json.loads(line)
            if 'subreddit' in post.keys():
                # promoted posts don't have subreddits
                subreddit = post['subreddit']
                if subreddit in subreddits:
                    title = post['title']
                    title = re.sub(r'[^\w\s]','',title)
                    title = title.lower().split()
                    subreddit_contents[subreddit].extend(title)
    end = time.time()
    print "TIME getting titles:", end-start
    with open(SUBREDDIT_TITLES, 'w') as titles_file:
        json.dump(subreddit_contents, titles_file)
        
def tf_idf():
    with open(SUBREDDIT_TITLES, 'r') as titles_file:
        subreddit_contents = json.load(titles_file)
    tfidf = TfidfVectorizer(norm='l2', stop_words='english', \
                            sublinear_tf=True)
    all_documents = []
    sorted_subs = sorted(subreddit_contents.keys())
    for i, subreddit in enumerate(sorted_subs):
        contents = ' '.join(subreddit_contents[subreddit])
        all_documents.append(contents)
    vectors = tfidf.fit_transform(all_documents)
    svd = TruncatedSVD(n_components=500, random_state=42)
    new_vectors = svd.fit_transform(vectors)
    # save the vectors
    res = {}
    for i, v in enumerate(new_vectors):
        res[sorted_subs[i]] = v.tolist()
    with open(TF_IDF, 'w') as tfidf_file:
        json.dump(res, tfidf_file)
        
def train_doc2vec(): 
    with open(SUBREDDIT_TITLES, 'r') as titles_file:
        subreddit_contents = json.load(titles_file)
    documents = []
    for subreddit in subreddit_contents:
        doc = TaggedDocument(words=subreddit_contents[subreddit],\
                             tags=[subreddit])
        documents.append(doc)
    model = Doc2Vec(size=50, window=8, min_count=5, workers=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=20)
    model.save(SUBREDDIT_MODEL)
    
def create_topic_graph(doc2vec=False, jaccard=False):
    with open(SUBREDDIT_USERS, 'r') as inp: 
        data = json.load(inp)
    subreddits = set()
    with open(TOP_100, 'r') as top: 
        for line in top: 
            subreddits.add(line.strip())
    subreddits.remove('fatpeoplehate')
    d = {}
    for sub in data:
        if sub in subreddits:
            d[sub] = set(data[sub])
    # assign node IDs to subs
    sorted_subs = sorted(subreddits)
    sub_id = {} 
    for i, sub in enumerate(sorted_subs):
        sub_id[sub] = i
    # write node file
    nodes_out = open(NODES2, 'w')
    nodes_out.write('Id\tLabel\tWeight\n')
    for sub in d:
        nodes_out.write(str(sub_id[sub]) + '\t' + sub + '\t' \
                        + str(len(d[sub])) + '\n')
    nodes_out.close() 
    # write edges file
    if doc2vec:
        model = Doc2Vec.load(SUBREDDIT_MODEL)
    elif jaccard: 
        with open(SUBREDDIT_TITLES, 'r') as titles_file:
            subreddit_contents = json.load(titles_file)
        stop = set(stopwords.words('english'))
    else:
        with open(TF_IDF, 'r') as tfidf_file:
            tfidf = json.load(tfidf_file)
    pairs = itertools.combinations(d.keys(), 2)
    edges_out = open(EDGES2, 'w')
    edges_out.write('Source\tTarget\tWeight\n')
    for p in pairs: 
        if doc2vec:
            vector1 = model.docvecs[p[0]]
            vector2 = model.docvecs[p[1]]
        elif not jaccard:
            vector1 = tfidf[p[0]]
            vector2 = tfidf[p[1]]
        if not jaccard:
            weight = cosine(vector1, vector2)
        else:
            set1 = set(subreddit_contents[p[0]])-stop
            set2 = set(subreddit_contents[p[1]])-stop
            weight = len(set1&set2)/float(len(set1|set2))
        edges_out.write(str(sub_id[p[0]]) + '\t' + \
                        str(sub_id[p[1]]) + '\t' + \
                        str(weight) + '\n')
    edges_out.close()

def topic_graph(): 
    """
    Subreddits by topic similarity 
    Maybe post title similarity
    """
    #subreddit_docs()
    #tf_idf()
    #train_doc2vec()
    create_topic_graph(jaccard=True)

def main():
    #generate_user_sets()
    #user_graph()
    topic_graph()

if __name__ == "__main__":
    main()
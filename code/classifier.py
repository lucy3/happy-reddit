"""
Classifier
"""
import json
import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from empath import Empath
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

DATA = "RANK"
if DATA == "GILDS":
    GILDS_BALANCED = "../logs/comment_gilds_classifier.json"
    SOCIAL_VECTORS = "/dfs/scratch1/jmendels/happy-reddit/logs/gilds_classifier_features/social_features/"
    LIWC_VECTORS = "../logs/gild_liwc_vectors/"
    LEXICAL_VECTORS = "../logs/gild_lexical_vectors/"
    RESULTS = "../results/gilds_classifier.txt"
elif DATA == "RANK":
    GILDS_BALANCED = "../logs/comment_rank_classifier.json"
    SOCIAL_VECTORS = "/dfs/scratch1/jmendels/happy-reddit/logs/rank_classifier_features/social_features/"
    LIWC_VECTORS = "../logs/rank_liwc_vectors/"
    LEXICAL_VECTORS = "../logs/rank_lexical_vectors/"
    RESULTS = "../results/rank_classifier.txt"
LIWC = "/dfs/scratch1/lucy3/twitter-relationships/data/en_liwc.txt"

def svc_param_selection(X, y, nfolds):
    # for tuning SVM 
    # best was {'loss': 'hinge', 'C': 18, 'tol': 0.5}
    # range for C: [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # range for tols: [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
    # losses: ['hinge','squared_hinge']
    tols = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
    param_grid = {'tol' : tols}
    grid_search = GridSearchCV(LinearSVC(loss='hinge', C=18), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

def rf_param_selection(X, y, nfolds):
    # for tuning Random Forest
    # BEST: {'n_estimators': 500, 'min_samples_leaf': 5}
    # leaves: [1, 3, 5, 7, 9]
    # estimators: [100, 200, 300, 400, 500, 600]
    leaves = [1, 3, 5, 7]
    estimators = [300, 400, 500, 600]
    param_grid = {'n_estimators': estimators, 'min_samples_leaf' : leaves}
    grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

def get_liwc_names():
    """
    Parses LIWC file
    return:
    - words with suffixed endings, other words
    - list of liwc group names
    """
    liwc = open(LIWC, 'r')
    liwc_names = [] 
    for line in liwc:
        l = line.split()
        g = l[0]
        liwc_names.append(g)
    return liwc_names

def get_feature_names():
    social_names = ['status','parent_pop','sub_loyalty',\
                    'user_loyalty','time_past','distance']
    liwc_names = get_liwc_names()
    relevance_names = ['post_rel', 'parent_rel']
    style_name = ['subreddit_prob']
    lexicon = Empath()
    empath_names = sorted(lexicon.cats.keys())
    brown_names = ['unigram_distinct', 'bigram_distinct']
    return liwc_names + social_names + relevance_names + \
        style_name + empath_names + brown_names

def get_features(): 
    '''
    @return
    - dictionary of ID: vector 
    '''
    with open(GILDS_BALANCED, 'r') as input_file:
        gilds = json.load(input_file)
    sorted_gilds = sorted(gilds.keys())
    X = []
    for comment in sorted_gilds:
        social = np.load(SOCIAL_VECTORS+comment+'.npy')
        liwc = np.load(LIWC_VECTORS+comment+'.npy')
        lexical = np.load(LEXICAL_VECTORS+comment+'.npy')
        vec = np.concatenate((liwc, social, lexical))
        X.append(vec)
    return np.array(X)
    
def get_labels(popularity=False):
    '''
    @input 
    - 
    @return
    - dictionary of ID: label
    '''
    labels = []
    with open(GILDS_BALANCED, 'r') as input_file:
        gilds = json.load(input_file)
    sorted_gilds = sorted(gilds.keys())
    for comment in sorted_gilds:
        labels.append(gilds[comment])
    return np.array(labels)
    
def split(X, y):
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    if DATA == "GILDS":
        gild_idx = np.where(y == 1)[0]
        nongild_idx = np.where(y == 0)[0]
        y_train = np.concatenate((np.ones(9000), np.zeros(9000)))
        y_test = np.concatenate((np.ones(len(gild_idx) - 9000), \
                                np.zeros(len(nongild_idx) - 9000)))
        X_train = np.concatenate((np.take(X, gild_idx[:9000], axis=0), \
                                 np.take(X, nongild_idx[:9000], axis=0)), axis=0)
        X_test = np.concatenate((np.take(X, gild_idx[9000:], axis=0), \
                                np.take(X, nongild_idx[9000:], axis=0)), axis=0)
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
    elif DATA == "RANK":
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                           test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test

def main():
    features = get_features()
    labels = get_labels()
    feature_names = get_feature_names()
    print "Done getting features"
    X, y = shuffle(features, labels, random_state=0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = split(X, y)
    print "Done splitting data"
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    out = open(RESULTS, 'w')
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, 
                                 random_state=0, n_jobs=-1) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print >> out,"RF Accuracy:", accuracy_score(y_test, y_pred)
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(y_test, y_pred, \
                                          labels=[True, False], average='macro')
    print >> out,"p macro:", p_macro, "\nr macro:", \
        r_macro, "\nf macro:", f_macro, "\nsupport macro", support_macro
    print >> out,"Features sorted by their score for Random Forest:"
    print >> out,sorted(zip(map(lambda x: round(x, 4), 
                clf.feature_importances_), feature_names), 
                         reverse=True)
                         
    clf = LinearSVC(loss='hinge', C=18, tol=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print >> out,"SVM Accuracy:", accuracy_score(y_test, y_pred)
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(y_test, y_pred, \
                                          labels=[True, False], average='macro')
    print >> out,"p macro:", p_macro, "\nr macro:", \
        r_macro, "\nf macro:", f_macro, "\nsupport macro", support_macro
    print >> out,"Features sorted by their score for LinearSVC:"
    print >> out, sorted(zip(map(lambda x: round(x, 4), 
                clf.coef_[0]), feature_names), 
                         reverse=True)
    
if __name__ == '__main__':
    main()

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

COM_ONLY = True # should always be False if DATA is Rank
DATA = "GILDS"
if DATA == "GILDS":
    GILDS_BALANCED = "../logs/comment_gilds_classifier.json"
    SOCIAL_VECTORS = "/dfs/scratch1/jmendels/happy-reddit/logs/gilds_classifier_features/social_features/"
    LIWC_VECTORS = "../logs/gild_liwc_vectors/"
    LEXICAL_VECTORS = "../logs/gild_lexical_vectors/"
    RESULTS = "../results/gilds_classifier_empathless.txt"
elif DATA == "RANK":
    GILDS_BALANCED = "../logs/comment_rank_classifier.json"
    SOCIAL_VECTORS = "/dfs/scratch1/jmendels/happy-reddit/logs/rank_classifier_features/social_features/"
    LIWC_VECTORS = "../logs/rank_liwc_vectors/"
    LEXICAL_VECTORS = "../logs/rank_lexical_vectors/"
    RESULTS = "../results/rank_classifier_empathless.txt"
if COM_ONLY: 
    RESULTS = "../results/gilds_classifier_communities_empathless.txt"
LIWC = "/dfs/scratch1/lucy3/twitter-relationships/data/en_liwc.txt"
COMMUNITY = "../results/communities.txt"

def svc_param_selection(X, y, nfolds):
    # for tuning SVM 
    # best was {'loss': 'hinge', 'C': 14, 'tol': 0.5}
    # range for C: [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # range for tols: [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
    # losses: ['hinge','squared_hinge']
    Cs = [8, 10, 12, 14]
    tols = [0.005, 0.01, 0.05, 0.1]
    param_grid = {'C' : Cs, 'tol' : tols}
    grid_search = GridSearchCV(LinearSVC(loss='hinge'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

def rf_param_selection(X, y, nfolds):
    # for tuning Random Forest
    # BEST: {'n_estimators': 500, 'min_samples_leaf': 5
    # 'max_features': auto, 'min_samples_split': 3}
    # leaves: [1, 3, 5, 7, 9]
    # estimators: [100, 200, 300, 400, 500, 600]
    # min_samples_split: [2, 3, 4, 5]
    # max_features: ['auto', 'log2']
    ests = [200, 300, 400]
    splits = [2, 3, 4, 5]
    leaves = [1, 3, 5]
    param_grid = {'min_samples_split': splits, 'n_estimators': ests, 'min_samples_leaf': leaves}
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
    liwc_names = ['length'] + get_liwc_names()
    social_names = ['status','parent_pop','sub_loyalty',\
                    'user_loyalty','time_past','distance']
    lexical_names = ['post_rel', 'parent_rel', 'subreddit_prob', \
                     'unigram_distinct', 'bigram_distinct']
    return social_names + liwc_names + lexical_names

def get_features(com=None): 
    '''
    @return
    - dictionary of ID: vector 
    '''
    with open(GILDS_BALANCED, 'r') as input_file:
        gilds = json.load(input_file)
    sorted_gilds = sorted(gilds.keys())
    X = []
    if not com: 
        for comment in sorted_gilds:
            liwc = np.load(LIWC_VECTORS+comment+'.npy') # 65
            social = np.load(SOCIAL_VECTORS+comment+'.npy') # 6
            lexical = np.load(LEXICAL_VECTORS+comment+'.npy') # 5
            vec = np.concatenate((social, liwc, lexical))
            X.append(vec)
    else:
        for comment in sorted_gilds:
            items = comment.split('_')
            subreddit = '_'.join(items[:-2])
            if subreddit in com: 
                social = np.load(SOCIAL_VECTORS+comment+'.npy')
                liwc = np.load(LIWC_VECTORS+comment+'.npy')
                lexical = np.load(LEXICAL_VECTORS+comment+'.npy')
                vec = np.concatenate((social, liwc, lexical))
                X.append(vec)
    return np.array(X)
    
def get_labels(com=None):
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
    if not com: 
        for comment in sorted_gilds:
            labels.append(gilds[comment])
    else:
        for comment in sorted_gilds:
            items = comment.split('_')
            subreddit = '_'.join(items[:-2])
            if subreddit in com: 
                labels.append(gilds[comment])
    return np.array(labels)
    
def split(X, y, com):
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    if DATA == "GILDS":
        gild_idx = np.where(y == 1)[0]
        nongild_idx = np.where(y == 0)[0]
        if not com: 
            y_train = np.concatenate((np.ones(8842), np.zeros(8842)))
            y_test = np.concatenate((np.ones(len(gild_idx) - 8842), \
                                    np.zeros(len(nongild_idx) - 8842)))
            X_train = np.concatenate((np.take(X, gild_idx[:8842], axis=0), \
                                     np.take(X, nongild_idx[:8842], axis=0)), axis=0)
            X_test = np.concatenate((np.take(X, gild_idx[8842:], axis=0), \
                                    np.take(X, nongild_idx[8842:], axis=0)), axis=0)
        else:
            y_train = np.concatenate((np.ones(765), np.zeros(765)))
            y_test = np.concatenate((np.ones(135), \
                                    np.zeros(675)))
            X_train = np.concatenate((np.take(X, gild_idx[:765], axis=0), \
                                     np.take(X, nongild_idx[:765], axis=0)), axis=0)
            X_test = np.concatenate((np.take(X, gild_idx[765:765+135], axis=0), \
                                    np.take(X, nongild_idx[765:765+675], axis=0)), axis=0)
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
    elif DATA == "RANK":
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                           test_size=0.15, random_state=0)
    return X_train, X_test, y_train, y_test

def do_classification(out, com=None):
    """
    @inputs
        - com: a set of subreddit names 
    """
    feature_names = get_feature_names()
    features = get_features(com)
    labels = get_labels(com)
    print "Done getting features"
    X, y = shuffle(features, labels, random_state=0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = split(X, y, com)
    print "Done splitting data"
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape, len(feature_names)
    
    print rf_param_selection(X_train, y_train, 5)
    '''
    if com: 
        print >> out, com
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, 
                                 random_state=0, min_samples_split=3, n_jobs=-1) 
    else:
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, 
                                 random_state=0, min_samples_split=3, n_jobs=-1) 
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
    if com:                     
        clf = LinearSVC(loss='hinge', C=8, tol=0.05, random_state=0)
    else:
        clf = LinearSVC(loss='hinge', C=14, tol=0.01, random_state=0)
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
    '''
def main():
    out = open(RESULTS, 'w')
    if not COM_ONLY: 
        do_classification(out)
    else:
        communities = {}
        with open(COMMUNITY, 'r') as community_file:
            for line in community_file:
                contents = line.split()
                communities[contents[0]] = set(contents[1:])
        for comm in communities: 
            print comm
            do_classification(out, communities[comm])
            break
    out.close()
    
if __name__ == '__main__':
    main()

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

ROC_OUT = '../logs/roc_plot'
GILDS_BALANCED = "../logs/comment_gilds_balanced.json"
SOCIAL_VECTORS = "/dfs/scratch1/jmendels/happy-reddit/logs/gilded_samples_features/social_features/"
LIWC_VECTORS = "../logs/liwc_vectors/"
LEXICAL_VECTORS = "../logs/lexical_vectors/"
LIWC = "/dfs/scratch1/lucy3/twitter-relationships/data/en_liwc.txt"

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
    social_names = ['status','parent_pop','sub_loyalty','user_loyalty','time_past']
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
    
def plot_roc(clf, X, y):
    '''
    Copied from sklearn documentation: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    '''
    cv = StratifiedKFold(n_splits=5)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('ROC')
    plt.savefig(ROC_OUT+'.png')
    plt.close()

def main():
    features = get_features()
    labels = get_labels()
    feature_names = get_feature_names()
    print "Done getting features"
    # TODO: sort by ID, put features and labels into numpy arrays
    X, y = shuffle(features, labels, random_state=0)
    
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, 
                                 random_state=0, n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=5, n_jobs=-1)
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(y, y_pred, \
                                          labels=[True, False], average='macro')
    print "p macro:", p_macro, "\nr macro:", \
        r_macro, "\nf macro:", f_macro, "\nsupport macro", support_macro
    #plot_roc(clf, X, y)
    clf.fit(X, y)
    print "Features sorted by their score for Random Forest:"
    print sorted(zip(map(lambda x: round(x, 4), 
                clf.feature_importances_), feature_names), 
                         reverse=True)

if __name__ == '__main__':
    main()
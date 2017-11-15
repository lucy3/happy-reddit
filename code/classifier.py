"""
Classifier
"""

ROC_OUT = '../logs/roc_plot'

def get_lexical_features():
    '''
    Load lexical features
    '''
    pass

def get_features(): 
    '''
    @return
    - dictionary of ID: vector 
    '''
    get_lexical_features()
    
def get_labels(popularity=False):
    '''
    @input 
    - 
    @return
    - dictionary of ID: label
    '''
    pass
    
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
    # TODO: sort by ID, put features and labels into numpy arrays
    X, y = shuffle(X, y, random_state=0)
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, 
                                 random_state=0, n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=5, n_jobs=-1)
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(y, y_pred, \
                                          labels=[True, False], average='macro')
    print "p macro:", p_macro, "\nr macro:", \
        r_macro, "\nf macro:", f_macro, "\nsupport macro", support_macro
    plot_roc(clf, X, y)
    clf.fit(X, y)
    print "Features sorted by their score:"
    print sorted(zip(map(lambda x: round(x, 4), 
                clf.feature_importances_), feature_names), 
                         reverse=True)

if __name__ == '__main__':
    main()
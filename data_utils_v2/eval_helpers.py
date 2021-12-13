"""
evaluation
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def evalROC(gold_scores, pred_scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(gold_scores, pred_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", roc_auc)


def evalPR(gold_scores, pred_scores):
    from sklearn.metrics import precision_recall_curve, auc
    prec, recall, _ = precision_recall_curve(gold_scores, pred_scores, pos_label=1)
    pr_auc = auc(recall, prec)
    print("pr_auc:", pr_auc)


def tuneThreshold(gold_scores, pred_scores):
    from  sklearn.metrics import f1_score
    best_t = 0.0
    best_fscore = 0.0
    for t in np.arange(0, 1.1, 0.1):
        pred_labels = [int(s > t) for s in pred_scores]
        fscore = f1_score(gold_scores, pred_labels)
        if (best_fscore < fscore):
            best_fscore = fscore
            best_t = t
    return best_t, best_fscore


def evalFscore(train_gold_scores, train_pred_scores, test_gold_scores, test_pred_scores):
    from sklearn.metrics import f1_score
    # threshold from train data
    threshold, _ = tuneThreshold(train_gold_scores, train_pred_scores)
    test_pred_labels = [int(s > threshold) for s in test_pred_scores]
    fscore = f1_score(test_gold_scores, test_pred_labels)
    print("fscore: {}".format(fscore))


def plot_roc_auc(y_test, y_score):

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)



    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC CURVES")
    plt.legend(loc="lower right")
    plt.show()
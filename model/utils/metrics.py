'''
This script has been either partially or fully inspired by the repository:

https://github.com/ireydiak/anomaly_detection_NRCAN

Alvarez, M., Verdier, J.C., Nkashama, D.K., Frappier, M., Tardif, P.M.,
Kabanza, F.: A revealing large-scale evaluation of unsupervised anomaly
detection algorithms
'''

from random import shuffle

import numpy as np
from sklearn import metrics as sk_metrics


def estimate_optimal_threshold(test_score, y_test, pos_label=1, nq=100, val_ratio=.2):
    # Generate indices the testscore
    n = len(test_score)
    idx = list(range(n))
    #shuffle(idx)
    idx = np.array(idx)

    # split score in test and validation
    n_test = int(n * (1 - val_ratio))

    # score_t = test_score[idx[:n_test]]
    # y_t = y_test[idx[:n_test]]
    # score_v = test_score[idx[n_test:]]
    # y_v = y_test[idx[n_test:]]

    idx_out = y_test == 1
    idx_in = y_test == 0
    y_test_in = y_test[idx_in]
    y_test_out = y_test[idx_out]
    test_score_in = test_score[idx_in]
    test_score_out = test_score[idx_out]
    #print(len(idx_out), len(idx_in), len(y_test), len(test_score), len(y_test_in), len(y_test_out))
    #print(np.concatenate([y_test[idx_out][:int(len(idx_out) * (1 - val_ratio))], y_test[idx_in][:int(len(idx_in) * (1 - val_ratio))]]))
    y_t = np.concatenate([y_test_out[:int(len(y_test_out) * (1 - val_ratio))], y_test_in[:int(len(y_test_in) * (1 - val_ratio))]])
    y_v = np.concatenate([y_test_out[int(len(y_test_out) * (1 - val_ratio)):], y_test_in[int(len(y_test_in) * (1 - val_ratio)):]])
    score_t = np.concatenate([test_score_out[:int(len(test_score_out) * (1 - val_ratio))], test_score_in[:int(len(test_score_in) * (1 - val_ratio))]])
    score_v = np.concatenate([test_score_out[int(len(test_score_out) * (1 - val_ratio)):], test_score_in[int(len(test_score_in) * (1 - val_ratio)):]])

    ratio_v = 100 * sum(y_v == 0) / len(y_v)
    print(f"Ratio of normal val data:{ratio_v}", flush=True)

    ratio_t = 100 * sum(y_t == 0) / len(y_t)
    print(f"Ratio of normal test data:{ratio_t}", flush=True)

    # Estimate the threshold on the validation set
    res = estimate_optimal_threshold_legacy(score_v, y_v, pos_label, nq)
    threshold = res["Thresh_star"]

    # Compute metrics on the test set
    metrics = compute_metrics(score_t, y_t, threshold, pos_label)

    return {
        "Precision": metrics[1],
        "Recall": metrics[2],
        "F1-Score": metrics[3],
        "AUPR": metrics[5],
        "AUROC": metrics[4],
        "Thresh_star": threshold,
        # "Quantile_star": qis
    }


def compute_metrics(test_score, y_test, thresh, pos_label=1):
    """
    This function compute metrics for a given threshold

    Parameters
    ----------
    test_score
    y_test
    thresh
    pos_label

    Returns
    -------

    """
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm


def estimate_optimal_threshold_legacy(test_score, y_test, pos_label=1, nq=100):
    ratio = 100 * sum(y_test == 0) / len(y_test)
    #print(f"Ratio of normal data:{ratio}")
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(test_score, y_test, thresh, pos_label)

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        # print(f"qi:{qi:.3f} ==> p:{precision:.3f}  r:{recall:.3f}  f1:{f_score:.3f}")
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)
    # print("BEST THRESHOLD", arm, q, thresholds)
    ret = {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }
    print(ret)
    return ret


def score_recall_precision_w_threshold(scores, y_true, threshold=None, pos_label=1):
    anomaly_ratio = (y_true == pos_label).sum() / len(y_true)
    threshold = threshold or int(np.ceil((1 - anomaly_ratio) * 100))
    thresh = np.percentile(scores, threshold)

    # Prediction using the threshold value
    y_pred = (scores >= thresh).astype(int)
    y_true = y_true.astype(int)

    # accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )

    return {"Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "AUROC": sk_metrics.roc_auc_score(y_true, scores),
            "AUPR": sk_metrics.average_precision_score(y_true, scores),
            "Thresh": thresh
            }, y_pred

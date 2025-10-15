

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metrics_binary(targets,probs,threshold = 0.5):
    '''
    :param targets:
    :param probs:
    :param threshold:
    :return:
    '''
    if targets.ndim == 2:
        if targets.shape[1] == 2:
            targets = np.argmax(targets,axis = 1)
        else:
            targets = targets.flatten()


    if probs.ndim == 2:
        if probs.shape[1] == 2:
            preds = np.argmax(probs,axis = 1)
        else:
            preds = (probs >= threshold).astype(int).flatten()

    else:
        preds = (probs >= threshold).astype(int)


    acc = accuracy_score(targets,preds)
    prec = precision_score(targets,preds)
    recall = recall_score(targets,preds)
    f1 = f1_score(targets,preds)

    result = {
        "acc":round(float(acc),6),
        "prec": round(float(prec), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)


    }

    return result










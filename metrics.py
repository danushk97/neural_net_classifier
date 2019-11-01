import numpy as np
import pandas as pd
 
def f1_score(a, y):
    actuals = pd.Series(np.squeeze(y))
    preds = pd.Series(np.squeeze(a))
    
    act_positive_index = actuals[actuals == 1].index
    act_negative_index = actuals[actuals == 0].index
    
    true_positive = np.sum(actuals[act_positive_index] == preds[act_positive_index])
    true_negative = np.sum(actuals[act_negative_index] == preds[act_negative_index])
    false_positive = np.sum(actuals[act_negative_index] != preds[act_negative_index])
    false_negative = np.sum(actuals[act_positive_index] != preds[act_positive_index])
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    score = 2 * ((precision * recall) / (precision + recall))
    
    return score

def accuracy(a, y):

    acc = 100 - np.mean(np.abs(a - y)) * 100

    return acc

def probability_to_preds(a, threshold):

        preds = a >= threshold

        return preds.astype(int)
        
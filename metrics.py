import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    true_neg = np.sum((y_pred == 0) & (y_true == 0))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1 = 2*(precision*recall/(precision+recall))
    
    return(accuracy, precision, recall, f1)
    

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score



def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    true_pos = np.sum(y_pred == y_true)
    
    accuracy = true_pos/len(y_true)
        
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
   
    y_mean = np.mean(y_true)
    
    rsq = 1 - (sum((y_true-y_pred)**2)/sum((y_true-y_mean)**2))
    
    return rsq


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    absolute_errors = np.abs(y_pred - y_true)
    mae = np.mean(absolute_errors)
    
    return mae
    
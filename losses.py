import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def Loglike_loss(y_true, y_pred, n_intervals=5):
    '''
    y_true: Tensor.
        First half: 1 if individual survived that interval, 0 if not.
        Second half: 1 for time interval before which failure has occured, 0 for other intervals.
    y_pred: Tensor.
        Predicted survival probability (1-hazard probability) for each time interval.
    '''

    cens_uncens = torch.clamp(
        1.0 + y_true[:, 0:n_intervals]*(y_pred-1.0), min=1e-5)
    uncens = torch.clamp(
        1.0 - y_true[:, n_intervals:2*n_intervals]*y_pred, min=1e-5)
    loss = -torch.mean(torch.log(cens_uncens) + torch.log(uncens))

    return loss


def L2_Regu_loss(weights, alpha=0.1):
    '''
    Loss for L2 Regularization on weights
    '''

    loss = torch.square(weights).sum()

    return alpha * loss


def NegativeLogLikelihood(risk_pred, y, e):
    mask = torch.ones(y.shape[0], y.shape[0]).cuda()
    mask[(y.T - y) > 0] = 0
    log_loss = torch.exp(risk_pred) * mask
    log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
    log_loss = torch.log(log_loss).reshape(-1, 1)
    neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
    return neg_log_loss


def cox_loss(score, time_value, event):
    score = - score
    ix = torch.where(event > 0)[0]
    sel_mat = (time_value.gather(0, ix.unsqueeze(1)
                                 ).squeeze(1) <= time_value).float()
    p_lik = torch.clamp(score.gather(0, ix.unsqueeze(1)).squeeze(
        1) - torch.log((sel_mat * torch.exp(score))), min=1e-5)

    p_lik[p_lik.isnan()] = 0.
    p_lik[p_lik.isinf()] = 1.

    loss = -torch.sum(p_lik * sel_mat)/(sel_mat.sum() + 1e-10)
    return loss


def PartialLogLikelihood(logits, fail_indicator, ties='noties'):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return -1.0*logL / observations


def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
            x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
            indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return(indicator_matrix)


def neg_par_log_likelihood(pred, ytime, yevent):
    '''Calculate the average Cox negative partial log-likelihood.
    Input:
            pred: linear predictors from trained model.
            ytime: true survival time from load_data().
            yevent: true censoring status from load_data().
    Output:
            cost: the cost that is to be minimized.
    '''
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    # if gpu is being used
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    ###
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (sum_diff_in_observed / n_observed).reshape((-1,))

    return cost

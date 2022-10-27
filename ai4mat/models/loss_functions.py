import torch

def MSELoss(y, preds, weights, reduction):
    '''
        MSE loss
    '''
    if reduction == 'mean':
        return ((y - preds) ** 2).mean()
    elif reduction == 'sum':
        return ((y - preds) ** 2).sum()
    else:
        raise ValueError


def MAELoss(y, preds, weights, reduction):
    '''
        MAE loss
    '''
    if reduction == 'mean':
        return torch.abs(y - preds).mean()
    elif reduction == 'sum':
        return torch.abs(y - preds).sum()
    else:
        raise ValueError


def weightedMSELoss(y, preds, weights, reduction):
    '''
    weights must sum up to length of dataset
    '''
    if reduction == 'mean':
        return (weights * (y - preds) ** 2).mean()
    elif reduction == 'sum':
        return (weights * (y - preds) ** 2).sum()
    else:
        raise ValueError


def weightedMAELoss(y, preds, weights, reduction):
    '''
    weights must sum up to length of dataset
    '''
    if reduction == 'mean':
        return (weights * torch.abs(y - preds)).mean()
    elif reduction == 'sum':
        return (weights * torch.abs(y - preds)).sum()
    else:
        raise ValueError

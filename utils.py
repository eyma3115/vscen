import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import TensorDataset, DataLoader

#GradientReversal source code: van Vugt, Joris. Pytorch Adversarial Domain Adaptation. https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
class GradientReversalFunction(Function):

    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def pbcorrelation(x, a):
    control = x[a==0]
    treat = x[a==1]

    m0 = torch.mean(control, dim=0)
    m1 = torch.mean(treat, dim=0)
    s = torch.std(x, dim=0)

    r = ((m1-m0)/s)*torch.sqrt(torch.Tensor([(control.shape[0]*treat.shape[0])/(x.shape[0]**2)]))

    return r


def outcome_loss(y_hat, y):
    return F.mse_loss(y_hat, y, reduction='mean')


def treat_loss(a_hat, a):
    return F.binary_cross_entropy(a_hat.float(), a.float(), reduction='mean')


def FDR(feat_selected, coef):
    feat_selected = np.unique(np.sort(feat_selected))
    true_feat = np.nonzero(coef)
    false_feat = np.delete(np.arange(0, 200), true_feat)

    f = 0
    for idx in feat_selected:
        if idx in false_feat:
            f += 1

    return f / feat_selected.shape[0]

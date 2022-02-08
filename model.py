import numpy as np
import os
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import TensorDataset, DataLoader
from utils import *


class ConcreteSelector(nn.Module):
    def __init__(self, x_dim, var_num, n_epochs, xa_corr, start_temp=10, min_temp=0.1, corr_weight=2, start_point=0.5):
        super(ConcreteSelector, self).__init__()
        self.x_dim = x_dim
        self.var_num = var_num
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.n_epochs = n_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.temperature = torch.Tensor([self.min_temp, self.start_temp])
        self.logits = nn.Parameter(torch.rand((self.var_num, self.x_dim), device='cuda'), requires_grad=True)
        self.feat_idx = torch.randint(0, x_dim, (var_num,))
        self.xa_corr = xa_corr.to(self.device)
        self.corr_weight = corr_weight
        self.start_point = start_point

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape)

        return torch.log(-torch.log(u + eps) + eps).to(self.device)

    def concrete(self, logits, temp=1, hard=False, dim=-1):
        m = logits + self.sample_gumbel(logits.size())
        m = F.softmax(m / temp, dim)
        if hard:
            idx = m.max(dim, keepdim=True)[1]
            m_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, idx, 1.0)
            m = m_hard - m.detach() + m
        return m

    def forward(self, x, epoch):
        self.temperature[1] = self.start_temp * ((self.min_temp / self.start_temp) ** (epoch / self.n_epochs))
        temp = torch.max(self.temperature)

        if self.training:
            if epoch < self.start_point * self.n_epochs:
                m = self.concrete(self.logits, temp, hard=False)
            else:
                corr = torch.zeros(x_dim, device='cuda')
                corr[self.feat_idx] = self.xa_corr[self.feat_idx]
                m = self.concrete(self.logits * torch.exp(self.corr_weight * torch.abs(corr)), temp, hard=False)
        else:
            if epoch < self.start_point * self.n_epochs:
                m = self.concrete(self.logits, temp, hard=True)
                self.feat_idx = torch.argmax(m, dim=1)
            else:
                corr = torch.zeros(x_dim, device='cuda')
                corr[self.feat_idx] = self.xa_corr[self.feat_idx]
                m = self.concrete(self.logits * torch.exp(self.corr_weight * torch.abs(corr)), temp, hard=True)
                self.feat_idx = torch.argmax(m, dim=1)

        selected = torch.matmul(x, m.transpose(0, 1))

        return selected, m


class CounterfactualPredictor(nn.Module):
    def __init__(self, xs_dim, a_dim, y_dim):
        super(CounterfactualPredictor, self).__init__()
        self.xs_dim = xs_dim
        self.a_dim = a_dim
        self.y_dim = y_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.representation = nn.Sequential(nn.Linear(self.xs_dim, 200),
                                            nn.ELU(),
                                            nn.Linear(200, 200))

        self.outcome_treat = nn.Sequential(nn.Linear(200 + self.a_dim, 100),
                                           nn.ELU(),
                                           nn.Linear(100, 100),
                                           nn.ELU(),
                                           nn.Linear(100, self.y_dim))

        self.outcome_control = nn.Sequential(nn.Linear(200 + self.a_dim, 100),
                                             nn.ELU(),
                                             nn.Linear(100, 100),
                                             nn.ELU(),
                                             nn.Linear(100, self.y_dim))

    def forward(self, xc, xp, a):
        rep = self.representation(torch.cat((xc, xp), dim=1))
        feat = torch.cat((rep, a), dim=1)
        y_treat_hat = self.outcome_treat(feat)
        y_control_hat = self.outcome_control(feat)
        y_obs_hat = (1 - a) * y_control_hat + a * y_treat_hat

        return y_obs_hat, y_treat_hat, y_control_hat


class AntiTreatmentPredictor(nn.Module):
    def __init__(self, xp_dim, a_dim):
        super(AntiTreatmentPredictor, self).__init__()

        self.xp_dim = xp_dim
        self.a_dim = a_dim

        self.treatment = nn.Sequential(GradientReversal(),
                                       nn.Linear(xp_dim, 100),
                                       nn.ELU(),
                                       nn.Linear(100, 100),
                                       nn.ELU(),
                                       nn.Linear(100, self.a_dim),
                                       nn.Sigmoid())

    def forward(self, xp, a):
        a_hat_soft = self.treatment(xp)
        a_hat_hard = torch.round(a_hat_soft)

        return a_hat_soft, a_hat_hard


class CSCR(nn.Module):
    def __init__(self, selector_c, selector_p, predictor_y, predictor_a):
        super(CSCR, self).__init__()
        self.selector_c = selector_c
        self.selector_p = selector_p
        self.predictor_y = predictor_y
        self.predictor_a = predictor_a
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, a, epoch):
        xc, mc = self.selector_c(x, epoch)
        xp, mp = self.selector_p(x, epoch)

        y_obs_hat, y_treat_hat, y_control_hat = self.predictor_y(xc, xp, a)
        a_hat_s, a_hat_h = self.predictor_a(xp, a)

        return y_obs_hat, y_treat_hat, y_control_hat, a_hat_s, a_hat_h, xc, xp, mc, mp


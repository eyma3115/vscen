import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import TensorDataset, DataLoader
from utils import *


def evaluate(model, test_loader, treat_coef):

    def RMSE(y_hat, y):
        return torch.sqrt(torch.mean((y_hat - y) ** 2))

    def PEHE(y_hat, y):
        effect = y[:, 1] - y[:, 0]
        effect_hat = y_hat[:, 1] - y_hat[:, 0]
        return torch.sqrt(torch.mean((effect_hat - effect) ** 2))

    def ATE(y_hat, y):
        effect = y[:, 1] - y[:, 0]
        effect_hat = y_hat[:, 1] - y_hat[:, 0]
        return torch.abs(torch.mean(effect_hat) - torch.mean(effect))

    def factLabel(a, mu0, mu1):
        return (1 - a) * (mu0) + a * mu1

    def cfLabel(a, mu0, mu1):
        return a * (mu0) + (1 - a) * mu1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_fact_label = torch.empty(0)
    y_cf_label = torch.empty(0)
    y_full_label = torch.empty(0)

    y_fact_pred = torch.empty(0)
    y_full_pred = torch.empty(0)

    a_label = torch.empty(0)
    a_pred = torch.empty(0)

    for i, (x, a, yf, ycf, mu0, mu1) in enumerate(test_loader):
        model.eval()

        x, a, yf, ycf, mu0, mu1 = x.to(device), a.to(device), yf.to(device), ycf.to(device), mu0.to(device), mu1.to(device)
        
        #Epoch set very high at test time to ensure minimum temperature
        y_obs_hat, y_treat_hat, y_control_hat, a_hat_s, a_hat_h, xc, xp, mc, mp = model(x, a, 10000)

        y_loss = outcome_loss(y_obs_hat.squeeze(), yf.squeeze())
        a_loss = treat_loss(a_hat_s.squeeze(), a.squeeze())

        loss = y_loss + treat_coef * a_loss

        y_fact_label = torch.cat((y_fact_label, factLabel(a, mu0, mu1).cpu()), dim=0)
        y_cf_label = torch.cat((y_cf_label, cfLabel(a, mu0, mu1).cpu()), dim=0)

        y_fact_pred = torch.cat((y_fact_pred, y_obs_hat.cpu()), dim=0)
        y_full_pred = torch.cat((y_full_pred, torch.cat((y_control_hat, y_treat_hat), dim=1).cpu()), dim=0)

        a_label = torch.cat((a_label, a.long().cpu()), dim=0)
        a_pred = torch.cat((a_pred, a_hat_h.long().cpu()), dim=0)

        y_full_label = torch.cat((y_full_label, torch.cat((mu0, mu1), dim=1).cpu()), dim=0)

        rmse = RMSE(y_fact_pred, y_fact_label)
        pehe = PEHE(y_full_pred, y_full_label)
        ate = ATE(y_full_pred, y_full_label)

    return model, y_fact_label, y_cf_label, y_full_label, y_fact_pred, y_full_pred, a_label, a_pred, xc, xp, mc, mp, rmse, pehe, ate

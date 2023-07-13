import numpy as np
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
        a_hat = self.treatment(xp)
        return a_hat


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
        a_hat = self.predictor_a(xp, a)

        return y_obs_hat, y_treat_hat, y_control_hat, a_hat, xc, xp, mc, mp


def train(model, n_epochs, s_optimizer, p_optimizer, s_scheduler, p_scheduler, train_loader, val_loader, treat_coef):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(n_epochs):

        s_optimizer.zero_grad()
        p_optimizer.zero_grad()

        train_loss = []
        train_a_loss = []
        train_y_loss = []

        val_loss = []
        val_a_loss = []
        val_y_loss = []

        for i, (x, a, yf, ycf, mu0, mu1) in enumerate(train_loader):
            model.train()

            x, a, yf, ycf, mu0, mu1 = x.to(device), a.to(device), yf.to(device), ycf.to(device), mu0.to(device), mu1.to(device)

            y_obs_hat, y_treat_hat, y_control_hat, a_hat, xc, xp, mc, mp = model(x, a, epoch)

            y_loss = outcome_loss(y_obs_hat.squeeze(), yf.squeeze())
            a_loss = treat_loss(a_hat.squeeze(), a.squeeze())

            loss = y_loss + treat_coef * a_loss

            loss.backward()
            s_optimizer.step()
            p_optimizer.step()
            s_scheduler.step()
            p_scheduler.step()

            train_loss.append(loss.item())
            train_a_loss.append(a_loss.item())
            train_y_loss.append(y_loss.item())

        for i, (x, a, yf, ycf, mu0, mu1) in enumerate(val_loader):
            model.eval()

            x, a, yf, ycf, mu0, mu1 = x.to(device), a.to(device), yf.to(device), ycf.to(device), mu0.to(device), mu1.to(device)

            y_obs_hat, y_treat_hat, y_control_hat, a_hat, xc, xp, mc, mp = model(x, a, epoch)

            y_loss = outcome_loss(y_obs_hat.squeeze(), yf.squeeze())
            a_loss = treat_loss(a_hat.squeeze(), a.squeeze())

            loss = y_loss + treat_coef * a_loss

            val_loss.append(loss.item())
            val_a_loss.append(a_loss.item())
            val_y_loss.append(y_loss.item())

        print("Epoch {}/{} Done, Train Loss: {:.4f}, Validation Loss: {:.4f}".format(epoch+1, n_epochs,
                                                                         sum(train_loss)/len(train_loss),
                                                                         sum(val_loss)/len(val_loss)))

    return model


def evaluate(model, test_loader, treat_coef):

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
        y_obs_hat, y_treat_hat, y_control_hat, a_hat, xc, xp, mc, mp = model(x, a, 10000)

        y_loss = outcome_loss(y_obs_hat.squeeze(), yf.squeeze())
        a_loss = treat_loss(a_hat.squeeze(), a.squeeze())

        loss = y_loss + treat_coef * a_loss

        y_fact_label = torch.cat((y_fact_label, factLabel(a, mu0, mu1).cpu()), dim=0)
        y_cf_label = torch.cat((y_cf_label, cfLabel(a, mu0, mu1).cpu()), dim=0)

        y_fact_pred = torch.cat((y_fact_pred, y_obs_hat.cpu()), dim=0)
        y_full_pred = torch.cat((y_full_pred, torch.cat((y_control_hat, y_treat_hat), dim=1).cpu()), dim=0)

        a_label = torch.cat((a_label, a.long().cpu()), dim=0)
        a_pred = torch.cat((a_pred, torch.round(a_hat).long().cpu()), dim=0)

        y_full_label = torch.cat((y_full_label, torch.cat((mu0, mu1), dim=1).cpu()), dim=0)

        pehe = PEHE(y_full_pred, y_full_label)
        ate = ATE(y_full_pred, y_full_label)

    return model, y_fact_label, y_cf_label, y_full_label, y_fact_pred, y_full_pred, a_label, a_pred, xc, xp, mc, mp, pehe, ate


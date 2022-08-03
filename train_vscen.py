import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import TensorDataset, DataLoader
from utils import *

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

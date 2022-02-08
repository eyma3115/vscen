import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from model import *
from train_vscen import *
from evaluate_vscen import *

#change index for other datasets
with open('./data/synthetic_0-10.pkl', 'rb') as f:
  synthetic = pickle.load(f)
#Example shown for replication0
data = {}
for k in synthetic.keys():
  if k == 'x':
    data[k] = synthetic[k][:,:,0]
  else:
    data[k] = synthetic[k][:,0]

train, val, test = split_data(data, train_size=0.63, val_size=0.27)

for dic in [train, val, test]:
    for k,v in dic.items():
        dic[k] = torch.Tensor(v)

x_dim = data['x'].shape[1]
y_dim = 1
a_dim = 1
n_epochs = 300
var_num_c = 5
var_num_p = 5
t = 2
c = 2
sp = 0.5

train_loader = DataLoader(dataset=TensorDataset(train['x'], train['a'].unsqueeze(dim=1), train['yf'].unsqueeze(dim=1), train['ycf'].unsqueeze(dim=1), 
                                                train['mu0'].unsqueeze(dim=1), train['mu1'].unsqueeze(dim=1)), batch_size=200, shuffle=True, drop_last=False)
val_loader = DataLoader(dataset=TensorDataset(val['x'], val['a'].unsqueeze(dim=1), val['yf'].unsqueeze(dim=1), val['ycf'].unsqueeze(dim=1), 
                                              val['mu0'].unsqueeze(dim=1), val['mu1'].unsqueeze(dim=1)), batch_size=200, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=TensorDataset(test['x'], test['a'].unsqueeze(dim=1), test['yf'].unsqueeze(dim=1), test['ycf'].unsqueeze(dim=1), 
                                               test['mu0'].unsqueeze(dim=1), sim_test['mu1'].unsqueeze(dim=1)), batch_size=200, shuffle=True, drop_last=True)

xa_corr = pbcorrelation(train['x'], train['a'])

selector_c = ConcreteSelector(x_dim, var_num_c, n_epochs, xa_corr, start_temp=10, min_temp=0.1, corr_weight=c, start_point=sp)
selector_p = ConcreteSelector(x_dim, var_num_p, n_epochs, xa_corr, start_temp=10, min_temp=0.1, corr_weight=0, start_point=1)
predictor_y = CounterfactualPredictor(var_num_c+var_num_p, a_dim, y_dim)
predictor_a = AntiTreatmentPredictor(var_num_p, a_dim)

model = CSCR(selector_c, selector_p, predictor_y, predictor_a)

s_param = list(model.selector_c.parameters())+list(model.selector_p.parameters())
p_param = list(model.predictor_y.parameters())+list(model.predictor_a.parameters())

s_optimizer = torch.optim.Adam(s_param, lr=0.01)
p_optimizer = torch.optim.Adam(p_param, lr=0.0001)
s_scheduler = torch.optim.lr_scheduler.StepLR(s_optimizer, step_size=100, gamma= 0.97)
p_scheduler = torch.optim.lr_scheduler.StepLR(p_optimizer, step_size=100, gamma= 0.97)

model, loss_dict = train(model, n_epochs, s_optimizer, p_optimizer, s_scheduler, p_scheduler, train_loader, val_loader, t)

model, y_fact_label, y_cf_label, y_full_label, y_fact_pred, y_full_pred, a_label, a_pred, xc, xp, mc, mp, pehe, ate = evaluate(model, train_loader, t)
print('-----------------------In-Sample-----------------------')
print("PEHE: {:.4f}, ATE: {:.4f}".format(pehe, ate))
model, y_fact_label, y_cf_label, y_full_label, y_fact_pred, y_full_pred, a_label, a_pred, xc, xp, mc, mp, pehe, ate = evaluate(model, test_loader, t)
print('-----------------------Out-Sample-----------------------')
print("PEHE: {:.4f}, ATE: {:.4f}".format(pehe, ate))

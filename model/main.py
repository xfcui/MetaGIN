"""Train / evaluate MetaGIN on PCQM4Mv2 (OGB-LSC).

Paper: https://link.springer.com/article/10.1007/s11704-024-3784-y
Run from this directory so ``data/`` and local imports resolve:

    python3 -BuW ignore main.py --model tiny311 --save .
"""

import re
import argparse
from time import time
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch as pt

from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from torch_geometric.loader import DataLoader
from adan import Adan

from model import MetaGIN
from optim import get_param, Scheduler


parser = argparse.ArgumentParser(
    description='MetaGIN on PCQM4Mv2 (https://link.springer.com/article/10.1007/s11704-024-3784-y)')
parser.add_argument('--model', type=str, default='tiny311',
                    help='preset: tiny311|base321|large321 (size+hop+kernel+virt; default: tiny311)')
parser.add_argument('--save', type=str, default='',
                    help='directory for modelXXX.pt checkpoints and OGB test-dev submission')
args = parser.parse_args()

if args.model.startswith('tiny'):
    model_config = {'depth': 4, 'num_head': 16, 'conv_kernel': [1, 1, 1, 1]}  # num_kernels: 1+4*1=5
elif args.model.startswith('base'):
    model_config = {'depth': 4, 'num_head': 16, 'conv_kernel': [1, 1, 3, 1]}  # num_kernels: 1+6*2=13
elif args.model.startswith('large'):
    model_config = {'depth': 4, 'num_head': 16, 'conv_kernel': [1, 4, 9, 1]}  # num_kernels: 1+15*2=31
else:
    assert False, args.model
model_name = re.sub('\D+', '', args.model)
assert len(model_name)==3 and 1<=int(model_name[0])<=3 and 1<=int(model_name[1]) and 0<=int(model_name[2])<=1, args.model
model_config['conv_hop'] = int(model_name[0])
for i in range(model_config['depth']):
    model_config['conv_kernel'][i] *= int(model_name[1])
model_config['use_virt'] = int(model_name[2])>0


lr_base, lr_min, wd_base = 3e-3, 1e-5, 2e-2
batch_size, cos_period, num_period = 256, 12, 12
print('#torch:', pt.__version__, pt.version.cuda)

from data import dataset, dataidx, dataeval
train_loader = DataLoader(dataset[dataidx["train"]], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
valid_loader = DataLoader(dataset[dataidx["valid"]], batch_size=batch_size*2, shuffle=False, drop_last=False, num_workers=6)
test_loader  = DataLoader(dataset[dataidx["test-dev"]], batch_size=batch_size*2, shuffle=False, drop_last=False, num_workers=6)
print('#loader:', batch_size, len(train_loader))

model = MetaGIN(**model_config).cuda()
param = get_param(model, lr_base, wd_base, lr_min)
optim = Adan(param, lr_base, weight_decay=wd_base)
sched = Scheduler(optim, cos_period//2, cos_period*2, cos_period)
print('#optim:', '%.2e'%lr_base, '%.2e'%wd_base, cos_period, num_period)


loss0_fn = pt.nn.L1Loss()
loss1_fn = pt.nn.SmoothL1Loss(beta=0.05)  # distance aux; clamp range matches KnnKernel
def train(model, loader, optim, param):
    """One training epoch; returns (gap L1, scaled distance aux)."""
    model.train()
    loss0_accum, loss1_accum = 0, 0

    optim.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        pred, dpred, dtrue = model(batch)
        with pt.no_grad():
            dpred.clamp_(0.5, 5.0)
            dtrue.clamp_(0.5, 5.0)
        loss0 = loss0_fn(pred.view(-1,), batch.y)
        loss1 = loss1_fn(dpred.log().view(-1,), dtrue.log().view(-1,))
        loss  = loss0 + loss1/4
        loss.backward()
        optim.step()
        with pt.no_grad(): model.clamp_()

        loss0_accum += loss0.detach().cpu().item()
        loss1_accum += loss1.detach().cpu().item()
        optim.zero_grad()

    return loss0_accum / (step + 1), loss1_accum / (step + 1) * 2

def eval(model, loader, evaluator):
    """Validation MAE via ``PCQM4Mv2Evaluator``."""
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        with pt.no_grad():
            pred = model(batch)
            y_true.append(batch.y.detach().cpu())
            y_pred.append(pred.view(-1,).detach().cpu())

    y_true = pt.cat(y_true, dim = 0)
    y_pred = pt.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, statefn, loader):
    """Load ``statefn`` and collect test-dev predictions for an OGB submission."""
    model_copy = deepcopy(model).cuda()
    model_copy.load_state_dict(pt.load(statefn))
    model_copy.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        with pt.no_grad():
            pred = model_copy(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = pt.cat(y_pred, dim = 0)

    return y_pred.cpu().detach().numpy()


print(); print('#training...')
best_mae, best_epoch, test_epoch, t0 = 9999, -1, -1, time()
for epoch in range(cos_period*num_period):
    epoch_lr, epoch_wd = sched.step(epoch)
    train_mae, loss_dist = train(model, train_loader, optim, param)
    valid_mae = eval(model, valid_loader, dataeval)
    eta = (time() - t0) / (epoch + 1) * (cos_period * num_period - epoch - 1) / 3600

    if valid_mae < best_mae:
        best_mae, best_epoch = valid_mae, epoch
        print('#epoch[%d]: %.4f %.4f %.4f %.2e %.2e %.1fh *' % (epoch, loss_dist, train_mae, valid_mae, epoch_lr, epoch_wd, eta))
    else:
        print('#epoch[%d]: %.4f %.4f %.4f %.2e %.2e %.1fh' % (epoch, loss_dist, train_mae, valid_mae, epoch_lr, epoch_wd, eta))

    if args.save != '':
        pt.save(model.state_dict(), args.save + '/model%03d.pt' % epoch)
        if epoch > cos_period and (epoch+1) % cos_period == 0 and best_epoch > test_epoch:
            test_fn, test_epoch = '%s/model%03d.pt' % (args.save, best_epoch), best_epoch
            test_pred = test(model, test_fn, test_loader)
            dataeval.save_test_submission({'y_pred': test_pred}, args.save, mode='test-dev')
            print('#saved[%d]: test-dev' % test_epoch)
print('#done!!!')


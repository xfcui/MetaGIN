import numpy as np
import torch as pt
import torch.nn.functional as nnf


def get_param(model, lr, wd, lr_min):
    param_groups = [{'params': [], 'lr_max': lr/2, 'lr_min': lr_min/2, 'wd_max': 0},     # 0:embed
                    {'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': 0},     # 1:scale with clamp
                    {'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': 0},     # 2:degree with clamp
                    {'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': 0},     # 3:bias without clamp
                    {'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': wd},    # 4:weight
                    {'params': [], 'lr_max': lr/2, 'lr_min': lr_min/2, 'wd_max': wd*2}]  # 5:head

    for n, p in model.named_parameters():
        if n.find('_encoder') > 0: param_groups[0]['params'].append(p)
        elif n.endswith('scale'):  param_groups[1]['params'].append(p)
        elif n.endswith('degree'): param_groups[2]['params'].append(p)
        elif n.endswith('bias'):   param_groups[3]['params'].append(p)
        elif n.find('head') > 0:   param_groups[5]['params'].append(p)
        elif n.endswith('weight'): param_groups[4]['params'].append(p)
        else: raise Exception('Unknown parameter name:', n)
        for pg in param_groups: assert len(pg) > 0

    return param_groups

def clamp_param(param, eps=1e-4):
    with pt.no_grad():
        for p in param[1]['params']:
            p.clamp_(np.log(eps), 0)
        for p in param[2]['params']:
            p.clamp_(-1, 0)

class Scheduler(object):
    def __init__(self, optim, lr_warmup=6, wd_warmup=12, cos_period=12):
        super().__init__()
        self.optim = optim
        self.lr_decay = (5 ** 0.5 - 1) / 2
        self.lr_warmup = lr_warmup
        self.wd_warmup = wd_warmup
        self.cos_period = cos_period

    def step(self, epoch):
        if epoch == 0:
            for pg in self.optim.param_groups:
                pg['lr'] = pg['lr_max'] / 1e4
                pg['weight_decay'] = pg['wd_max'] / 1e3
        elif epoch <= self.lr_warmup:
            for pg in self.optim.param_groups:
                pg['lr'] = pg['lr_max'] * epoch / self.lr_warmup
                pg['weight_decay'] = pg['wd_max'] / 1e3
        elif epoch <= self.wd_warmup:
            for pg in self.optim.param_groups:
                pg['lr'] = pg['lr_max']
                pg['weight_decay'] = pg['wd_max'] * (epoch-self.lr_warmup) / (self.wd_warmup-self.lr_warmup)
        else:
            for pg in self.optim.param_groups:
                i = (epoch - self.wd_warmup) // self.cos_period
                j = (epoch - self.wd_warmup) % self.cos_period
                lr_max = max(pg['lr_max'] * self.lr_decay ** i, pg['lr_min'] / (1 - self.lr_decay))
                lr_cos = np.cos(j / (self.cos_period - 1) * np.pi / 2) * self.lr_decay + (1 - self.lr_decay)
                pg['lr'] = max(lr_max * lr_cos, pg['lr_min'])
                pg['weight_decay'] = pg['wd_max']
        return self.optim.param_groups[4]['lr'], self.optim.param_groups[4]['weight_decay']


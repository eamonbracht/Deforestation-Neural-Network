from __future__ import print_function
import os
import json
from collections import defaultdict

import torch


def progress(prog, processing, total):
    """Output progress bar to the terminal to track progress of job. Useful for data_utils when generating yearly deforestation losses.
    
    Args:
        prog (int): iteration status of job prog < total
        processinig (str): descriptor of what step is currently being performed.
        total (int): total number of steps expected. total > prog.
    
    Yields:
        Progress bar printed to the terminal.
    Examples:
        >>> progress(8, "step 8", 10)
        Progress step 8 [################    ] 8/10
    """
    amount = (int(prog/total*100)//2)
    fmt = "{:<10}{:<5}[{:<50}]\t{}"
    status = str(prog)+"/"+str(total)
    print(fmt.format("Progress", processing,'#'* amount, status), end = '\r')
    if amount == 50:
        print("\r")




def rmse(x_pred, x_target, reduce=True):
    if reduce:
        return x_pred.sub(x_target).pow(2).sum(-1).sqrt().mean().item()
    return x_pred.sub(x_target).pow(2).sum(2).sqrt().mean(1).squeeze()


def normalize(mx):
    """Row-normalize matrix"""
    rowsum = mx.sum(1)
    r_inv = 1 / rowsum
    r_inv[r_inv == float('Inf')] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv.matmul(mx)
    return mx


def identity(input):
    return input


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Logger(object):
    def __init__(self, log_dir, name, chkpt_interval):
        super(Logger, self).__init__()
        os.makedirs(os.path.join(log_dir, name))
        self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.model_path = os.path.join(log_dir, name, 'model.pt')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.model_path)

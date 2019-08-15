import train_comp
import os
import random
import json
from collections import defaultdict, OrderedDict
import numpy as np
import configargparse
from data_utils import make_relation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data_utils import roundup
from datasets_comp import dataset_factory, get_relations
from utils import DotDict, Logger, rmse
from stnn import SaptioTemporalNN
print("Initializing....")
#######################################################################
# Options - CUDA - Random seed
#######################################################################
p = configargparse.ArgParser()
# -- data
p.add('--datadir',      type=str, help='path to dataset', default='data')
p.add('--dataset',      type=str, help='dataset name', default='heat')
# -- xp
p.add('--outputdir',    type=str, help='path to save xp', default='output')
p.add('--xp',           type=str, help='xp name', default='stnn')
# -- model
p.add('--mode',         type=str, help='STNN mode (default|refine|discover)', default='default')
p.add('--nz',           type=int, help='laten factors size', default=1)
p.add('--activation',   type=str, help='dynamic module activation function (identity|tanh)', default='identity')
p.add('--khop',         type=int, help='spatial depedencies order', default=1)
p.add('--nhid',         type=int, help='dynamic function hidden size', default=0)
p.add('--nlayers',      type=int, help='dynamic function num layers', default=1)
p.add('--dropout_f',    type=float, help='latent factors dropout', default=.0)
p.add('--dropout_d',    type=float, help='dynamic function dropout', default=.0)
p.add('--lambd',        type=float, help='lambda between reconstruction and dynamic losses', default=.1)
# -- optim
p.add('--lr',           type=float, help='learning rate', default=3e-3)
p.add('--beta1',        type=float, default=.0, help='adam beta1')
p.add('--beta2',        type=float, default=.999, help='adam beta2')
p.add('--eps',          type=float, default=1e-9, help='adam eps')
p.add('--wd',           type=float, help='weight decay', default=1e-6)
p.add('--wd_z',         type=float, help='weight decay on latent factors', default=1e-7)
p.add('--l2_z',         type=float, help='l2 between consecutives latent factors', default=0.)
p.add('--l1_rel',       type=float, help='l1 regularization on relation discovery mode', default=0.)
p.add('--batch_size',   type=int, default=1000, help='batch size')
p.add('--patience',     type=int, default=150, help='number of epoch to wait before trigerring lr decay')
p.add('--nepoch',       type=int, default=250,  help='number of epochs to train for')
p.add('--width',	type=int, help = 'width of dataset', default = 0)
p.add('--height', 	type=int, help = 'height of dataset', default = 0)
p.add('--lrsch', 	type=int, help = 'min rmse before lr can be decresed', default = 1)
# -- gpu
p.add('--device',       type=int, default=-1, help='-1: cpu; > -1: cuda device id')
# -- seed
p.add('--manualSeed',   type=int, help='manual seed')
p.add('--modeldir',     type = str, help = "directory of model for prediction", default = "")
# -- convolution
p.add('--ksize',    type = int, help = 'size of kernel used to grid area', default = 0)
p.add('--tsize',    type = int, help = 'dimension of area to break area into for training', default = 50)
opt = DotDict(vars(p.parse_args()))
opt.mode = opt.mode if opt.mode in ('refine', 'discover') else None
print("training area {}".format(opt.dataset))
setup, data = dataset_factory(opt)
for k, v in setup.items():
    opt[k] = v

for count, area in enumerate(data):
    if count == 0:
        opt.xp += ("_"+str(count+1))
    else:
        if count > 9:
            opt.xp = opt.xp[:-2]
        else:
            opt.xp = opt.xp[:-1]
        opt.xp += str(count+1)
    print(opt.xp)
    if np.sum(area) == 0:
        continue
    area_tensor = torch.from_numpy(area)
    area_tensor = area_tensor.type(torch.FloatTensor)
    area_tensor = area_tensor.unsqueeze(2)
    relations = get_relations(opt)
    train_data = area_tensor[:opt.nt_train]
    test_data = area_tensor[opt.nt_train:]
    train_comp.train_network(opt, train_data, test_data, relations)

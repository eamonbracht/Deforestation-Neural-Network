import os
import math
import numpy as np
import torch
import data_utils as du
from utils import DotDict, normalize
from data_utils import roundup

def dataset_factory(parm):
    # get dataset
    try:
        opt, data = import_data(parm.datadir, '{}.csv'.format(parm.dataset),[parm.height, parm.width], parm.tsize)
    except:
        raise ValueError('Non dataset named `{}`.'.format(parm.dataset))
    print(parm.datadir)
    return opt, data

def import_data(data_dir, file, dims, tsize):
    # dataset configuration
    print(dims[0], dims[1])
    opt = DotDict()
    opt.nt = 19
    opt.nt_train = 16
    opt.nx = tsize**2
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    csv = os.path.join(data_dir, file)
    reduced = np.genfromtxt(csv, delimiter = ",")
    print(reduced.shape)
    data = reduced.reshape(opt.nt,dims[0], dims[1])
    new_dims = [roundup(dims[0], tsize), roundup(dims[1], tsize)]
    step_h = int(new_dims[1]/tsize)
    step_v = int(new_dims[0]/tsize)
    pad_data = np.zeros((opt.nt, new_dims[0], new_dims[1]))
    xmin = int((new_dims[1] - dims[1])/2)
    xmax = new_dims[1]-(new_dims[1]-dims[1]-xmin)
    ymin = int((new_dims[0] - dims[0])/2)
    ymax = new_dims[0]-(new_dims[0]-dims[0]-ymin)
    pad_data[:, ymin:ymax, xmin:xmax] = data
    broken_data = []
    count = 0
    for j in np.arange(0, new_dims[1], tsize):
        for i in np.arange(0, new_dims[0], tsize):
            data = np.expand_dims(pad_data[:, i:i+tsize, j:j+tsize], axis = 0)
            data = data.reshape(1, opt.nt, -1)
            if count == 0:
                broken_data = data
            else:
                broken_data = np.append(broken_data, data, axis = 0)
            count +=1
    broken_data = np.array(broken_data)
    return opt, broken_data

def get_relations(opt):
    dims = [opt.tsize, opt.tsize]
    x = du.make_relation(["all"], dims, save = False, combine = False)
    relations = x.float()
    for i in relations:
    	i = normalize(i).unsqueeze(1)
    new_rels = [relations]
    for n in range(opt.khop - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    print(relations.size())
    return relations

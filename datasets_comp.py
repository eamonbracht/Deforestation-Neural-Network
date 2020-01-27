import os
import math
import numpy as np
import torch
import data_utils as du
from utils import DotDict, normalize
from data_utils import roundup

def dataset_factory(opt):
    # get dataset
    parm = DotDict(opt)
    try:
        opt, data = import_data(parm.datadir, '{}.csv'.format(parm.dataset), parm)
    except:
        raise ValueError('Non dataset named `{}`.'.format(parm.dataset))
    print(parm.datadir)
    return opt, data

def import_data(data_dir, file, parm):
    # dataset configuration
    dims = [parm.height, parm.width]
    tsize = parm.tsize
    stride = parm.stride
    numtrain = parm.nt_train
    print(dims[0], dims[1])
    opt = DotDict()
    opt.nx = tsize**2
    opt.nd = 1
    opt.periode = parm.nt
    # loading data
    csv = os.path.join(data_dir, file)
    reduced = np.genfromtxt(csv, delimiter = ",")
    print(reduced.shape)
    data = reduced.reshape(parm.nt_data,dims[0], dims[1])
    new_dims = [roundup(dims[0], tsize), roundup(dims[1], tsize)]
    opt.new_dims = new_dims
    pad_data = np.empty((parm.nt_data, new_dims[0], new_dims[1]))
    pad_data[:] = np.nan
    step_x = int((new_dims[1] - tsize)/stride) + 1
    step_y = int((new_dims[0] - tsize)/stride) + 1
    xmin = int((new_dims[1] - dims[1])/2)
    xmax = new_dims[1]-(new_dims[1]-dims[1]-xmin)
    ymin = int((new_dims[0] - dims[0])/2)
    ymax = new_dims[0]-(new_dims[0]-dims[0]-ymin)
    pad_data[:, ymin:ymax, xmin:xmax] = data
    broken_data = []
    count = 0
    for j in np.arange(0, step_x*stride, stride):
        for i in np.arange(0, step_y*stride, stride):
            data = np.expand_dims(pad_data[:, i:i+tsize, j:j+tsize], axis = 0)
            data = data.reshape(1, parm.nt_data, -1)
            if count == 0:
                broken_data = data
            else:
                broken_data = np.append(broken_data, data, axis = 0)
            count +=1
    broken_data = np.array(broken_data)
    return opt, broken_data

def tc_to_linearcord(data, height):
    output = []
    for i,j in data:
        output.append(i*height+j)
    output = np.array(output, dtype = int)

def get_relations(opt, data):
    print("getrel", data.shape)
    if opt.exclude:
        print("excluding perimeter values")
        exclude = np.argwhere(np.isnan(data[0]))
    else:
        print("not excluding values")
        exclude = np.empty(0)
    # exclude_linear = tc_to_linearcord(exclude, opt.height)
    dims = [opt.tsize, opt.tsize]
    x = du.make_relation(["all"], dims, exclude, save = False,
    combine = False)
    relations = x.float()
    for i in relations:
    	i = normalize(i).unsqueeze(1)
    new_rels = [relations]
    for n in range(opt.khop - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    print(relations.size())
    return relations

import os
import math
import numpy as np
import torch
import data_utils as du
from utils import DotDict, normalize


def dataset_factory(data_dir, name, height, width, k=1, makerel = True):
    # get dataset
    try:
        opt, data, relations = import_data(data_dir, '{}.csv'.format(name),[height, width], makerel)
    except:
        raise ValueError('Non dataset named `{}`.'.format(name))
    # make k hop
    print(data_dir)
    if makerel:
        new_rels = [relations]
        for n in range(k - 1):
            new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
        relations = torch.cat(new_rels, 1)
    # split train / test
    train_data = data[:opt.nt_train]
    test_data = data[opt.nt_train:]
    return opt, (train_data, test_data), relations


def import_data(data_dir, file, dims, makerel):
    # dataset configuration
    print(dims[0], dims[1])
    opt = DotDict()
    opt.nt = 17
    opt.nt_train = 14
    opt.nx = dims[0]*dims[1]
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    csv = os.path.join(data_dir, file)
    reduced = np.genfromtxt(csv, delimiter = ",")
    print(reduced.shape)
    reduced = np.expand_dims(reduced, axis = 2)
    data = torch.from_numpy(reduced)
    data = data.float()
    if makerel:
        print("failed")
        x = du.make_relation(["all"], dims, save = False, combine = False)
        relations = x.float()
        for i in relations:
        	i = normalize(i).unsqueeze(1)
        print(relations[:9, 0, :9], relations.size())
    else:
        relations = []
    return opt, data, relations

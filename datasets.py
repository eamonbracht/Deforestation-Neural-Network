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
    else:
        relations = []
    # split train / test
    train_data = data[:opt.nt_train]
    test_data = data[opt.nt_train:]
    return opt, (train_data, test_data), relations

def dataloader_nt(params, k=1, makerel = True):
    # get dataset
    try:
        opt, data, relations = import_data(params.datadir, '{}.csv'.format(params.dataset),[params.shape[0], params.shape[1]], makerel)
    except:
        raise ValueError('Non dataset named `{}`.'.format(params.dataset))
    # make k hop
    if makerel:
        new_rels = [relations]
        for n in range(k - 1):
            new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
        relations = torch.cat(new_rels, 1)
    # split train / test

    data = data[opt.nt-params.years:]
    opt.nt_train = params.years-3
    train_data = data[:opt.nt_train]
    test_data = data[opt.nt_train:]
    return opt, (train_data, test_data), relations


def import_data(data_dir, file, dims, makerel):
    # dataset configuration
    print(dims[0], dims[1])
    opt = DotDict()
    opt.nt = 18
    opt.nt_train = 15
    opt.nx = dims[0]*dims[1]
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    csv_nan = os.path.join(data_dir, file)
    csv = os.path.join(data_dir, file[:-8]+'.csv')

    # exclude_dir = os.path.join(data_dir, "tree_cover", file)
    # exclude = np.genfromtxt(exclude_dir, delimiter = ",")
    # if opt.exclude:
    # ex = np.genfromtxt(csv_nan, delimiter = ",")
    # exclude = np.argwhere(np.isnan(ex))
    exclude = np.empty((0))
    area = np.genfromtxt(csv, delimiter = ",")
    area_final = np.nan_to_num(area)
    data = torch.from_numpy(np.expand_dims(area_final, axis = 2)).float()
    if makerel:
        x = du.make_relation(["all"], dims, exclude, save = False, combine = False)
        relations = x.float()
        for i in relations:
        	i = normalize(i).unsqueeze(1)
        print(relations[:9, 0, :9], relations.size())
    else:
        relations = []
    return opt, data, relations

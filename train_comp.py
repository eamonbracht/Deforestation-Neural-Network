import os
import random
import json
from collections import defaultdict, OrderedDict
import numpy as np
import configargparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from stnn import _CustomDataParallel
from datasets import dataset_factory
from utils import DotDict, Logger, rmse
from stnn import SaptioTemporalNN

def train_network(opt, train_data, test_data, relations):
    # cudnn
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.device > -1:
        torch.cuda.manual_seed_all(opt.manualSeed)
    # -- train inputs
    t_idx = torch.arange(opt.nt_train, out=torch.LongTensor()).unsqueeze(1).expand(opt.nt_train, opt.nx).contiguous()
    x_idx = torch.arange(opt.nx, out=torch.LongTensor()).expand_as(t_idx).contiguous()
    # dynamic
    idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
    nex_dyn = idx_dyn.size(1)
    # decoder
    idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(device)
    nex_dec = idx_dec.size(1)
    if opt.datagpu == 'true':
        train_data = train_data.to(device)
        test_data = test_data.to(device)
        relations = relations.to(device)
        # idx_dyn = idx_dyn.to(device)
        # idx_dec = idx_dec.to(device)

    #######################################################################
    # Model
    #######################################################################

    model = SaptioTemporalNN(relations, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers,
                             opt.dropout_f, opt.dropout_d, opt.activation, opt.periode)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model =_CustomDataParallel(model)

    model.to(device)
    #######################################################################
    # Optimizer
    #######################################################################
    params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
              {'params': model.dynamic.parameters()},
              {'params': model.decoder.parameters()}]
    if opt.mode in ('refine', 'discover'):
        params.append({'params': model.rel_parameters(), 'weight_decay': 0.})
    optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
    if opt.patience > 0:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)


    #######################################################################
    # Logs
    #######################################################################
    logger = Logger(opt.outputdir, opt.xp, 100)
    with open(os.path.join(opt.outputdir, opt.xp, 'config.json'), 'w') as f:
        json.dump(opt, f, sort_keys=True, indent=4)


    #######################################################################
    # Training
    #######################################################################
    lr = opt.lr
    pb = np.arange(opt.nepoch)
    print("Training")
    for e in pb:
        # ------------------------ Train ------------------------
        model.train()
        # --- decoder ---
        idx_perm = torch.randperm(nex_dec).to(device)
        batches = idx_perm.split(opt.batch_size)
        logs_train = defaultdict(float)
        for i, batch in enumerate(batches):
            optimizer.zero_grad()
            # data
            input_t = idx_dec[0][batch]
            input_x = idx_dec[1][batch]
            x_target = train_data[input_t, input_x]
            # closure
            x_rec = model.dec_closure(input_t, input_x)
            mse_dec = F.mse_loss(x_rec, x_target)
            # backward
            mse_dec.backward()
            # step
            optimizer.step()
            # log
            logger.log('train_iter.mse_dec', mse_dec.item())
            logs_train['mse_dec'] += mse_dec.item() * len(batch)
        # --- dynamic ---
        idx_perm = torch.randperm(nex_dyn).to(device)
        batches = idx_perm.split(opt.batch_size)
        for i, batch in enumerate(batches):
            optimizer.zero_grad()
            # data
            input_t = idx_dyn[0][batch]
            input_x = idx_dyn[1][batch]
            # closure
            z_inf = model.factors[input_t, input_x]
            z_pred = model.dyn_closure(input_t - 1, input_x)
            # loss
            mse_dyn = z_pred.sub(z_inf).pow(2).mean()
            loss_dyn = mse_dyn * opt.lambd
            if opt.l2_z > 0:
                loss_dyn += opt.l2_z * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()
            if opt.mode in('refine', 'discover') and opt.l1_rel > 0:
                rel_weights_tmp = model.rel_weights.data.clone()
                loss_dyn += opt.l1_rel * model.get_relations().abs().mean()
            # backward
            loss_dyn.backward()
            # step
            optimizer.step()
            # clip
            if opt.mode == 'discover' and opt.l1_rel > 0:  # clip
                sign_changed = rel_weights_tmp.sign().ne(model.rel_weights.data.sign())
                model.rel_weights.data.masked_fill_(sign_changed, 0)
            # log
            logger.log('train_iter.mse_dyn', mse_dyn.item())
            logs_train['mse_dyn'] += mse_dyn.item() * len(batch)
            logs_train['loss_dyn'] += loss_dyn.item() * len(batch)
        # --- logs ---
        logs_train['mse_dec'] /= nex_dec
        logs_train['mse_dyn'] /= nex_dyn
        logs_train['loss_dyn'] /= nex_dyn
        logs_train['loss'] = logs_train['mse_dec'] + logs_train['loss_dyn']
        logger.log('train_epoch', logs_train)
        # ------------------------ Test ------------------------
        model.eval()
        with torch.no_grad():
            x_pred, _ = model.generate(opt.nt - opt.nt_train)
    #        print("x_pred", x_pred.size(), "test_data", test_data.size())
            # score_ts = rmse(x_pred, test_data, reduce=False)
            score = rmse(x_pred, test_data)
    #        if (e+1)%50 == 0:
    #            logger.save_pred(x_pred, e)
        logger.log('test_epoch.rmse', score)

    #    logger.log('test_epoch.ts', {t: {'rmse': scr.item()} for t, scr in enumerate(score_ts)})

        if (e+1) % 10  == 0:
            print("Epoch {} | train {} | test rmse {}".format(e+1, logs_train['loss'], score))
        # checkpoint
        logger.log('train_epoch.lr', lr)
        logger.checkpoint(model)
        # schedule lr
        if opt.patience > 0 and score < opt.lrsch:
            lr_scheduler.step(score)
        lr = optimizer.param_groups[0]['lr']
        if lr <= 1e-5:
            break
    logger.save(model)

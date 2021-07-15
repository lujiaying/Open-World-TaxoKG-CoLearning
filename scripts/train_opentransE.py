"""
Training and evaluate OpenKG - TransE
Author: Jiaying Lu
Create Date: Jul 13, 2021
"""

import time
import os
import random

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import sacred

from model.data_loader import prepare_ingredients_transE, collate_fn_transE
from model.TransE import OpenTransE

# Sacred Setup to keep everything in record
ex = sacred.Experiment('base-OpenTransE')


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'dataset_dir': '',     # to set
           'checkpoint_dir': '',  # to set
           'epoch': 1000,
           'batch_size': 128,
           'dist_norm': 1,
           'emb_dim': 100,
           }


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))

    # Load corpus
    train_set, cg_pairs_dev, dev_oie_set, cg_pairs_test, test_oie_set,\
        tok_vocab, mention_vocab, concept_vocab = prepare_ingredients_transE(opt['dataset_dir'])
    train_iter = DataLoader(train_set, collate_fn=collate_fn_transE, batch_size=opt['batch_size'], shuffle=True)
    _log.info('[%s] Load dataset Done, len=%d(tr), %d,%d(dev), %d,%d(tst)' % (time.ctime(),
              len(train_set), len(cg_pairs_dev), len(dev_oie_set), len(cg_pairs_test), len(test_oie_set)))
    _log.info('corpus=%s, #Tok=%d, #Mention=%d, #Concept=%d' % (opt['dataset_dir'], len(tok_vocab), len(mention_vocab),
              len(concept_vocab)))
    # Build model
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')
    model = OpenTransE(len(tok_vocab), opt['emb_dim'], opt['dist_norm'])
    model = model.to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))

    for i_epoch in range(opt['epoch']):
        # do train
        for i_batch, (h_batch, r_batch, t_batch, h_lens, r_lens, t_lens) in enumerate(train_iter):
            pos_scores = model(h_batch, r_batch, t_batch, h_lens, r_lens, t_lens)
            exit(0)
            # neg_scores

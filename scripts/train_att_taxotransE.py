"""
Training and evaluate Att-Taxo-TransE
Can not apply big sparse adjacency matrix,
as many operations are not multiplications.
Author: Jiaying Lu
Create Date: Jun 22, 2021
"""

import time
import os
import random
from typing import Tuple

import numpy as np
from scipy import sparse as spsp
import torch as th
from torch.utils.data import DataLoader
import sacred

from model.data_loader import prepare_ingredients, sample_negative_triples,\
        get_taxo_parents_children, get_normalized_adj_matrix
from model.TransE import cal_metrics
from model.TaxoTransE import TaxoTransE

# Sacred Setup to keep everything in record
ex = sacred.Experiment('Att-Taxo-TransE')


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'corpus_type': '',     # WN18RR | CN100k
           'checkpoint_dir': '',  # to set
           'dataset_dir': {
               'WN18RR': 'data/WN18RR',
               'CN100k': 'data/CN-100K'
               },
           'epoch': 1000,
            }


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))

    # Sanity check
    if opt['corpus_type'] not in ['WN18RR', 'CN100k']:
        _log.error('corpus_type=%s invalid' % (opt['corpus_type']))
        exit(-1)
    if opt['checkpoint_dir'] == '':
        _log.error('checkpoint_dir=%s invalid' % (opt['checkpoint_dir']))
        exit(-1)
    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])
    # Setup essential vars
    dataset_dir = opt['dataset_dir'][opt['corpus_type']]
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')

    # Load corpus
    train_set, dev_set, test_set, \
        ent_vocab, rel_vocab, train_triples, all_triples_map = prepare_ingredients(dataset_dir, opt['corpus_type'])
    train_iter = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True)
    dev_iter = DataLoader(dev_set, batch_size=opt['batch_size']//4, shuffle=False)
    test_iter = DataLoader(test_set, batch_size=opt['batch_size']//4, shuffle=False)
    _log.info('[%s] Load dataset Done, len=%d,%d,%d' % (time.ctime(),
              len(train_set), len(dev_set), len(test_set)))
    _log.info('corpus=%s, Entity cnt=%d, rel cnt=%d' % (opt['corpus_type'], len(ent_vocab), len(rel_vocab)))
    taxo_dict = get_taxo_parents_children(train_triples, rel_vocab, opt['corpus_type'])
    _log.info('[%s] avg parents each ent=%.2f, avg children=%.2f' % (time.ctime(), sum(len(_) for _ in taxo_dict['p'].values())/len(taxo_dict['p']),
                                                                     sum(len(_) for _ in taxo_dict['c'].values())/len(taxo_dict['c'])))

    for i_epoch in range(opt['epoch']):
        # do train
        # model.train()
        # train_loss = []
        for i_batch, (batch_h, batch_r, batch_t) in enumerate(train_iter):
            # prepare parents, children for h and t.
            # use padded sequences
            batch_h_p, batch_h_c = prepare_batch_taxo_ents(batch_h, taxo_dict)
            print('batch_h shape: ', batch_h.shape)
            print('batch_h_p shape: ', batch_h_p.shape)
            print('batch_h_c shape: ', batch_h_c.shape)
            exit(0)

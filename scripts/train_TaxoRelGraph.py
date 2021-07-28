"""
Training and evaluate TaxoRelGraph
Author: Jiaying Lu
Create Date: Jul 26, 2021
"""

import time
import os
import random
from typing import Tuple

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import prepare_ingredients_TaxoRelGraph, get_concept_tok_tensor
from model.data_loader import CGCEgoGraphDst
from model.TaxoRelGraph import TokenEncoder, CompGCN
from utils.metrics import cal_AP_atk, cal_reciprocal_rank

# Sacred Setup to keep everything in record
ex = sacred.Experiment('TaxoRelGraph')


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'dataset_type': '',     # MSCG-ReVerb, ..., SEMusic-OPIEC
           'checkpoint_dir': '',  # to set
           'dataset_dir': {
               'MSCG-ReVerb': "data/CGC-OLP-BENCH/MSCG-ReVerb",
               'SEMedical-ReVerb': "data/CGC-OLP-BENCH/SEMedical-ReVerb",
               'SEMusic-ReVerb': "data/CGC-OLP-BENCH/SEMusic-ReVerb",
               'MSCG-OPIEC': "data/CGC-OLP-BENCH/MSCG-OPIEC",
               'SEMedical-OPIEC': "data/CGC-OLP-BENCH/SEMedical-OPIEC",
               'SEMusic-OPIEC': "data/CGC-OLP-BENCH/SEMusic-OPIEC",
               },
           'epoch': 1000,
           'validate_freq': 10,
           'batch_size': 8,
           'dist_norm': 1,
           'emb_dim': 256,
           'optim_type': 'Adam',   # Adam | SGD
           'optim_lr': 1e-3,
           'optim_wdecay': 0.5e-4,
           'loss_margin': 3.0,
           'clip_grad_max_norm': 2.0,
           'pretrain_tok_emb': ''
           }


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))
    # Sanity check
    if opt['dataset_type'] not in opt['dataset_dir']:
        _log.error('dataset_type=%s invalid' % (opt['dataset_type']))
        exit(-1)
    if opt['checkpoint_dir'] == '':
        _log.error('checkpoint_dir=%s invalid' % (opt['checkpoint_dir']))
        exit(-1)
    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])
    dataset_dir = opt['dataset_dir'][opt['dataset_type']]
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')
    # Load corpus
    train_CGC_set, dev_CGC_set, test_CGC_set,\
        tok_vocab, mention_vocab, concept_vocab,\
        rel_vocab, all_oie_triples_map = prepare_ingredients_TaxoRelGraph(dataset_dir)
    train_CGC_iter = DataLoader(train_CGC_set, collate_fn=CGCEgoGraphDst.collate_fn,
                                batch_size=opt['batch_size'], shuffle=True)
    # Build model
    token_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim'])
    token_encoder = token_encoder.to(device)
    comp_gcn = CompGCN(opt['emb_dim'], 10)
    comp_gcn = comp_gcn.to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))

    for i_epoch in range(opt['epoch']):
        # do train
        for i_batch, (bg, node_toks, node_tlens, edge_toks, edge_tlens, cep_toks,
                      cep_tlens, batch_num_concepts) in enumerate(train_CGC_iter):
            node_toks = node_toks.to(device)
            node_toks = node_toks.to(device)
            bg = bg.to(device)
            # get emb for nodes, edges, concepts
            node_embs = token_encoder(node_toks, node_tlens)  # (n_cnt, emb_d)
            edge_embs = token_encoder(edge_toks, edge_tlens)  # (e_cnt, emb_d)
            cep_embs = token_encoder(cep_toks, cep_tlens)     # (c_cnt, emb_d)
            h_nodes, h_edges = comp_gcn(bg, node_embs, edge_embs)
            print('h_nodes size', h_nodes.size())
            print('h_edges size', h_edges.size())
            exit(0)

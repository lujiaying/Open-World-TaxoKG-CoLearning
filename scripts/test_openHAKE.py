"""
Test Scripts
Author: Jiaying Lu
Create Date: Aug 26, 2021
"""
import time
import os
import json
import random
from typing import Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sacred

from model.data_loader import prepare_ingredients_HAKE, get_concept_tok_tensor
from model.data_loader import collate_fn_CGCpairs, CompGCNOLPTripleDst
from model.data_loader import HAKETrainDst, BatchType
from model.TaxoRelGraph import TokenEncoder
from model.HAKE import HAKE
from .train_openHAKE import test_CGC_task, test_OLP_task

# Sacred Setup
ex = sacred.Experiment('test_HAKE')


@ex.config
def my_config():
    config_path = ''
    checkpoint_path = ''


@ex.automain
def test_model(config_path, checkpoint_path, _run, _log):
    if not config_path or not checkpoint_path:
        _log.error('missing arg=config_path | checkpoint_path')
        exit(-1)

    # Load config
    _log.info('Load config from %s' % (config_path))
    with open(config_path) as fopen:
        loaded_cfg = json.load(fopen)
        opt = loaded_cfg['opt']
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    # Set up
    dataset_dir = opt['dataset_dir'][opt['dataset_type']]
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')
    # device = th.device('cpu')
    # Load corpus
    train_set_head_batch, train_set_tail_batch,\
        dev_cg_set, test_cg_set, dev_olp_set, test_olp_set, concept_vocab,\
        tok_vocab, train_mention_vocab, train_rel_vocab, all_mention_vocab,\
        all_rel_vocab, all_oie_triples_map = prepare_ingredients_HAKE(dataset_dir, opt['negative_size'])
    test_cg_iter = DataLoader(test_cg_set, collate_fn=collate_fn_CGCpairs, batch_size=opt['batch_size'], shuffle=False)
    test_olp_iter = DataLoader(test_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    # Build model
    tok_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim']).to(device)
    scorer = HAKE(opt['emb_dim'], opt['gamma'], opt['mod_w'], opt['pha_w']).to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    # do test
    checkpoint = th.load(checkpoint_path)
    tok_encoder.load_state_dict(checkpoint['tok_encoder'])
    tok_encoder = tok_encoder.to(device)
    tok_encoder.eval()
    scorer.load_state_dict(checkpoint['scorer'])
    scorer = scorer.to(device)
    scorer.eval()
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, scorer, test_cg_iter,
                                              tok_vocab, concept_vocab, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, scorer, test_olp_iter, tok_vocab, all_mention_vocab,
                                           all_rel_vocab, device, all_oie_triples_map)
    _run.log_scalar("test.OLP.MRR", OLP_MRR)
    _run.log_scalar("test.OLP.Hits@10", H10)
    _run.log_scalar("test.OLP.Hits@30", H30)
    _run.log_scalar("test.OLP.Hits@50", H50)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

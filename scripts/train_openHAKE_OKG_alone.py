"""
Training and evaluate OpenKG - HAKE
Now only train/test on OKG alone, w/o Taxo
Author: Anonymous Siamese
Create Date: Aug 7, 2022
"""

import time
import os
import random
from typing import Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import prepare_ingredients_HAKE, get_concept_tok_tensor
from model.data_loader import collate_fn_CGCpairs, CompGCNOLPTripleDst
from model.data_loader import HAKETrainDst, BatchType
from model.TaxoRelGraph import TokenEncoder
from model.HAKE import HAKE
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics
from .train_openHAKE import train_step, test_CGC_task, test_OLP_task

# Sacred Setup to keep everything in record
ex = sacred.Experiment('base-OpenHAKE-verify_motivation')
ex.observers.append(FileStorageObserver("logs/open-HAKE-verify_motivation"))
ex.observers.append(NeptuneObserver(project_name='jlu/CGC-OLP-Bench', source_extensions=['.py']))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'dataset_type': '',     # MSCG-ReVerb, ..., SEMusic-OPIEC
           'checkpoint_dir': 'checkpoints/OpenHAKE-verify_motivation',
           'dataset_dir': {
               'MSCG-ReVerb': "data/CGC-OLP-BENCH/MSCG-ReVerb",
               'SEMedical-ReVerb': "data/CGC-OLP-BENCH/SEMedical-ReVerb",
               'SEMusic-ReVerb': "data/CGC-OLP-BENCH/SEMusic-ReVerb",
               'MSCG-OPIEC': "data/CGC-OLP-BENCH/MSCG-OPIEC",
               'SEMedical-OPIEC': "data/CGC-OLP-BENCH/SEMedical-OPIEC",
               'SEMusic-OPIEC': "data/CGC-OLP-BENCH/SEMusic-OPIEC",
               },
           'dataset_mix_mode': 'OKG_only',   # OKG_only, TAXO_only
           'epoch': 1000,
           'validate_freq': 10,
           'batch_size': 256,
           'negative_size': 128,
           'emb_dim': 500,
           'gamma': 12,
           'mod_w': 1.0,
           'pha_w': 0.5,
           'adv_temp': 1.0,   # adversarial temperature
           'optim_type': 'Adam',   # Adam | SGD
           'optim_lr': 1e-3,
           'optim_wdecay': 0.5e-4,
           }


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))
    # Sanity check
    assert opt['dataset_mix_mode'] in ['OKG_only', 'TAXO_only']
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
    train_set_head_batch, train_set_tail_batch,\
        dev_cg_set, test_cg_set, dev_olp_set, test_olp_set, concept_vocab,\
        tok_vocab, train_mention_vocab, train_rel_vocab, all_mention_vocab,\
        all_rel_vocab, all_oie_triples_map = prepare_ingredients_HAKE(dataset_dir, opt['negative_size'], mix_mode=opt['dataset_mix_mode'])
    train_iter_head_batch = DataLoader(
            train_set_head_batch,
            collate_fn=HAKETrainDst.collate_fn,
            batch_size=opt['batch_size'],
            shuffle=True
            )
    train_iter_tail_batch = DataLoader(
            train_set_tail_batch,
            collate_fn=HAKETrainDst.collate_fn,
            batch_size=opt['batch_size'],
            shuffle=True
            )
    dev_cg_iter = DataLoader(dev_cg_set, collate_fn=collate_fn_CGCpairs, batch_size=opt['batch_size'], shuffle=False)
    test_cg_iter = DataLoader(test_cg_set, collate_fn=collate_fn_CGCpairs, batch_size=opt['batch_size'], shuffle=False)
    dev_olp_iter = DataLoader(dev_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    test_olp_iter = DataLoader(test_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    _log.info('[%s] Load dataset [%s] Done, len=%d(tr), %d(CGC-dev)|%d(OLP-dev), %d(CGC-tst)|%d(OLP-tst)' % (time.ctime(), opt['dataset_mix_mode'],
              len(train_set_head_batch), len(dev_cg_set), len(dev_olp_set), len(test_cg_set), len(test_olp_set)))
    _log.info('corpus=%s, #Tok=%d, #Mention=%d, #Rel=%d, #Concept=%d' % (opt['dataset_type'], len(tok_vocab),
              len(all_mention_vocab), len(all_rel_vocab), len(concept_vocab)))
    # Build model
    tok_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim']).to(device)
    scorer = HAKE(opt['emb_dim'], opt['gamma'], opt['mod_w'], opt['pha_w']).to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    params = list(tok_encoder.parameters()) + list(scorer.parameters())
    optimizer = th.optim.Adam(params, opt['optim_lr'], weight_decay=opt['optim_wdecay'])

    best_MRR = 0.0
    # start training
    for i_epoch in range(opt['epoch']):
        # do train
        tok_encoder.train()
        scorer.train()
        train_loss = []
        tail_iter = iter(train_iter_tail_batch)
        for i_batch, (pos_samples, neg_samples, subsample_weights, batch_type) in enumerate(train_iter_head_batch):
            # head batch
            optimizer.zero_grad()
            ment_toks, ment_tok_lens = get_concept_tok_tensor(train_mention_vocab, tok_vocab)
            rel_toks, rel_tok_lens = get_concept_tok_tensor(train_rel_vocab, tok_vocab)
            ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
            rel_tok_embs = tok_encoder(rel_toks.to(device), rel_tok_lens.to(device))
            loss = train_step(scorer, pos_samples, neg_samples, subsample_weights,
                              ment_tok_embs, rel_tok_embs, batch_type, opt, device)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # tail batch
            optimizer.zero_grad()
            pos_samples, neg_samples, subsample_weights, batch_type = next(tail_iter)
            ment_toks, ment_tok_lens = get_concept_tok_tensor(train_mention_vocab, tok_vocab)
            rel_toks, rel_tok_lens = get_concept_tok_tensor(train_rel_vocab, tok_vocab)
            ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
            rel_tok_embs = tok_encoder(rel_toks.to(device), rel_tok_lens.to(device))
            loss = train_step(scorer, pos_samples, neg_samples, subsample_weights,
                              ment_tok_embs, rel_tok_embs, batch_type, opt, device)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            tok_encoder.eval()
            scorer.eval()
            if opt['dataset_mix_mode'] != 'OKG_only':
                MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, scorer, dev_cg_iter,
                                                          tok_vocab, concept_vocab, device)
                _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
                _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
                _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
                _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
                _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
                _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' %
                          (time.ctime(), i_epoch, MAP, CGC_MRR, P1, P3, P10))
                cur_MRR = CGC_MRR
            if opt['dataset_mix_mode'] != 'TAXO_only':
                OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, scorer, dev_olp_iter, tok_vocab, all_mention_vocab,
                                                       all_rel_vocab, device, all_oie_triples_map)
                _run.log_scalar("dev.OLP.MRR", OLP_MRR, i_epoch)
                _run.log_scalar("dev.OLP.Hits@10", H10, i_epoch)
                _run.log_scalar("dev.OLP.Hits@30", H30, i_epoch)
                _run.log_scalar("dev.OLP.Hits@50", H50, i_epoch)
                _log.info('[%s] epoch#%d OLP evaluate, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), i_epoch, OLP_MRR, H10, H30, H50))
                cur_MRR = OLP_MRR
            if cur_MRR >= best_MRR:
                best_MRR = cur_MRR
                _log.info('Save best model at eopoch#%d, prev best_MRR=%.4f, cur best_MRR=%.4f ' % (i_epoch, best_MRR, cur_MRR))
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['dataset_type'])
                th.save({
                         'tok_encoder': tok_encoder.state_dict(),
                         'scorer': scorer.state_dict(),
                         }, save_path)

    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    tok_encoder.load_state_dict(checkpoint['tok_encoder'])
    tok_encoder = tok_encoder.to(device)
    tok_encoder.eval()
    scorer.load_state_dict(checkpoint['scorer'])
    scorer = scorer.to(device)
    scorer.eval()
    if opt['dataset_mix_mode'] != 'OKG_only':
        MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, scorer, test_cg_iter,
                                                  tok_vocab, concept_vocab, device)
        _run.log_scalar("test.CGC.MAP", MAP)
        _run.log_scalar("test.CGC.MRR", CGC_MRR)
        _run.log_scalar("test.CGC.P@1", P1)
        _run.log_scalar("test.CGC.P@3", P3)
        _run.log_scalar("test.CGC.P@10", P10)
        _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    if opt['dataset_mix_mode'] != 'TAXO_only':
        OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, scorer, test_olp_iter, tok_vocab, all_mention_vocab,
                                               all_rel_vocab, device, all_oie_triples_map)
        _run.log_scalar("test.OLP.MRR", OLP_MRR)
        _run.log_scalar("test.OLP.Hits@10", H10)
        _run.log_scalar("test.OLP.Hits@30", H30)
        _run.log_scalar("test.OLP.Hits@50", H50)
        _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

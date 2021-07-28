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
from model.TaxoRelGraph import TokenEncoder, TaxoRelCGC
from utils.metrics import cal_AP_atk, cal_reciprocal_rank

# Sacred Setup to keep everything in record
ex = sacred.Experiment('TaxoRelGraph')
ex.observers.append(FileStorageObserver("logs/TaxoRelGraph"))
ex.observers.append(NeptuneObserver(project_name='jlu/CGC-OLP-Bench', source_extensions=['.py']))


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
           'epoch': 500,
           'validate_freq': 10,
           'batch_size': 16,
           'emb_dim': 256,
           'tok_emb_dropout': 0.2,
           'cgc_g_readout': 'mean',
           'optim_lr': 3e-4,
           'optim_wdecay': 0.5e-4,
           }


def test_CGC_task(token_encoder: th.nn.Module, taxorel_cgc: th.nn.Module, test_iter: DataLoader,
                  all_cep_toks: th.Tensor, all_cep_tok_lens: th.Tensor, device: th.device) -> tuple:
    MAP_topk = 15
    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    with th.no_grad():
        all_cep_embs = token_encoder(all_cep_toks, all_cep_tok_lens)  # (all_c_cnt, emb_d)
        for bg, node_toks, node_tlens, edge_toks, edge_tlens, cep_targets in test_iter:
            node_toks = node_toks.to(device)
            node_tlens = node_tlens.to(device)
            edge_toks = edge_toks.to(device)
            edge_tlens = edge_tlens.to(device)
            bg = bg.to(device)
            node_embs = token_encoder(node_toks, node_tlens)  # (n_cnt, emb_d)
            edge_embs = token_encoder(edge_toks, edge_tlens)  # (e_cnt, emb_d)
            logits = taxorel_cgc(bg, node_embs, edge_embs, all_cep_embs)  # (B, all_c_cnt)
            logits_idx = logits.argsort(dim=1, descending=True)   # (B, all_c_cnt)
            for i in range(bg.batch_size):
                gold = (cep_targets[i] == 1.0).nonzero(as_tuple=True)[0].tolist()
                pred = logits_idx[i].tolist()
                AP = cal_AP_atk(gold, pred, k=MAP_topk)
                MAP.append(AP)
                RR = cal_reciprocal_rank(gold, pred)
                MRR.append(RR)
                gold = set(gold)
                p1 = len(gold.intersection(set(pred[:1]))) / 1.0
                p3 = len(gold.intersection(set(pred[:3]))) / 3.0
                p10 = len(gold.intersection(set(pred[:10]))) / 10.0
                P1.append(p1)
                P3.append(p3)
                P10.append(p10)
    MAP = sum(MAP) / len(MAP)
    MRR = sum(MRR) / len(MRR)
    P1 = sum(P1) / len(P1)
    P3 = sum(P3) / len(P3)
    P10 = sum(P10) / len(P10)
    return MAP, MRR, P1, P3, P10


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
    dev_CGC_iter = DataLoader(dev_CGC_set, collate_fn=CGCEgoGraphDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    # Build model
    token_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim'])
    token_encoder = token_encoder.to(device)
    taxorel_cgc = TaxoRelCGC(opt['emb_dim'], opt['tok_emb_dropout'], opt['cgc_g_readout'])
    taxorel_cgc = taxorel_cgc.to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    criterion = th.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    params = list(token_encoder.parameters()) + list(taxorel_cgc.parameters())
    optimizer = th.optim.Adam(params, opt['optim_lr'], weight_decay=opt['optim_wdecay'])

    all_cep_toks, all_cep_tok_lens = get_concept_tok_tensor(concept_vocab, tok_vocab)
    all_cep_toks = all_cep_toks.to(device)
    all_cep_tok_lens = all_cep_tok_lens.to(device)
    best_CGC_MRR = 0.0
    for i_epoch in range(opt['epoch']):
        # do train
        token_encoder.train()
        taxorel_cgc.train()
        train_loss = []
        for i_batch, (bg, node_toks, node_tlens, edge_toks, edge_tlens,
                      cep_targets) in enumerate(train_CGC_iter):
            optimizer.zero_grad()
            node_toks = node_toks.to(device)
            node_tlens = node_tlens.to(device)
            edge_toks = edge_toks.to(device)
            edge_tlens = edge_tlens.to(device)
            bg = bg.to(device)
            cep_targets = cep_targets.to(device)
            # get emb for nodes, edges, concepts
            node_embs = token_encoder(node_toks, node_tlens)  # (n_cnt, emb_d)
            edge_embs = token_encoder(edge_toks, edge_tlens)  # (e_cnt, emb_d)
            all_cep_embs = token_encoder(all_cep_toks, all_cep_tok_lens)  # (all_c_cnt, emb_d)
            logits = taxorel_cgc(bg, node_embs, edge_embs, all_cep_embs)  # (B, all_c_cnt)
            loss = criterion(logits, cep_targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            token_encoder.eval()
            taxorel_cgc.eval()
            MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(token_encoder, taxorel_cgc, dev_CGC_iter,
                                                      all_cep_toks, all_cep_tok_lens, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
            _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
            _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
            _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, MAP, CGC_MRR, P1, P3, P10))
            if CGC_MRR >= best_CGC_MRR:
                _log.info('epoch#%d Save best model: prev best_CGC_MRR=%.3f, now=%.3f' % (i_epoch,
                          best_CGC_MRR, CGC_MRR))
                best_CGC_MRR = CGC_MRR
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['dataset_type'])
                th.save({
                         'token_encoder': token_encoder.state_dict(),
                         'taxorel_cgc': taxorel_cgc.state_dict(),
                         }, save_path)
    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    token_encoder.load_state_dict(checkpoint['token_encoder'])
    taxorel_cgc.load_state_dict(checkpoint['taxorel_cgc'])
    token_encoder = token_encoder.to(device)
    taxorel_cgc = taxorel_cgc.to(device)
    token_encoder.eval()
    taxorel_cgc.eval()
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(token_encoder, taxorel_cgc, dev_CGC_iter,
                                              all_cep_toks, all_cep_tok_lens, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))

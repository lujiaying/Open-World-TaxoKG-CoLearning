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
from model.data_loader import CGCEgoGraphDst, OLPEgoGraphDst
from model.TaxoRelGraph import TokenEncoder, TaxoRelCGC, TaxoRelOLP
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics

# Sacred Setup to keep everything in record
ex = sacred.Experiment('TaxoRelGraph')
# ex.observers.append(FileStorageObserver("logs/TaxoRelGraph"))
# ex.observers.append(NeptuneObserver(project_name='jlu/CGC-OLP-Bench', source_extensions=['.py']))


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
           'CGC_batch_size': 16,
           'OLP_batch_size': 256,
           'OLP_score_norm': 1,
           'emb_dim': 256,
           'tok_emb_dropout': 0.2,
           'cgc_g_readout': 'mean',
           'olp_g_readout': 'mean',
           'optim_lr': 3e-4,
           'optim_wdecay': 0.5e-4,
           'OLP_loss_margin': 1.0,
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


def add_mention_embs(all_mention_embs: list, data_iter: DataLoader,
                     token_encoder: th.nn.Module, taxorel_olp: th.nn.Module,
                     device: th.device):
    """
    all_mention_embs inplace append
    """
    with th.no_grad():
        for (subj_bg, subj_node_toks, subj_node_tlens,
             rel_toks, rel_tlens, obj_bg, obj_node_toks, obj_node_tlens,
             triple_batch) in data_iter:
            subj_bg = subj_bg.to(device)
            subj_node_toks = subj_node_toks.to(device)
            subj_node_tlens = subj_node_tlens.to(device)
            obj_bg = obj_bg.to(device)
            obj_node_toks = obj_node_toks.to(device)
            obj_node_tlens = obj_node_tlens.to(device)
            subj_node_embs = token_encoder(subj_node_toks, subj_node_tlens)
            obj_node_embs = token_encoder(obj_node_toks, obj_node_tlens)
            subj_bg_embs = taxorel_olp._compute_graph_emb(subj_bg, subj_node_embs)   # (B, emb_d)
            obj_bg_embs = taxorel_olp._compute_graph_emb(obj_bg, obj_node_embs)      # (B, emb_d)
            batch_size = len(triple_batch)
            for i in range(batch_size):
                subj_mid, obj_mid = triple_batch[i][0], triple_batch[i][2]
                if all_mention_embs[subj_mid] is None:
                    all_mention_embs[subj_mid] = subj_bg_embs[i]
                if all_mention_embs[obj_mid] is None:
                    all_mention_embs[obj_mid] = obj_bg_embs[i]
    return


def test_OLP_task(token_encoder: th.nn.Module, taxorel_olp: th.nn.Module,
                  train_iter: DataLoader, dev_iter: DataLoader, test_iter: DataLoader,
                  mention_vocab: dict, all_oie_triples_map: dict, device: th.device,
                  is_dev: bool) -> tuple:
    MRR = 0.0
    Hits10 = 0.0
    Hits30 = 0.0
    Hits50 = 0.0
    total_cnt = 0.0
    # obtain all mention embs
    all_mention_embs = [None for _ in range(len(mention_vocab))]
    add_mention_embs(all_mention_embs, train_iter, token_encoder, taxorel_olp, device)
    add_mention_embs(all_mention_embs, dev_iter, token_encoder, taxorel_olp, device)
    add_mention_embs(all_mention_embs, test_iter, token_encoder, taxorel_olp, device)
    all_mention_embs = th.stack(all_mention_embs)   # (m_cnt, emb_d)
    # do evaluation
    if is_dev:
        the_iter = dev_iter
    else:
        the_iter = test_iter
    with th.no_grad():
        for (subj_bg, subj_node_toks, subj_node_tlens,
             rel_toks, rel_tlens, obj_bg, obj_node_toks, obj_node_tlens,
             triple_batch) in the_iter:
            rel_toks = rel_toks.to(device)
            rel_tlens = rel_tlens.to(device)
            subj_mids = [_[0] for _ in triple_batch]
            r_rids = [_[1] for _ in triple_batch]
            obj_mids = [_[2] for _ in triple_batch]
            subj_embs = all_mention_embs[subj_mids]  # (B, emb_d)
            obj_embs = all_mention_embs[obj_mids]
            rel_tok_embs = token_encoder(rel_toks, rel_tlens)
            # tail pred
            pred_tails = taxorel_olp.test_tail_pred(subj_embs, rel_tok_embs, all_mention_embs)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_tails, subj_mids, r_rids,
                                                         obj_mids, True, all_oie_triples_map)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            # head pred
            pred_heads = taxorel_olp.test_head_pred(obj_embs, rel_tok_embs, all_mention_embs)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_heads, subj_mids, r_rids,
                                                         obj_mids, False, all_oie_triples_map)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            total_cnt += (2 * len(triple_batch))
    MRR /= total_cnt
    Hits10 /= total_cnt
    Hits30 /= total_cnt
    Hits50 /= total_cnt
    return MRR, Hits10, Hits30, Hits50


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
        train_OLP_set, dev_OLP_set, test_OLP_set,\
        tok_vocab, mention_vocab, concept_vocab,\
        rel_vocab, all_oie_triples_map = prepare_ingredients_TaxoRelGraph(dataset_dir)
    train_CGC_iter = DataLoader(train_CGC_set, collate_fn=CGCEgoGraphDst.collate_fn,
                                batch_size=opt['CGC_batch_size'], shuffle=True)
    dev_CGC_iter = DataLoader(dev_CGC_set, collate_fn=CGCEgoGraphDst.collate_fn,
                              batch_size=opt['CGC_batch_size'], shuffle=False)
    test_CGC_iter = DataLoader(test_CGC_set, collate_fn=CGCEgoGraphDst.collate_fn,
                               batch_size=opt['CGC_batch_size'], shuffle=False)
    train_OLP_iter = DataLoader(train_OLP_set, collate_fn=OLPEgoGraphDst.collate_fn,
                                batch_size=opt['OLP_batch_size'], shuffle=True)
    dev_OLP_iter = DataLoader(dev_OLP_set, collate_fn=OLPEgoGraphDst.collate_fn,
                              batch_size=opt['OLP_batch_size'], shuffle=False)
    test_OLP_iter = DataLoader(test_OLP_set, collate_fn=OLPEgoGraphDst.collate_fn,
                               batch_size=opt['OLP_batch_size'], shuffle=False)
    _log.info('Train CGC ego graph avg #node=%.2f, #edge=%.2f' % (train_CGC_set.avg_node_cnt,
              train_CGC_set.avg_edge_cnt))
    _log.info('Train OLP ego graph avg #node=%.2f, #edge=%.2f' % (train_OLP_set.avg_node_cnt,
              train_OLP_set.avg_edge_cnt))
    # Build model
    token_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim'])
    token_encoder = token_encoder.to(device)
    taxorel_cgc = TaxoRelCGC(opt['emb_dim'], opt['tok_emb_dropout'], opt['cgc_g_readout'])
    taxorel_cgc = taxorel_cgc.to(device)
    taxorel_olp = TaxoRelOLP(opt['emb_dim'], opt['tok_emb_dropout'], opt['olp_g_readout'],
                             opt['OLP_score_norm'])
    taxorel_olp = taxorel_olp.to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    CGC_criterion = th.nn.BCEWithLogitsLoss()
    CGC_criterion = CGC_criterion.to(device)
    OLP_criterion = th.nn.MarginRankingLoss(margin=opt['OLP_loss_margin'])
    OLP_criterion = OLP_criterion.to(device)
    params_CGC = list(token_encoder.parameters()) + list(taxorel_cgc.parameters())
    params_OLP = list(token_encoder.parameters()) + list(taxorel_olp.parameters())
    optimizer_CGC = th.optim.Adam(params_CGC, opt['optim_lr'], weight_decay=opt['optim_wdecay'])
    optimizer_OLP = th.optim.Adam(params_OLP, opt['optim_lr'], weight_decay=opt['optim_wdecay'])

    all_cep_toks, all_cep_tok_lens = get_concept_tok_tensor(concept_vocab, tok_vocab)
    all_cep_toks = all_cep_toks.to(device)
    all_cep_tok_lens = all_cep_tok_lens.to(device)
    best_sum_MRR = 0.0
    w_CGC_MRR = 0.5
    for i_epoch in range(opt['epoch']):
        # do train: iteratively train two tasks (future: batch iterative)
        token_encoder.train()
        taxorel_cgc.train()
        taxorel_olp.train()
        train_OLP_loss = []
        train_CGC_loss = []
        for i_batch, (subj_bg, subj_node_toks, subj_node_tlens,
                      rel_toks, rel_tlens,
                      obj_bg, obj_node_toks, obj_node_tlens,
                      triple_batch) in enumerate(train_OLP_iter):
            optimizer_OLP.zero_grad()
            subj_bg = subj_bg.to(device)
            subj_node_toks = subj_node_toks.to(device)
            subj_node_tlens = subj_node_tlens.to(device)
            obj_bg = obj_bg.to(device)
            obj_node_toks = obj_node_toks.to(device)
            obj_node_tlens = obj_node_tlens.to(device)
            rel_toks = rel_toks.to(device)
            rel_tlens = rel_tlens.to(device)
            tic = time.perf_counter()
            subj_node_embs = token_encoder(subj_node_toks, subj_node_tlens)
            obj_node_embs = token_encoder(obj_node_toks, obj_node_tlens)
            rel_embs = token_encoder(rel_toks, rel_tlens)
            toc = time.perf_counter()
            _log.info('token encoder elapsed time %.3f' % (toc-tic))
            tic = time.perf_counter()
            pos_scores, neg_scores = taxorel_olp(subj_bg, subj_node_embs, rel_embs,
                                                 obj_bg, obj_node_embs)
            toc = time.perf_counter()
            _log.info('taxorel_olp elapsed time %.3f' % (toc-tic))
            if i_batch >= 1:
                exit(0)  # debug
            target = pos_scores.new_tensor([-1])
            loss = OLP_criterion(pos_scores, neg_scores, target)
            train_OLP_loss.append(loss.item())
            loss.backward()
            optimizer_OLP.step()
        avg_loss = sum(train_OLP_loss) / len(train_OLP_loss)
        _run.log_scalar("train.OLP.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg OLP loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # TODO: MTL debug
        for i_batch, (bg, node_toks, node_tlens, edge_toks, edge_tlens,
                      cep_targets) in enumerate(train_CGC_iter):
            optimizer_CGC.zero_grad()
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
            loss = CGC_criterion(logits, cep_targets)
            train_CGC_loss.append(loss.item())
            loss.backward()
            optimizer_CGC.step()
        avg_loss = sum(train_CGC_loss) / len(train_CGC_loss)
        _run.log_scalar("train.CGC.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg CGC loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            token_encoder.eval()
            taxorel_cgc.eval()
            taxorel_olp.eval()
            MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(token_encoder, taxorel_cgc, dev_CGC_iter,
                                                      all_cep_toks, all_cep_tok_lens, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
            _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
            _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
            _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, MAP, CGC_MRR, P1, P3, P10))
            # CGC_MRR = 0.0   # TODO: MTL debug
            OLP_MRR, H10, H30, H50 = test_OLP_task(token_encoder, taxorel_olp, train_OLP_iter,
                                                   dev_OLP_iter, test_OLP_iter, mention_vocab,
                                                   all_oie_triples_map, device, is_dev=True)
            _run.log_scalar("dev.OLP.MRR", OLP_MRR, i_epoch)
            _run.log_scalar("dev.OLP.Hits@10", H10, i_epoch)
            _run.log_scalar("dev.OLP.Hits@30", H30, i_epoch)
            _run.log_scalar("dev.OLP.Hits@50", H50, i_epoch)
            _log.info('[%s] epoch#%d OLP evaluate, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, OLP_MRR, H10, H30, H50))
            if w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR >= best_sum_MRR:
                sum_MRR = w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR
                _log.info('Save best model at eopoch#%d, prev sum_MRR=%.3f, cur sum_MRR=%.3f (CGC-%.3f,OLP-%.3f)' % (
                          i_epoch, best_sum_MRR, sum_MRR, CGC_MRR, OLP_MRR))
                best_sum_MRR = sum_MRR
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['dataset_type'])
                th.save({
                         'token_encoder': token_encoder.state_dict(),
                         'taxorel_cgc': taxorel_cgc.state_dict(),
                         'taxorel_olp': taxorel_olp.state_dict(),
                         }, save_path)
    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    token_encoder.load_state_dict(checkpoint['token_encoder'])
    taxorel_cgc.load_state_dict(checkpoint['taxorel_cgc'])
    taxorel_olp.load_state_dict(checkpoint['taxorel_olp'])
    token_encoder = token_encoder.to(device)
    taxorel_cgc = taxorel_cgc.to(device)
    taxorel_olp = taxorel_olp.to(device)
    token_encoder.eval()
    taxorel_cgc.eval()
    taxorel_olp.eval()
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(token_encoder, taxorel_cgc, test_CGC_iter,
                                              all_cep_toks, all_cep_tok_lens, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(token_encoder, taxorel_olp, train_OLP_iter,
                                           dev_OLP_iter, test_OLP_iter, mention_vocab,
                                           all_oie_triples_map, device, is_dev=False)
    _run.log_scalar("test.OLP.MRR", OLP_MRR, i_epoch)
    _run.log_scalar("test.OLP.Hits@10", H10, i_epoch)
    _run.log_scalar("test.OLP.Hits@30", H30, i_epoch)
    _run.log_scalar("test.OLP.Hits@50", H50, i_epoch)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

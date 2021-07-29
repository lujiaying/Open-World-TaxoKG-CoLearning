"""
Training and evaluate OpenKG - TransE
Author: Jiaying Lu
Create Date: Jul 13, 2021
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

from model.data_loader import prepare_ingredients_transE, get_concept_tok_tensor
from model.data_loader import collate_fn_triples, collate_fn_CGCpairs, collate_fn_oie_triples
from model.TransE import OpenTransE
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics

# Sacred Setup to keep everything in record
ex = sacred.Experiment('base-OpenTransE')
ex.observers.append(FileStorageObserver("logs/open-TransE"))
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
           'epoch': 1000,
           'validate_freq': 10,
           'batch_size': 256,
           'dist_norm': 1,
           'emb_dim': 256,
           'optim_type': 'Adam',   # Adam | SGD
           'optim_lr': 1e-3,
           'optim_wdecay': 0.5e-4,
           'loss_margin': 3.0,
           'clip_grad_max_norm': 2.0,
           'pretrain_tok_emb': ''
           }


def cal_CGC_metrics(preds: th.Tensor, golds: list) -> Tuple[list, list, list, list, list]:
    topk = 15
    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    preds_idx = preds.argsort(dim=1)    # (B, cep_cnt)
    for i_batch in range(len(golds)):
        gold = golds[i_batch]
        pred_idx = preds_idx[i_batch].tolist()
        AP = cal_AP_atk(gold, pred_idx, k=topk)
        MAP.append(AP)
        RR = cal_reciprocal_rank(gold, pred_idx)
        MRR.append(RR)
        gold = set(gold)
        p1 = len(gold.intersection(set(pred_idx[:1]))) / 1.0
        p3 = len(gold.intersection(set(pred_idx[:3]))) / 3.0
        p10 = len(gold.intersection(set(pred_idx[:10]))) / 10.0
        P1.append(p1)
        P3.append(p3)
        P10.append(p10)
    return MAP, MRR, P1, P3, P10


def test_CGC_task(model: th.nn.Module, cg_iter: DataLoader, tok_vocab: dict,
                  concept_vocab: dict, device: th.device) -> tuple:
    all_concepts, all_cep_lens = get_concept_tok_tensor(concept_vocab, tok_vocab)
    all_concepts = all_concepts.to(device)   # (cep_cnt, max_l)
    all_concept_embs = model._get_composition_emb(all_concepts, all_cep_lens, model.mention_func)  # (cep_cnt, emb_d)

    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    with th.no_grad():
        for (ent_batch, gold_ceps_batch, ent_lens) in cg_iter:
            ent_batch = ent_batch.to(device)    # (B, max_ent_l)
            B = ent_batch.size(0)
            r_batch = ent_batch.new_tensor([tok_vocab["IsA"] for _ in range(B)]).unsqueeze(-1)   # (B, 1)
            r_lens = ent_lens.new_ones(B)   # (B, )
            preds = model.test_tail_pred(ent_batch, r_batch, all_concept_embs, ent_lens, r_lens)  # (B, cep_cnt)
            MAP_b, MRR_b, P1_b, P3_b, P10_b = cal_CGC_metrics(preds, gold_ceps_batch)
            MAP.extend(MAP_b)
            MRR.extend(MRR_b)
            P1.extend(P1_b)
            P3.extend(P3_b)
            P10.extend(P10_b)
    MAP = sum(MAP) / len(MAP)
    MRR = sum(MRR) / len(MRR)
    P1 = sum(P1) / len(P1)
    P3 = sum(P3) / len(P3)
    P10 = sum(P10) / len(P10)
    return MAP, MRR, P1, P3, P10


def test_OLP_task(model: th.nn.Module, oie_iter: DataLoader, tok_vocab: dict,
                  mention_vocab: dict, rel_vocab: dict, device: th.device,
                  all_oie_triples_map: dict) -> tuple:
    all_mentions, all_mention_lens = get_concept_tok_tensor(mention_vocab, tok_vocab)
    all_mentions = all_mentions.to(device)
    all_mention_embs = model._get_composition_emb(all_mentions, all_mention_lens, model.mention_func)
    # all_mention_embs size: (ment_cnt, emb_d)

    MRR = 0.0
    Hits10 = 0.0
    Hits30 = 0.0
    Hits50 = 0.0
    total_cnt = 0.0
    with th.no_grad():
        for (h_batch, r_batch, t_batch, h_lens, r_lens, t_lens, h_mids, r_rids, t_mids) in oie_iter:
            # tail pred
            h_batch = h_batch.to(device)
            r_batch = r_batch.to(device)
            t_batch = t_batch.to(device)
            pred_tails = model.test_tail_pred(h_batch, r_batch, all_mention_embs, h_lens, r_lens)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_tails, h_mids, r_rids, t_mids, True, all_oie_triples_map)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            # head pred
            pred_heads = model.test_head_pred(t_batch, r_batch, all_mention_embs, t_lens, r_lens)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_heads, h_mids, r_rids, t_mids, False, all_oie_triples_map)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            total_cnt += (2 * h_batch.size(0))
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
    train_set, dev_cg_set, dev_oie_set, test_cg_set, test_oie_set,\
        tok_vocab, mention_vocab, concept_vocab,\
        rel_vocab, all_oie_triples_map = prepare_ingredients_transE(dataset_dir)
    train_iter = DataLoader(train_set, collate_fn=collate_fn_triples, batch_size=opt['batch_size'], shuffle=True)
    dev_cg_iter = DataLoader(dev_cg_set, collate_fn=collate_fn_CGCpairs, batch_size=opt['batch_size']//4, shuffle=False)
    dev_oie_iter = DataLoader(dev_oie_set, collate_fn=collate_fn_oie_triples,
                              batch_size=opt['batch_size']//4, shuffle=False)
    test_cg_iter = DataLoader(dev_cg_set, collate_fn=collate_fn_CGCpairs,
                              batch_size=opt['batch_size']//4, shuffle=False)
    test_oie_iter = DataLoader(test_oie_set, collate_fn=collate_fn_oie_triples,
                               batch_size=opt['batch_size']//4, shuffle=False)
    _log.info('[%s] Load dataset Done, len=%d(tr), %d(CGC-dev)|%d(OLP-dev), %d(CGC-tst)|%d(OLP-tst)' % (time.ctime(),
              len(train_set), len(dev_cg_set), len(dev_oie_set), len(test_cg_set), len(test_oie_set)))
    _log.info('corpus=%s, #Tok=%d, #Mention=%d, #Rel=%d, #Concept=%d' % (opt['dataset_type'], len(tok_vocab),
              len(mention_vocab), len(rel_vocab), len(concept_vocab)))
    # Build model
    model = OpenTransE(len(tok_vocab), opt['emb_dim'], opt['dist_norm'])
    if opt['pretrain_tok_emb'] != '':
        model.init_tok_emb_by_pretrain(tok_vocab, opt['pretrain_tok_emb'])
        _log.info('[%s] model token embedding init by %s' % (time.ctime(), opt['pretrain_tok_emb']))
    model = model.to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    criterion = th.nn.MarginRankingLoss(margin=opt['loss_margin'])
    criterion = criterion.to(device)
    optimizer = th.optim.Adam(model.parameters(), opt['optim_lr'], weight_decay=opt['optim_wdecay'])

    best_sum_MRR = 0.0
    w_CGC_MRR = 0.5
    for i_epoch in range(opt['epoch']):
        # do train
        model.train()
        train_loss = []
        for i_batch, (h_batch, r_batch, t_batch, h_lens, r_lens, t_lens) in enumerate(train_iter):
            optimizer.zero_grad()
            h_batch = h_batch.to(device)
            r_batch = r_batch.to(device)
            t_batch = t_batch.to(device)
            pos_scores, neg_scores = model(h_batch, r_batch, t_batch, h_lens, r_lens, t_lens)
            target = h_batch.new_tensor([-1])
            loss = criterion(pos_scores, neg_scores, target)
            train_loss.append(loss.item())
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt['clip_grad_max_norm'], norm_type=2)
            optimizer.step()
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            model.eval()
            MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(model, dev_cg_iter, tok_vocab, concept_vocab, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
            _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
            _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
            _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), i_epoch, MAP, CGC_MRR, P1, P3, P10))
            OLP_MRR, H10, H30, H50 = test_OLP_task(model, dev_oie_iter, tok_vocab, mention_vocab,
                                                   rel_vocab, device, all_oie_triples_map)
            _run.log_scalar("dev.OLP.MRR", OLP_MRR, i_epoch)
            _run.log_scalar("dev.OLP.Hits@10", H10, i_epoch)
            _run.log_scalar("dev.OLP.Hits@30", H30, i_epoch)
            _run.log_scalar("dev.OLP.Hits@50", H50, i_epoch)
            _log.info('[%s] epoch#%d OLP evaluate, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), i_epoch, OLP_MRR, H10, H30, H50))
            if w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR >= best_sum_MRR:
                sum_MRR = w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR
                _log.info('Save best model at eopoch#%d, prev best_sum_MRR=%.3f, cur best_sum_MRR=%.3f (CGC-%.3f,OLP-%.3f)' % (i_epoch, best_sum_MRR, sum_MRR, CGC_MRR, OLP_MRR))
                best_sum_MRR = sum_MRR
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['dataset_type'])
                th.save(model.state_dict(), save_path)

    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(model, test_cg_iter, tok_vocab, concept_vocab, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(model, test_oie_iter, tok_vocab, mention_vocab,
                                           rel_vocab, device, all_oie_triples_map)
    _run.log_scalar("test.OLP.MRR", OLP_MRR)
    _run.log_scalar("test.OLP.Hits@10", H10)
    _run.log_scalar("test.OLP.Hits@30", H30)
    _run.log_scalar("test.OLP.Hits@50", H50)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

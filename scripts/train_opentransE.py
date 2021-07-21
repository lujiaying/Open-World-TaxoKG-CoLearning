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

from model.data_loader import prepare_ingredients_transE, get_concept_tok_tensor
from model.data_loader import collate_fn_triples, collate_fn_CGCpairs
from model.TransE import OpenTransE
from utils.metrics import cal_AP_atk, cal_reciprocal_rank

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
           'validate_freq': 10,
           'batch_size': 128,
           'dist_norm': 1,
           'emb_dim': 128,
           'optim_type': 'Adam',   # Adam | SGD
           'optim_lr': 3e-4,
           'optim_wdecay': 0.5e-4,
           'loss_margin': 3.0,
           }


def cal_CGC_metrics(preds: th.Tensor, golds: list) -> Tuple[list, list, list, list, list]:
    topk = 15
    MAP = []     # MAP@15
    MRR = []
    Hits1 = []
    Hits3 = []
    Hits10 = []
    preds_idx = preds.argsort(dim=1)    # (B, cep_cnt)
    for i_batch in range(len(golds)):
        gold = golds[i_batch]
        pred_idx = preds_idx[i_batch].tolist()
        AP = cal_AP_atk(gold, pred_idx, k=topk)
        MAP.append(AP)
        RR = cal_reciprocal_rank(gold, pred_idx)
        MRR.append(RR)
        gold = set(gold)
        H1 = len(gold.intersection(set(pred_idx[:1]))) / 1.0
        H3 = len(gold.intersection(set(pred_idx[:3]))) / 3.0
        H10 = len(gold.intersection(set(pred_idx[:10]))) / 10.0
        Hits1.append(H1)
        Hits3.append(H3)
        Hits10.append(H10)
    return MAP, MRR, Hits1, Hits3, Hits10


def test_CGC_task(model: th.nn.Module, cg_iter: DataLoader, tok_vocab: dict,
                  concept_vocab: dict, device: th.device) -> tuple:
    all_concepts, all_cep_lens = get_concept_tok_tensor(concept_vocab, tok_vocab)
    all_concepts = all_concepts.to(device)   # (cep_cnt, max_l)
    all_cep_lens = all_cep_lens.to(device)
    all_concept_embs = model._get_composition_emb(all_concepts, all_cep_lens, model.mention_func)  # (cep_cnt, emb_d)

    MAP = []     # MAP@15
    MRR = []
    Hits1 = []
    Hits3 = []
    Hits10 = []
    with th.no_grad():
        for (ent_batch, gold_ceps_batch, ent_lens) in cg_iter:
            ent_batch = ent_batch.to(device)    # (B, max_ent_l)
            ent_lens = ent_lens.to(device)      # (B, )
            B = ent_batch.size(0)
            r_batch = ent_batch.new_tensor([tok_vocab["IsA"] for _ in range(B)]).unsqueeze(-1)   # (B, 1)
            r_lens = ent_batch.new_ones(B)   # (B, )
            preds = model.test_CGC(ent_batch, r_batch, all_concept_embs, ent_lens, r_lens)  # (B, cep_cnt)
            MAP_b, MRR_b, H1_b, H3_b, H10_b = cal_CGC_metrics(preds, gold_ceps_batch)
            MAP.extend(MAP_b)
            MRR.extend(MRR_b)
            Hits1.extend(H1_b)
            Hits3.extend(H3_b)
            Hits10.extend(H10_b)
    MAP = sum(MAP) / len(MAP)
    MRR = sum(MRR) / len(MRR)
    Hits1 = sum(Hits1) / len(Hits1)
    Hits3 = sum(Hits3) / len(Hits3)
    Hits10 = sum(Hits10) / len(Hits10)
    return MAP, MRR, Hits1, Hits3, Hits10


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))

    # Load corpus
    train_set, dev_cg_set, dev_oie_set, test_cg_set, test_oie_set,\
        tok_vocab, mention_vocab, concept_vocab = prepare_ingredients_transE(opt['dataset_dir'])
    train_iter = DataLoader(train_set, collate_fn=collate_fn_triples, batch_size=opt['batch_size'], shuffle=True)
    dev_cg_iter = DataLoader(dev_cg_set, collate_fn=collate_fn_CGCpairs, batch_size=opt['batch_size']//4, shuffle=False)
    _log.info('[%s] Load dataset Done, len=%d(tr), %d,%d(dev), %d,%d(tst)' % (time.ctime(),
              len(train_set), len(dev_cg_set), len(dev_oie_set), len(test_cg_set), len(test_oie_set)))
    _log.info('corpus=%s, #Tok=%d, #Mention=%d, #Concept=%d' % (opt['dataset_dir'], len(tok_vocab), len(mention_vocab),
              len(concept_vocab)))
    # Build model
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')
    model = OpenTransE(len(tok_vocab), opt['emb_dim'], opt['dist_norm'])
    model = model.to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    criterion = th.nn.MarginRankingLoss(margin=opt['loss_margin'])
    criterion = criterion.to(device)
    optimizer = th.optim.Adam(model.parameters(), opt['optim_lr'], weight_decay=opt['optim_wdecay'])

    for i_epoch in range(opt['epoch']):
        # do train
        model.train()
        train_loss = []
        for i_batch, (h_batch, r_batch, t_batch, h_lens, r_lens, t_lens) in enumerate(train_iter):
            pos_scores, neg_scores = model(h_batch, r_batch, t_batch, h_lens, r_lens, t_lens)
            target = h_batch.new_tensor([-1])
            loss = criterion(pos_scores, neg_scores, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            break     # TODO: debug
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            model.eval()
            MAP, MRR, Hits1, Hits3, Hits10 = test_CGC_task(model, dev_cg_iter, tok_vocab, concept_vocab, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", MRR, i_epoch)
            _run.log_scalar("dev.CGC.Hits1", Hits1, i_epoch)
            _run.log_scalar("dev.CGC.Hits3", Hits3, i_epoch)
            _run.log_scalar("dev.CGC.Hits10", Hits10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, hits@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), i_epoch, MAP, MRR, Hits1, Hits3, Hits10))
            exit(0)    # TODO: debug

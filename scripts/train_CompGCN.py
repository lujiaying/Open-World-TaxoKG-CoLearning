"""
Training and evaluate CompGCN
Author: Jiaying Lu
Create Date: Aug 8, 2021
"""

import time
import os
import random
from typing import Tuple

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import dgl
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import prepare_ingredients_CompGCN, CGCOLPGraphTrainDst, get_concept_tok_tensor
from model.data_loader import CompGCNCGCTripleDst, CompGCNOLPTripleDst
from model.TaxoRelGraph import TokenEncoder
from model.CompGCN import CompGCNTransE
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics

# Sacred Setup to keep everything in record
ex = sacred.Experiment('base-CompGCN')
ex.observers.append(FileStorageObserver("logs/CompGCN"))
ex.observers.append(NeptuneObserver(project_name='jlu/CGC-OLP-Bench', source_extensions=['.py']))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'dataset_type': '',     # MSCG-ReVerb, ..., SEMusic-OPIEC
           'checkpoint_dir': 'checkpoints/CompGCN',
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
           'batch_size': 128,
           'dist_norm': 1,
           'gamma': 9.0,
           'gcn_layer': 1,
           'score_func': 'TransE',
           'tok_emb_dim': 200,
           'gcn_emb_dim': 200,
           'dropout': 0.1,
           'gcn_dropout': 0.1,
           'optim_lr': 1e-3,
           'optim_wdecay': 0.0,
           'label_smooth': 0.1,
           'clip_grad_max_norm': 2.0,
           }


def test_CGC_task(compgcn_transe: th.nn.Module, test_iter: DataLoader,
                  all_node_embs: th.tensor, all_edge_embs: th.tensor, node_vocab: dict,
                  concept_vocab: dict, device: th.device) -> tuple:
    MAP_topk = 15
    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    cid_nid_mapping = [-1 for _ in range(len(concept_vocab))]
    for cep, cid in concept_vocab.items():
        nid = node_vocab[cep]
        cid_nid_mapping[cid] = nid
    cep_embs = all_node_embs[cid_nid_mapping]   # (cep_cnt, emb)
    with th.no_grad():
        for hids, rids, cep_ids_l in test_iter:
            h_embs = all_node_embs[hids.to(device)]
            r_embs = all_edge_embs[rids.to(device)]
            cep_pred = compgcn_transe.predict(h_embs, r_embs, None, cep_embs, False)  # (B, cand_cnt)
            cep_pred_idices = cep_pred.argsort(dim=1, descending=True)   # (B, cand_cnt)
            for i in range(hids.size(0)):
                gold = cep_ids_l[i]
                pred = cep_pred_idices[i].tolist()
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


def test_OLP_task(compgcn_transe: th.nn.Module, test_iter: DataLoader,
                  all_node_embs: th.tensor, all_edge_embs: th.tensor, node_vocab: dict,
                  mention_vocab: dict, all_oie_triples_map: dict, device: th.device) -> tuple:
    MRR = 0.0
    Hits10 = 0.0
    Hits30 = 0.0
    Hits50 = 0.0
    total_cnt = 0.0
    mid_nid_mapping = [-1 for _ in range(len(mention_vocab))]
    for m, mid in mention_vocab.items():
        nid = node_vocab[m]
        mid_nid_mapping[mid] = nid
    ment_embs = all_node_embs[mid_nid_mapping]  # (ment_cnt, emb)
    with th.no_grad():
        for (sids, rids, oids) in test_iter:
            sids = sids.to(device)
            rids = rids.to(device)
            oids = oids.to(device)
            subj_embs = ment_embs[sids]
            rel_embs = all_edge_embs[rids]
            obj_embs = ment_embs[oids]
            # tail pred
            pred_tails = compgcn_transe.predict(subj_embs, rel_embs, obj_embs, ment_embs, False)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_tails, sids, rids,
                                                         oids, True, all_oie_triples_map,
                                                         descending=True)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            # head pred
            pred_heads = compgcn_transe.predict(subj_embs, rel_embs, obj_embs, ment_embs, True)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_heads, sids, rids,
                                                         oids, False, all_oie_triples_map,
                                                         descending=True)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            total_cnt += (2 * sids.size(0))
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
    (train_set, dev_CGC_set, test_CGC_set,
     dev_OLP_set, test_OLP_set,
     tok_vocab, node_vocab, edge_vocab,
     mention_vocab, concept_vocab, all_oie_triples_map) = prepare_ingredients_CompGCN(dataset_dir)
    _log.info('[%s] #node_vocab=%d, #edge_vocab=%d, #ment_vocab=%d, #cep_vocab=%d' % (time.ctime(),
              len(node_vocab), len(edge_vocab), len(mention_vocab), len(concept_vocab)))
    train_iter = DataLoader(train_set, collate_fn=CGCOLPGraphTrainDst.collate_fn,
                            batch_size=opt['batch_size'], shuffle=True)
    dev_CGC_iter = DataLoader(dev_CGC_set, collate_fn=CompGCNCGCTripleDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    test_CGC_iter = DataLoader(test_CGC_set, collate_fn=CompGCNCGCTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    dev_OLP_iter = DataLoader(dev_OLP_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    test_OLP_iter = DataLoader(test_OLP_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    _log.info('DataLoader Done, trt #triple=%d, graph info:%s' % (len(train_set), train_set.graph))
    _log.info('CGC dev #triple=%d, test #triple=%d' % (len(dev_CGC_set), len(test_CGC_set)))
    _log.info('OLP dev #triple=%d, test #triple=%d' % (len(dev_OLP_set), len(test_OLP_set)))
    # build model
    token_encoder = TokenEncoder(len(tok_vocab), opt['tok_emb_dim']).to(device)
    compgcn_transe = CompGCNTransE(opt['tok_emb_dim'], opt['gcn_emb_dim'], opt['dropout'],
                                   opt['gcn_dropout'], opt['dist_norm'], opt['gamma'],
                                   opt['gcn_layer'], opt['score_func']).to(device)
    criterion = th.nn.BCELoss().to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    params = list(token_encoder.parameters()) + list(compgcn_transe.parameters())
    optimizer = th.optim.Adam(params, opt['optim_lr'], weight_decay=opt['optim_wdecay'])
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # start train, eval
    best_sum_MRR = 0.0
    w_CGC_MRR = 0.5
    kill_cnt = 0
    for i_epoch in range(opt['epoch']):
        # do train
        token_encoder.train()
        compgcn_transe.train()
        is_head_pred = 0
        train_loss = []
        g = train_set.graph.to(device)
        for i_batch, (hids, rids, tids, head_BCE_labels, tail_BCE_labels) in enumerate(train_iter):
            optimizer.zero_grad()
            hids = hids.to(device)
            rids = rids.to(device)
            tids = tids.to(device)
            # try update graph node embedding every batch
            node_toks, node_tok_lens = get_concept_tok_tensor(node_vocab, tok_vocab)
            edge_toks, edge_tok_lens = get_concept_tok_tensor(edge_vocab, tok_vocab)
            node_tok_embs = token_encoder(node_toks.to(device), node_tok_lens.to(device))
            edge_tok_embs = token_encoder(edge_toks.to(device), edge_tok_lens.to(device))
            hn, score = compgcn_transe(g, node_tok_embs, edge_tok_embs,
                                       hids, rids, tids, is_head_pred)
            target = head_BCE_labels if is_head_pred else tail_BCE_labels
            # label smoothing for better performance
            target = (1.0 - opt['label_smooth']) * target + (1.0 / len(node_vocab))
            loss = criterion(score, target.to(device))
            th.nn.utils.clip_grad_norm_(params, max_norm=opt['clip_grad_max_norm'], norm_type=2)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            is_head_pred = (is_head_pred + 1) % 2
            scheduler.step()
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.CGC.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg BCE loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            token_encoder.eval()
            compgcn_transe.eval()
            node_toks, node_tok_lens = get_concept_tok_tensor(node_vocab, tok_vocab)
            edge_toks, edge_tok_lens = get_concept_tok_tensor(edge_vocab, tok_vocab)
            node_tok_embs = token_encoder(node_toks.to(device), node_tok_lens.to(device))
            edge_tok_embs = token_encoder(edge_toks.to(device), edge_tok_lens.to(device))
            node_embs = compgcn_transe.get_all_node_embs(g, node_tok_embs, edge_tok_embs)
            edge_embs = compgcn_transe.get_all_edge_embs(edge_tok_embs)
            MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(compgcn_transe, dev_CGC_iter, node_embs,
                                                      edge_embs, node_vocab, concept_vocab, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
            _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
            _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
            _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, MAP, CGC_MRR, P1, P3, P10))
            OLP_MRR, H10, H30, H50 = test_OLP_task(compgcn_transe, dev_OLP_iter, node_embs, edge_embs,
                                                   node_vocab, mention_vocab, all_oie_triples_map, device)
            _run.log_scalar("dev.OLP.MRR", OLP_MRR, i_epoch)
            _run.log_scalar("dev.OLP.Hits@10", H10, i_epoch)
            _run.log_scalar("dev.OLP.Hits@30", H30, i_epoch)
            _run.log_scalar("dev.OLP.Hits@50", H50, i_epoch)
            _log.info('[%s] epoch#%d OLP evaluate, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, OLP_MRR, H10, H30, H50))
            if w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR - best_sum_MRR > 1e-5:
                sum_MRR = w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR
                _log.info('Save best model at eopoch#%d, prev sum_MRR=%.3f, cur sum_MRR=%.3f (CGC-%.3f,OLP-%.3f)' % (
                          i_epoch, best_sum_MRR, sum_MRR, CGC_MRR, OLP_MRR))
                best_sum_MRR = sum_MRR
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['dataset_type'])
                th.save({
                         'token_encoder': token_encoder.state_dict(),
                         'compgcn_transe': compgcn_transe.state_dict(),
                         }, save_path)
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and compgcn_transe.gamma > 5.0:
                    compgcn_transe.gamma -= 5.0
                    _log.info('Gamma decay on saturation, update to %f' % (compgcn_transe.gamma))
                if kill_cnt > 25:
                    _log.info('Early Stopping!!')
                    break  # break epoch training
    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    token_encoder.load_state_dict(checkpoint['token_encoder'])
    compgcn_transe.load_state_dict(checkpoint['compgcn_transe'])
    token_encoder = token_encoder.to(device)
    compgcn_transe = compgcn_transe.to(device)
    token_encoder.eval()
    compgcn_transe.eval()
    node_toks, node_tok_lens = get_concept_tok_tensor(node_vocab, tok_vocab)
    edge_toks, edge_tok_lens = get_concept_tok_tensor(edge_vocab, tok_vocab)
    node_tok_embs = token_encoder(node_toks.to(device), node_tok_lens.to(device))
    edge_tok_embs = token_encoder(edge_toks.to(device), edge_tok_lens.to(device))
    g = train_set.graph.to(device)
    node_embs = compgcn_transe.get_all_node_embs(g, node_tok_embs, edge_tok_embs)
    edge_embs = compgcn_transe.get_all_edge_embs(edge_tok_embs)
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(compgcn_transe, test_CGC_iter, node_embs,
                                              edge_embs, node_vocab, concept_vocab, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(compgcn_transe, test_OLP_iter, node_embs, edge_embs,
                                           node_vocab, mention_vocab, all_oie_triples_map, device)
    _run.log_scalar("test.OLP.MRR", OLP_MRR, i_epoch)
    _run.log_scalar("test.OLP.Hits@10", H10, i_epoch)
    _run.log_scalar("test.OLP.Hits@30", H30, i_epoch)
    _run.log_scalar("test.OLP.Hits@50", H50, i_epoch)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

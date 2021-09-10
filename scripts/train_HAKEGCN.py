"""
Training and evaluate HAKEGCN
Author: Jiaying Lu
Create Date: Sep 7, 2021
"""

import time
import os
import random
from typing import Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import get_concept_tok_tensor, BatchType, CompGCNCGCTripleDst, CompGCNOLPTripleDst
from model.data_loader_HAKEGCN import prepare_ingredients_HAKEGCN, HAKEGCNDst
from model.TaxoRelGraph import TokenEncoder
from model.HAKE import HAKEGCNEncoder, HAKE
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics

# Sacred Setup to keep everything in record
ex = sacred.Experiment('HAKEGCN')
ex.observers.append(FileStorageObserver("logs/HAKEGCN"))
ex.observers.append(NeptuneObserver(project_name='jlu/CGC-OLP-Bench', source_extensions=['.py']))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'dataset_type': '',     # MSCG-ReVerb, ..., SEMusic-OPIEC
           'checkpoint_dir': 'checkpoints/HAKEGCN',
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
           'batch_size': 128,
           'neg_method': 'self_adversarial',
           'neg_size': 128,
           'gsample_method': 'edge_sampling',
           'gsample_prob': 0.0,
           'emb_dim': 500,
           'tok_emb_dropout': 0.3,
           'gcn_dropout': 0.0,
           'comp_opt': 'TransE',
           'gamma': 12,
           'mod_w': 1.0,
           'pha_w': 0.5,
           'adv_temp': 1.0,   # adversarial temperature
           'optim_type': 'Adam',   # Adam | SGD
           'optim_lr': 1e-3,
           'optim_wdecay': 0.5e-4,
           }


def cal_CGC_metrics(preds: th.Tensor, golds: list, descending: bool) -> Tuple[list, list, list, list, list]:
    topk = 15
    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    preds_idx = preds.argsort(dim=1, descending=descending)    # (B, cep_cnt)
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


def test_CGC_task(tok_encoder: th.nn.Module, gcn_encoder: th.nn.Module, scorer: th.nn.Module, cg_iter: DataLoader,
                  tok_vocab: dict, all_phrase2id: dict, pid_graph_dict: dict,
                  concept_vocab: dict, opt: dict, device: th.device) -> tuple:
    # pre-compute concept embs
    with th.no_grad():
        ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
        ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
        cep_pids = [all_phrase2id[cep] for cep in concept_vocab.keys()]
        cep_embs = []
        for i in range(0, len(cep_pids), 4*opt['batch_size']):
            chunk = cep_pids[i:i+4*opt['batch_size']]
            cep_graphs = [pid_graph_dict[pid] for pid in chunk]
            cep_graphs = dgl.batch(cep_graphs).to(device)
            cep_node_embs = ment_tok_embs[cep_graphs.ndata['phrid']]
            cep_edge_embs = ment_tok_embs[cep_graphs.edata['phrid']]
            chunk_cep_embs = gcn_encoder(cep_graphs, cep_node_embs, cep_edge_embs)
            cep_embs.append(chunk_cep_embs)
        cep_embs = th.cat(cep_embs, dim=0)   # (cep_cnt, h)

    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    with th.no_grad():
        for (h_pids, r_pids, cep_ids_l) in cg_iter:
            h_graphs = [pid_graph_dict[pid.item()] for pid in h_pids]
            h_graphs = dgl.batch(h_graphs).to(device)
            hg_node_embs = ment_tok_embs[h_graphs.ndata['phrid']]
            hg_edge_embs = ment_tok_embs[h_graphs.edata['phrid']]
            head_embs = gcn_encoder(h_graphs, hg_node_embs, hg_edge_embs)  # (B, h)
            rel_embs = ment_tok_embs[r_pids.to(device)]   # (B, h)
            B = h_pids.size(0)
            cep_embs_batch = cep_embs.unsqueeze(0).expand(B, -1, -1)  # (B, ment_cnt, emb)
            pred_scores = scorer((head_embs, rel_embs, cep_embs_batch), BatchType.TAIL_BATCH)   # (B, cep_cnt)
            MAP_b, MRR_b, P1_b, P3_b, P10_b = cal_CGC_metrics(pred_scores, cep_ids_l, True)
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


def test_OLP_task(tok_encoder: th.nn.Module, gcn_encoder: th.nn.Module, scorer: th.nn.Module,
                  olp_iter: DataLoader, tok_vocab: dict, all_phrase2id: dict, pid_graph_dict: dict,
                  candidate_vocab: dict, opt: dict, device: th.device, all_triple_ids_map: dict) -> tuple:
    """
    Args:
        all_triple_ids_map: subj/olp use ment_vocab id, rel use all_phrase2id
        olp_iter: same as all_triple_ids_map
    """
    descending = True
    # pre-compute candidate embs
    with th.no_grad():
        ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
        ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
        candidate_pids = [all_phrase2id[cand] for cand in candidate_vocab.keys()]
        # pid_candidx_map = {pid: idx for idx, pid in enumerate(candidate_pids)}
        cand_embs = []
        for i in range(0, len(candidate_pids), 4*opt['batch_size']):
            chunk = candidate_pids[i:i+4*opt['batch_size']]
            cand_graphs = [pid_graph_dict[pid] for pid in chunk]
            cand_graphs = dgl.batch(cand_graphs).to(device)
            node_embs = ment_tok_embs[cand_graphs.ndata['phrid']]
            edge_embs = ment_tok_embs[cand_graphs.edata['phrid']]
            chunk_cand_embs = gcn_encoder(cand_graphs, node_embs, edge_embs)
            cand_embs.append(chunk_cand_embs)
        cand_embs = th.cat(cand_embs, dim=0)  # (pool_cnt, h)

    MRR = 0.0
    Hits10 = 0.0
    Hits30 = 0.0
    Hits50 = 0.0
    total_cnt = 0.0
    with th.no_grad():
        for (h_batch, r_batch, t_batch) in olp_iter:
            B = h_batch.size(0)
            # all_mention_embs_batch = all_mention_embs.unsqueeze(0).expand(B, -1, -1)  # (B, ment_cnt, emb)
            cand_embs_batch = cand_embs.unsqueeze(0).expand(B, -1, -1)   # (B, pool_cnt, h)
            h_embs = cand_embs[h_batch.to(device)]
            r_embs = ment_tok_embs[r_batch.to(device)]   # (B, h)
            t_embs = cand_embs[t_batch.to(device)]
            h_mids = h_batch.tolist()
            r_pids = r_batch.tolist()
            t_mids = t_batch.tolist()
            # tail pred
            pred_tails = scorer((h_embs, r_embs, cand_embs_batch), BatchType.TAIL_BATCH)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_tails, h_mids, r_pids, t_mids,
                                                         True, all_triple_ids_map, descending)
            MRR += MRR_b
            Hits10 += H10_b
            Hits30 += H30_b
            Hits50 += H50_b
            # head pred
            pred_heads = scorer((cand_embs_batch, r_embs, t_embs), BatchType.HEAD_BATCH)  # (B, ment_cnt)
            MRR_b, H10_b, H30_b, H50_b = cal_OLP_metrics(pred_heads, h_mids, r_pids, t_mids,
                                                         False, all_triple_ids_map, descending)
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


def train_step(gcn_encoder: th.nn.Module, scorer: th.nn.Module, batch: tuple, pid_graph_dict: dict,
               ment_tok_embs: th.tensor, opt: dict, device: th.device) -> th.tensor:
    (pos_samples, neg_samples, subsample_weights, batch_type, h_graphs, t_graphs) = batch
    # pos_samples: (batch, 3), neg_sampels: (batch, neg_size)
    # subsample_weights: (batch, ), batch_type: int
    h_graphs = h_graphs.to(device)
    t_graphs = t_graphs.to(device)
    hg_node_embs = ment_tok_embs[h_graphs.ndata['phrid']]
    hg_edge_embs = ment_tok_embs[h_graphs.edata['phrid']]
    head_embs = gcn_encoder(h_graphs, hg_node_embs, hg_edge_embs)  # (B, h)
    tg_node_embs = ment_tok_embs[t_graphs.ndata['phrid']]
    tg_edge_embs = ment_tok_embs[t_graphs.edata['phrid']]
    tail_embs = gcn_encoder(t_graphs, tg_node_embs, tg_edge_embs)
    rel_pids = pos_samples[:, 1].to(device)
    rel_embs = ment_tok_embs[rel_pids]   # (B, h)
    pos_scores = scorer((head_embs, rel_embs, tail_embs), BatchType.SINGLE)   # (B, 1)
    pos_scores = F.logsigmoid(pos_scores.squeeze(dim=1))   # (B,)
    B = neg_samples.size(0)
    neg_size = neg_samples.size(1)
    neg_samples_graphs = [pid_graph_dict[pid.item()] for pid in neg_samples.view(-1)]
    neg_samples_graphs = dgl.batch(neg_samples_graphs).to(device)   # (B*neg_size, )
    neg_node_embs = ment_tok_embs[neg_samples_graphs.ndata['phrid']]
    neg_edge_embs = ment_tok_embs[neg_samples_graphs.edata['phrid']]
    neg_embs = gcn_encoder(neg_samples_graphs, neg_node_embs, neg_edge_embs)  # (B*neg_size, h)
    neg_embs = neg_embs.view(B, neg_size, -1)   # (B, neg_size, h)
    if batch_type == BatchType.HEAD_BATCH:
        neg_scores = scorer((neg_embs, rel_embs, tail_embs), BatchType.HEAD_BATCH)  # (B, neg)
    elif batch_type == BatchType.TAIL_BATCH:
        neg_scores = scorer((head_embs, rel_embs, neg_embs), BatchType.TAIL_BATCH)  # (B, neg)
    else:
        raise ValueError('train_step() batch_type %s not supported' % (batch_type))
    if opt['neg_method'] == 'self_adversarial':
        neg_scores = (F.softmax(neg_scores * opt['adv_temp'], dim=1).detach()
                      * F.logsigmoid(-neg_scores)).sum(dim=1)   # (B,)
    # TODO: implement other neg_method
    subsample_weights = subsample_weights.to(device)
    pos_sample_loss = - (subsample_weights * pos_scores).sum() / subsample_weights.sum()
    neg_sample_loss = - (subsample_weights * neg_scores).sum() / subsample_weights.sum()
    loss = (pos_sample_loss + neg_sample_loss) / 2.0
    return loss


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
    train_set_head_batch, train_set_tail_batch,\
        dev_cg_set, test_cg_set, dev_olp_set, test_olp_set,\
        all_triple_ids_map, olp_ment_vocab,\
        tok_vocab, concept_vocab, all_phrase2id, pid_graph_dict\
        = prepare_ingredients_HAKEGCN(dataset_dir, opt['neg_method'], opt['neg_size'],
                                      opt['gsample_method'], opt['gsample_prob'])
    _log.info('[%s] Load dataset Done, len=%d(tr), %d(CGC-dev)|%d(OLP-dev), %d(CGC-tst)|%d(OLP-tst)' % (time.ctime(),
              len(train_set_head_batch), len(dev_cg_set), len(dev_olp_set), len(test_cg_set), len(test_olp_set)))
    train_iter_head_batch = DataLoader(
            train_set_head_batch,
            collate_fn=HAKEGCNDst.collate_fn,
            batch_size=opt['batch_size'],
            shuffle=True
            )
    train_iter_tail_batch = DataLoader(
            train_set_tail_batch,
            collate_fn=HAKEGCNDst.collate_fn,
            batch_size=opt['batch_size'],
            shuffle=True
            )
    dev_cg_iter = DataLoader(dev_cg_set, collate_fn=CompGCNCGCTripleDst.collate_fn,
                             batch_size=opt['batch_size'], shuffle=False)
    test_cg_iter = DataLoader(test_cg_set, collate_fn=CompGCNCGCTripleDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    dev_olp_iter = DataLoader(dev_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    test_olp_iter = DataLoader(test_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    _log.info('corpus=%s, #Tok=%d, #Mention=%d, #Concept=%d' % (opt['dataset_type'], len(tok_vocab),
              len(all_phrase2id), len(concept_vocab)))
    # Build model
    tok_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim']).to(device)
    gcn_encoder = HAKEGCNEncoder(opt['emb_dim'], opt['tok_emb_dropout'], opt['emb_dim'], opt['gcn_dropout'],
                                 opt['comp_opt']).to(device)
    scorer = HAKE(opt['emb_dim'], opt['gamma'], opt['mod_w'], opt['pha_w']).to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    params = list(tok_encoder.parameters()) + list(gcn_encoder.parameters()) + list(scorer.parameters())
    optimizer = th.optim.Adam(params, opt['optim_lr'], weight_decay=opt['optim_wdecay'])

    best_sum_MRR = 0.0
    w_CGC_MRR = 0.5
    for i_epoch in range(opt['epoch']):
        # do train
        tok_encoder.train()
        gcn_encoder.train()
        scorer.train()
        train_loss = []
        tail_iter = iter(train_iter_tail_batch)
        for i_batch, batch in enumerate(train_iter_head_batch):
            # head batch
            optimizer.zero_grad()
            ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
            ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
            loss = train_step(gcn_encoder, scorer, batch, pid_graph_dict, ment_tok_embs, opt, device)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # tail batch
            batch = next(tail_iter)
            optimizer.zero_grad()
            ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
            ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
            loss = train_step(gcn_encoder, scorer, batch, pid_graph_dict, ment_tok_embs, opt, device)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f' % (time.ctime(), i_epoch, avg_loss))

        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            tok_encoder.eval()
            gcn_encoder.eval()
            scorer.eval()
            MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, gcn_encoder, scorer, dev_cg_iter,
                                                      tok_vocab, all_phrase2id, pid_graph_dict,
                                                      concept_vocab, opt, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
            _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
            _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
            _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' %
                      (time.ctime(), i_epoch, MAP, CGC_MRR, P1, P3, P10))
            OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, gcn_encoder, scorer, dev_olp_iter,
                                                   tok_vocab, all_phrase2id, pid_graph_dict,
                                                   olp_ment_vocab, opt, device, all_triple_ids_map)
            _run.log_scalar("dev.OLP.MRR", OLP_MRR, i_epoch)
            _run.log_scalar("dev.OLP.Hits@10", H10, i_epoch)
            _run.log_scalar("dev.OLP.Hits@30", H30, i_epoch)
            _run.log_scalar("dev.OLP.Hits@50", H50, i_epoch)
            _log.info('[%s] epoch#%d OLP evaluate, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, OLP_MRR, H10, H30, H50))
            if w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR >= best_sum_MRR:
                sum_MRR = w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR
                _log.info('Save best model at eopoch#%d, prev best_sum_MRR=%.3f, cur best_sum_MRR=%.3f (CGC-%.3f,OLP-%.3f)' % (i_epoch, best_sum_MRR, sum_MRR, CGC_MRR, OLP_MRR))
                best_sum_MRR = sum_MRR
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['dataset_type'])
                th.save({
                         'tok_encoder': tok_encoder.state_dict(),
                         'gcn_encoder': gcn_encoder.state_dict(),
                         'scorer': scorer.state_dict(),
                         }, save_path)

    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    tok_encoder.load_state_dict(checkpoint['tok_encoder'])
    tok_encoder = tok_encoder.to(device)
    tok_encoder.eval()
    gcn_encoder.load_state_dict(checkpoint['gcn_encoder'])
    gcn_encoder = gcn_encoder.to(device)
    gcn_encoder.eval()
    scorer.load_state_dict(checkpoint['scorer'])
    scorer = scorer.to(device)
    scorer.eval()
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, gcn_encoder, scorer, test_cg_iter,
                                              tok_vocab, all_phrase2id, pid_graph_dict,
                                              concept_vocab, opt, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, gcn_encoder, scorer, dev_olp_iter,
                                           tok_vocab, all_phrase2id, pid_graph_dict,
                                           olp_ment_vocab, opt, device, all_triple_ids_map)
    _run.log_scalar("test.OLP.MRR", OLP_MRR)
    _run.log_scalar("test.OLP.Hits@10", H10)
    _run.log_scalar("test.OLP.Hits@30", H30)
    _run.log_scalar("test.OLP.Hits@50", H50)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

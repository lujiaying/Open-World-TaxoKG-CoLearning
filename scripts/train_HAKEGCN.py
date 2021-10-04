"""
Training and evaluate HAKEGCN using one big graph one shot projection
Author: Jiaying Lu
Create Date: Sep 10, 2021
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
from model.data_loader_HAKEGCN import prepare_ingredients_HAKEGCN
from model.data_loader import HAKETrainDst
from model.TaxoRelGraph import TokenEncoder
from model.HAKE import HAKEGCNEncoder, HAKEGCNScorer
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics
from utils.radam import RAdam

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
           'epoch': 700,
           'validate_freq': 10,
           'batch_size': 256,
           'neg_method': 'self_adversarial',   # 'self_adversarial' | 'cept_neg_sampling'
           'neg_size': 256,
           'tok_emb_dim': 200,
           'emb_dim': 1000,
           'emb_dropout': 0.5,
           'g_edge_sampling': 0.15,
           'gcn_layer': 2,  # 1 or 2
           'gamma': 12.0,
           'mod_w': 1.0,
           'pha_w': 0.5,
           'adv_temp': 1.0,   # adversarial temperature
           'optim_type': 'Adam',   # Adam | RAdam | SGD
           'optim_lr': 1e-3,
           'optim_wdecay': 0.5e-4,
           'w_CGC_MRR': 0.55,   # larger CGC as we perform not well
           'train_from_checkpoint': '',
           'keep_edges': 'both',  # both | relational | taxonomic
           'gcn_type': 'uniform',  # uniform | specific
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
                  tok_vocab: dict, all_phrase2id: dict, test_G: dgl.DGLGraph, test_g_nid_map: dict,
                  concept_vocab: dict, opt: dict, device: th.device) -> tuple:
    # pre-compute concept embs
    with th.no_grad():
        ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
        ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
        node_embs = ment_tok_embs[test_G.ndata['phrid']]  # (n_cnt, tok_emb)
        edge_embs = ment_tok_embs[test_G.edata['phrid']]  # (e_cng, tok_emb)
        node_embs = gcn_encoder.encode_graph(test_G, node_embs, edge_embs)
        cep_nids = [test_g_nid_map[cep] for cep in concept_vocab.keys()]
        cep_embs = node_embs[cep_nids]   # (cep_cnt, h)
    cep_cnt = cep_embs.size(0)

    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    with th.no_grad():
        for (h_nids, r_pids, cep_ids_l) in cg_iter:
            head_embs = node_embs[h_nids.to(device)]      # (B, h)
            rel_embs = ment_tok_embs[r_pids.to(device)]   # (B, tok_h)
            rel_embs = gcn_encoder.encode_relation(rel_embs)  # (B, h)
            B = head_embs.size(0)
            cep_embs_batch = cep_embs.unsqueeze(0).expand(B, cep_cnt, -1)  # (B, cep_cnt, emb)
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
                  olp_iter: DataLoader, tok_vocab: dict, all_phrase2id: dict,
                  test_G: dgl.DGLGraph, test_g_nid_map: dict,
                  candidate_vocab: dict, opt: dict, device: th.device, all_triple_ids_map: dict) -> tuple:
    """
    Args:
        test_G, test_g_nid_map: contains entities from CG and OKG
        all_triple_ids_map: subj/obj use ment_vocab id, rel use all_phrase2id
        olp_iter: subj/obj use ment_vocab id, rel use all_phrase2id
        candidate_vocab: is the ment_vocab
    """
    descending = True
    # pre-compute candidate embs
    with th.no_grad():
        ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
        ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
        node_embs = ment_tok_embs[test_G.ndata['phrid']]  # (n_cnt, tok_emb)
        edge_embs = ment_tok_embs[test_G.edata['phrid']]  # (e_cng, tok_emb)
        node_embs = gcn_encoder.encode_graph(test_G, node_embs, edge_embs)  # (n_cnt, h)
        cand_nids = [test_g_nid_map[cand] for cand in candidate_vocab.keys()]
        cand_embs = node_embs[cand_nids]  # (pool_cnt, h)
    pool_cnt = cand_embs.size(0)

    MRR = 0.0
    Hits10 = 0.0
    Hits30 = 0.0
    Hits50 = 0.0
    total_cnt = 0.0
    with th.no_grad():
        for (h_batch, r_batch, t_batch) in olp_iter:
            B = h_batch.size(0)
            cand_embs_batch = cand_embs.unsqueeze(0).expand(B, pool_cnt, -1)   # (B, pool_cnt, h)
            h_embs = cand_embs[h_batch.to(device)]
            r_embs = ment_tok_embs[r_batch.to(device)]    # (B, tok_emb)
            r_embs = gcn_encoder.encode_relation(r_embs)  # (B, h)
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


def sample_sg4_bigg(graph: dgl.DGLGraph, node_ids: list) -> dgl.DGLGraph:
    """
    Assume always sample 2-hop neighbours
    """
    # TODO: may need to change gcn_encoder forward function to accomdate.
    """
    # This version do not subsample neighbors
    in_sg = dgl.in_subgraph(graph, node_ids, relabel_nodes=True)
    in_sg_nids = in_sg.ndata[dgl.NID]
    if layer == 2:
        in_sg = dgl.in_subgraph(graph, in_sg_nids, relabel_nodes=True)
        in_sg_nids = in_sg.ndata[dgl.NID]
    out_sg = dgl.out_subgraph(graph, node_ids, relabel_nodes=True)
    out_sg_nids = out_sg.ndata[dgl.NID]
    if layer == 2:
        out_sg = dgl.out_subgraph(graph, out_sg_nids, relabel_nodes=True)
        out_sg_nids = out_sg.ndata[dgl.NID]
    sg_nids = set(in_sg_nids.tolist()+out_sg_nids.tolist())
    sg = dgl.node_subgraph(graph, list(sg_nids), relabel_nodes=True)
    """
    one_hop_max_neighs = 6
    two_hop_max_neighs = 3
    in_sg = dgl.sampling.sample_neighbors(graph, node_ids, one_hop_max_neighs)
    in_sg_nids = ((in_sg.in_degrees() != 0) | (in_sg.out_degrees() != 0)).nonzero().squeeze(1).tolist()
    in_sg_nids = list(set(in_sg_nids) - set(node_ids))   # keep only 1-hop neighbors
    in_sg = dgl.sampling.sample_neighbors(graph, in_sg_nids, two_hop_max_neighs)
    in_sg_nids = ((in_sg.in_degrees() != 0) | (in_sg.out_degrees() != 0)).nonzero().squeeze(1).tolist()  # both 1,2-hop
    out_sg = dgl.sampling.sample_neighbors(graph, node_ids, one_hop_max_neighs, edge_dir='out')
    out_sg_nids = ((out_sg.in_degrees() != 0) | (in_sg.out_degrees() != 0)).nonzero().squeeze(1).tolist()
    out_sg_nids = list(set(out_sg_nids) - set(node_ids))   # keep only 1-hop neighbors
    out_sg = dgl.sampling.sample_neighbors(graph, out_sg_nids, two_hop_max_neighs)
    out_sg_nids = ((out_sg.in_degrees() != 0) | (out_sg.out_degrees() != 0)).nonzero().squeeze(1).tolist()  # both 1,2
    node_ids = list(set(node_ids) | set(in_sg_nids) | set(out_sg_nids))
    sg = dgl.node_subgraph(graph, node_ids, relabel_nodes=True)
    return sg


def train_step(scorer: th.nn.Module, batch: tuple, g_node_embs: th.tensor,
               rels: th.tensor, opt: dict, device: th.device) -> th.tensor:
    """
    Args:
        g_node_embs: gcn computed embs for all nodes in big graph
        rels: gcn computed embs for current batch
    """
    (pos_samples, neg_samples, subsample_weights, batch_type) = batch
    subsample_weights = subsample_weights.to(device)
    heads = pos_samples[:, 0].to(device)   # (B,)
    tails = pos_samples[:, 2].to(device)   # (B,)
    heads = g_node_embs[heads]   # (B, h)
    tails = g_node_embs[tails]   # (B, h)
    pos_scores = scorer((heads, rels, tails), BatchType.SINGLE)   # (B, 1)
    pos_scores = F.logsigmoid(pos_scores.squeeze(dim=1))   # (B,)
    B = neg_samples.size(0)
    neg_size = neg_samples.size(1)   # neg_embs: (B, neg)
    neg_embs = g_node_embs[neg_samples.view(-1, 1).squeeze(1)]\
        .view(B, neg_size, -1)   # (B, neg, hid)
    if batch_type == BatchType.HEAD_BATCH:
        neg_scores = scorer((neg_embs, rels, tails), BatchType.HEAD_BATCH)  # (B, neg)
    elif batch_type == BatchType.TAIL_BATCH:
        neg_scores = scorer((heads, rels, neg_embs), BatchType.TAIL_BATCH)  # (B, neg)
    else:
        raise ValueError('train_step() batch_type %s not supported' % (batch_type))
    neg_scores = (F.softmax(neg_scores * opt['adv_temp'], dim=1).detach()
                  * F.logsigmoid(-neg_scores)).sum(dim=1)   # (B,)
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
        all_triple_ids_map, olp_ment_vocab, tok_vocab, concept_vocab,\
        all_phrase2id, train_G, train_g_nid_map, test_G, test_g_nid_map\
        = prepare_ingredients_HAKEGCN(dataset_dir, opt['neg_method'], opt['neg_size'], opt['keep_edges'])
    _log.info('[%s] Load dataset Done, len=%d(tr), %d(CGC-dev)|%d(OLP-dev), %d(CGC-tst)|%d(OLP-tst)' % (time.ctime(),
              len(train_set_head_batch), len(dev_cg_set), len(dev_olp_set), len(test_cg_set), len(test_olp_set)))
    _log.info('Train G info=%s; Test G info=%s' % (train_G, test_G))
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
    tok_encoder = TokenEncoder(len(tok_vocab), opt['tok_emb_dim']).to(device)
    gcn_encoder = HAKEGCNEncoder(opt['tok_emb_dim'], opt['emb_dropout'], opt['emb_dim'],
                                 opt['gcn_layer'], opt['gcn_type']).to(device)
    scorer = HAKEGCNScorer(opt['emb_dim'], opt['gamma'], opt['mod_w'], opt['pha_w']).to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    if opt['train_from_checkpoint']:
        checkpoint = th.load(opt['train_from_checkpoint'])
        tok_encoder.load_state_dict(checkpoint['tok_encoder'])
        gcn_encoder.load_state_dict(checkpoint['gcn_encoder'])
        scorer.load_state_dict(checkpoint['scorer'])
    no_decay = list(tok_encoder.parameters()) + list(scorer.parameters())   # embedding, score weight no need
    decay = []
    for name, param in gcn_encoder.named_parameters():
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': no_decay, 'weight_decay': 0.0},
              {'params': decay, 'weight_decay': opt['optim_wdecay']}]
    if opt['optim_type'] == 'Adam':
        optimizer = th.optim.Adam(params, lr=opt['optim_lr'])
        scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt['optim_lr'],
                                                     epochs=opt['epoch'],
                                                     steps_per_epoch=len(train_iter_head_batch))
    elif opt['optim_type'] == 'RAdam':
        optimizer = RAdam(params, lr=opt['optim_lr'])
        scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt['optim_lr'],
                                                     epochs=opt['epoch'],
                                                     steps_per_epoch=len(train_iter_head_batch))
    elif opt['optim_type'] == 'SGD':
        optimizer = th.optim.SGD(params, lr=opt['optim_lr'], momentum=0.9, nesterov=True)
        scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=64, T_mult=2)
    else:
        _log.error('invalid optim_type = %s' % (opt['optim_type']))
        exit(-1)

    best_sum_MRR = 0.0
    w_CGC_MRR = opt['w_CGC_MRR']
    test_G = test_G.to(device)
    for i_epoch in range(opt['epoch']):
        # do train
        tok_encoder.train()
        gcn_encoder.train()
        scorer.train()
        train_loss = []
        tail_iter = iter(train_iter_tail_batch)
        # conduct graph edge sampling per epoch
        for i_batch, batch in enumerate(train_iter_head_batch):
            # conduct graph edge sampling per batch
            if opt['g_edge_sampling'] > 0.0:
                edge_mask = th.rand(train_G.num_edges()) >= opt['g_edge_sampling']
                train_sG = dgl.edge_subgraph(train_G, edge_mask, relabel_nodes=False)
            else:
                train_sG = train_G
            # head batch
            optimizer.zero_grad()
            ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
            ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
            train_sG = train_sG.to(device)
            node_embs = ment_tok_embs[train_sG.ndata['phrid']]  # (n_cnt, tok_emb)
            edge_embs = ment_tok_embs[train_sG.edata['phrid']]  # (e_cng, tok_emb)
            rel_embs = ment_tok_embs[batch[0][:, 1].to(device)]  # (B, tok_emb)
            node_embs, rel_embs = gcn_encoder(train_sG, node_embs, edge_embs, rel_embs)   # (n_cnt, h), (B, h)
            loss = train_step(scorer, batch, node_embs, rel_embs, opt, device)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # tail batch
            batch = next(tail_iter)
            optimizer.zero_grad()
            ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
            ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
            node_embs = ment_tok_embs[train_sG.ndata['phrid']]  # (n_cnt, tok_emb)
            edge_embs = ment_tok_embs[train_sG.edata['phrid']]  # (e_cng, tok_emb)
            rel_embs = ment_tok_embs[batch[0][:, 1].to(device)]  # (B, tok_emb)
            node_embs, rel_embs = gcn_encoder(train_sG, node_embs, edge_embs, rel_embs)   # (n_cnt, h), (B, h)
            loss = train_step(scorer, batch, node_embs, rel_embs, opt, device)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f' % (time.ctime(), i_epoch, avg_loss))

        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            tok_encoder.eval()
            gcn_encoder.eval()
            scorer.eval()
            MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, gcn_encoder, scorer, dev_cg_iter,
                                                      tok_vocab, all_phrase2id, test_G, test_g_nid_map,
                                                      concept_vocab, opt, device)
            _run.log_scalar("dev.CGC.MAP", MAP, i_epoch)
            _run.log_scalar("dev.CGC.MRR", CGC_MRR, i_epoch)
            _run.log_scalar("dev.CGC.P@1", P1, i_epoch)
            _run.log_scalar("dev.CGC.P@3", P3, i_epoch)
            _run.log_scalar("dev.CGC.P@10", P10, i_epoch)
            _log.info('[%s] epoch#%d CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' %
                      (time.ctime(), i_epoch, MAP, CGC_MRR, P1, P3, P10))
            OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, gcn_encoder, scorer,
                                                   dev_olp_iter, tok_vocab, all_phrase2id,
                                                   test_G, test_g_nid_map,
                                                   olp_ment_vocab, opt, device, all_triple_ids_map)
            _run.log_scalar("dev.OLP.MRR", OLP_MRR, i_epoch)
            _run.log_scalar("dev.OLP.Hits@10", H10, i_epoch)
            _run.log_scalar("dev.OLP.Hits@30", H30, i_epoch)
            _run.log_scalar("dev.OLP.Hits@50", H50, i_epoch)
            _log.info('[%s] epoch#%d OLP evaluate, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(),
                      i_epoch, OLP_MRR, H10, H30, H50))
            if w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR >= best_sum_MRR:
                sum_MRR = w_CGC_MRR * CGC_MRR + (1-w_CGC_MRR) * OLP_MRR
                _log.info('Save best model at eopoch#%d, prev best_sum_MRR=%.3f, cur best_sum_MRR=%.3f (CGC-%.3f,OLP-%.3f)'
                          % (i_epoch, best_sum_MRR, sum_MRR, CGC_MRR, OLP_MRR))
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
                                              tok_vocab, all_phrase2id, test_G, test_g_nid_map,
                                              concept_vocab, opt, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, gcn_encoder, scorer,
                                           test_olp_iter, tok_vocab, all_phrase2id,
                                           test_G, test_g_nid_map,
                                           olp_ment_vocab, opt, device, all_triple_ids_map)
    _run.log_scalar("test.OLP.MRR", OLP_MRR)
    _run.log_scalar("test.OLP.Hits@10", H10)
    _run.log_scalar("test.OLP.Hits@30", H30)
    _run.log_scalar("test.OLP.Hits@50", H50)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

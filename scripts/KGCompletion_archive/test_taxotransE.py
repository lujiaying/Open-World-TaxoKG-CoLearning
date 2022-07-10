"""
Case study and analysis of Taxo-TransE
Author: Anonymous Siamese
Create Date: Jun 20, 2021
"""

import time
import json
import random
from typing import Tuple
from collections import defaultdict

import numpy as np
from scipy import sparse as spsp
import torch as th
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import prepare_ingredients, sample_negative_triples, load_WN18RR_definition,\
        get_taxo_parents_children, get_normalized_adj_matrix, get_taxo_relations
from model.TransE import cal_metrics, get_ranked_predction
from model.TaxoTransE import TaxoTransE


# Sacred Setup
ex = sacred.Experiment('test-Taxo-TransE')


@ex.config
def my_config():
    config_path = ''
    checkpoint_path = ''


def cal_taxo_nontaxo_metrics(metrics_rels: tuple, taxo_rels: list) -> tuple:
    (hits_1_rels, hits_3_rels, hits_10_rels, mrr_rels, total_cnt_rels) = metrics_rels
    # calculate taxo/non-taxo rels metrics
    taxo_hits_1 = 0.0
    taxo_hits_3 = 0.0
    taxo_hits_10 = 0.0
    taxo_mrr = 0.0
    taxo_cnt = 0.0
    nontaxo_hits_1 = 0.0
    nontaxo_hits_3 = 0.0
    nontaxo_hits_10 = 0.0
    nontaxo_mrr = 0.0
    nontaxo_cnt = 0.0
    for r, cnt in total_cnt_rels.items():
        if r in taxo_rels:
            taxo_hits_1 += hits_1_rels[r]
            taxo_hits_3 += hits_3_rels[r]
            taxo_hits_10 += hits_10_rels[r]
            taxo_mrr += mrr_rels[r]
            taxo_cnt += cnt
        else:
            nontaxo_hits_1 += hits_1_rels[r]
            nontaxo_hits_3 += hits_3_rels[r]
            nontaxo_hits_10 += hits_10_rels[r]
            nontaxo_mrr += mrr_rels[r]
            nontaxo_cnt += cnt
    taxo_hits_1 /= taxo_cnt
    taxo_hits_3 /= taxo_cnt
    taxo_hits_10 /= taxo_cnt
    taxo_mrr /= taxo_cnt
    nontaxo_hits_1 /= nontaxo_cnt
    nontaxo_hits_3 /= nontaxo_cnt
    nontaxo_hits_10 /= nontaxo_cnt
    nontaxo_mrr /= nontaxo_cnt
    return (taxo_hits_1, taxo_hits_3, taxo_hits_10, taxo_mrr), (nontaxo_hits_1, nontaxo_hits_3, nontaxo_hits_10, nontaxo_mrr)


def test(model: th.nn.Module, data_loader: DataLoader, ent_count: int, device: th.device,
         known_triples_map: dict, adj_taxo_p: spsp.csr_matrix,
         adj_taxo_c: spsp.csr_matrix) -> Tuple[float, float, float, float, tuple]:
    hits_1 = 0.0
    hits_3 = 0.0
    hits_10 = 0.0
    mrr = 0.0
    total_cnt = 0.0
    hits_1_rels = defaultdict(float)
    hits_3_rels = defaultdict(float)
    hits_10_rels = defaultdict(float)
    mrr_rels = defaultdict(float)
    total_cnt_rels = defaultdict(float)

    with th.no_grad():
        ent_ids = th.arange(end=ent_count, device=device)  # ent_c
        all_embs = model._aggregate_over_taxo(ent_ids, adj_taxo_p, adj_taxo_c, is_predict=True)     # ent_c*dim

        for (batch_h, batch_r, batch_t) in data_loader:
            batch_size = batch_h.size(0)
            all_ents = ent_ids.unsqueeze(0).repeat(batch_size, 1)    # B*ent_c
            batch_h, batch_r, batch_t = batch_h.to(device), batch_r.to(device), batch_t.to(device)   # (B, )
            batch_h = batch_h.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            batch_r = batch_r.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            batch_t = batch_t.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c

            # check all possible tails
            triples = th.stack((batch_h, batch_r, all_ents), dim=2).reshape(-1, 3)  # (B*ent_c)*3
            tail_preds = model.predict(triples, all_embs).reshape(batch_size, -1)   # B*ent_c
            # check all possible heads
            triples = th.stack((all_ents, batch_r, batch_t), dim=2).reshape(-1, 3)  # (B*ent_c)*3
            head_preds = model.predict(triples, all_embs).reshape(batch_size, -1)   # B*ent_c
            # get metrics
            batch_h = batch_h[:, 0].unsqueeze(1)   # B*1
            batch_r = batch_r[:, 0].unsqueeze(1)   # B*1
            batch_t = batch_t[:, 0].unsqueeze(1)   # B*1
            # metrics for each relation type
            for i in range(batch_size):
                r = batch_r[i:i+1, :].item()
                b_hits_1, b_hits_3, b_hits_10, b_mrr = cal_metrics(tail_preds[i:i+1, :], batch_h[i:i+1, :], batch_r[i:i+1, :], batch_t[i:i+1, :],
                                                                   is_tail_preds=True, known_triples_map=known_triples_map)
                hits_1 += b_hits_1
                hits_3 += b_hits_3
                hits_10 += b_hits_10
                mrr += b_mrr
                hits_1_rels[r] += b_hits_1
                hits_3_rels[r] += b_hits_3
                hits_10_rels[r] += b_hits_10
                mrr_rels[r] += b_mrr
                b_hits_1, b_hits_3, b_hits_10, b_mrr = cal_metrics(head_preds[i:i+1, :], batch_h[i:i+1, :], batch_r[i:i+1, :], batch_t[i:i+1, :],
                                                                   is_tail_preds=False, known_triples_map=known_triples_map)
                hits_1 += b_hits_1
                hits_3 += b_hits_3
                hits_10 += b_hits_10
                mrr += b_mrr
                total_cnt += 2
                hits_1_rels[r] += b_hits_1
                hits_3_rels[r] += b_hits_3
                hits_10_rels[r] += b_hits_10
                mrr_rels[r] += b_mrr
                total_cnt_rels[r] += 2
    # hits_1_rels = {r: v/total_cnt_rels[r] for r, v in hits_1_rels.items()}
    # hits_3_rels = {r: v/total_cnt_rels[r] for r, v in hits_3_rels.items()}
    # hits_10_rels = {r: v/total_cnt_rels[r] for r, v in hits_10_rels.items()}
    # mrr_rels = {r: v/total_cnt_rels[r] for r, v in mrr_rels.items()}
    # print('total_cnt_rels: %s' % (total_cnt_rels.items()))
    return hits_1/total_cnt, hits_3/total_cnt, hits_10/total_cnt, mrr/total_cnt, (hits_1_rels, hits_3_rels, hits_10_rels, mrr_rels, total_cnt_rels)


def case_study(model: th.nn.Module, data_loader: DataLoader, ent_count: int, device: th.device,
               known_triples_map: dict, adj_taxo_p: spsp.csr_matrix, adj_taxo_c: spsp.csr_matrix,
               ent_vocab: dict, rel_vocab: dict, wordnet_def: dict = {}):
    ent_inv_vocab = {v: k for k, v in ent_vocab.items()}
    rel_inv_vocab = {v: k for k, v in rel_vocab.items()}
    # Mostly copy from test()
    with th.no_grad():
        ent_ids = th.arange(end=ent_count, device=device)  # ent_c
        all_embs = model._aggregate_over_taxo(ent_ids, adj_taxo_p, adj_taxo_c, is_predict=True)     # ent_c*dim
        for (batch_h, batch_r, batch_t) in data_loader:
            batch_size = batch_h.size(0)
            all_ents = ent_ids.unsqueeze(0).repeat(batch_size, 1)    # B*ent_c
            batch_h, batch_r, batch_t = batch_h.to(device), batch_r.to(device), batch_t.to(device)   # (B, )
            batch_h = batch_h.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            batch_r = batch_r.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            batch_t = batch_t.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c

            # check all possible tails
            triples = th.stack((batch_h, batch_r, all_ents), dim=2).reshape(-1, 3)  # (B*ent_c)*3
            tail_preds = model.predict(triples, all_embs).reshape(batch_size, -1)   # B*ent_c
            # check all possible heads
            triples = th.stack((all_ents, batch_r, batch_t), dim=2).reshape(-1, 3)  # (B*ent_c)*3
            head_preds = model.predict(triples, all_embs).reshape(batch_size, -1)   # B*ent_c
            batch_h = batch_h[:, 0].unsqueeze(1)   # B*1
            batch_r = batch_r[:, 0].unsqueeze(1)   # B*1
            batch_t = batch_t[:, 0].unsqueeze(1)   # B*1
            for i in range(batch_size):
                h = ent_inv_vocab[batch_h[i:i+1, :].item()]
                r = rel_inv_vocab[batch_r[i:i+1, :].item()]
                t = ent_inv_vocab[batch_t[i:i+1, :].item()]
                indices = get_ranked_predction(tail_preds[i:i+1, :], batch_h[i:i+1, :], batch_r[i:i+1, :], batch_t[i:i+1, :],
                                               is_tail_preds=True, known_triples_map=known_triples_map)
                # indices shape: (1, ent_c)
                if not wordnet_def:
                    print('gold triple: <%s, %s, %s>' % (h, r, t))
                    print('pred tails: %s' % ([ent_inv_vocab[_.item()] for _ in indices[0, :10]]))
                else:
                    print('gold triple: <%s, %s, %s>' % (wordnet_def[h], r, wordnet_def[t]))
                    print('pred tails: %s' % ([wordnet_def[ent_inv_vocab[_.item()]] for _ in indices[0, :10]]))
                if i > 5:
                    return


@ex.automain
def test_model(config_path, checkpoint_path, _run, _log):
    if not config_path or not checkpoint_path:
        _log.error('missing arg config_path | checkpoint_path')

    # Load config
    _log.info('Load config from %s' % (config_path))
    with open(config_path) as fopen:
        loaded_cfg = json.load(fopen)
        opt = loaded_cfg['opt']
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    # Setup essential vars
    dataset_dir = opt['dataset_dir'][opt['corpus_type']]
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')

    # Load corpus
    train_set, dev_set, test_set, \
        ent_vocab, rel_vocab, train_triples, all_triples_map = prepare_ingredients(dataset_dir, opt['corpus_type'])
    test_iter = DataLoader(test_set, batch_size=opt['batch_size']//8, shuffle=False)
    _log.info('[%s] Load dataset Done, len=%d,%d,%d' % (time.ctime(),
              len(train_set), len(dev_set), len(test_set)))
    _log.info('corpus=%s, Entity cnt=%d, rel cnt=%d' % (opt['corpus_type'], len(ent_vocab), len(rel_vocab)))
    taxo_dict = get_taxo_parents_children(train_triples, rel_vocab, opt['corpus_type'])
    adj_taxo_p = get_normalized_adj_matrix(taxo_dict['p'], len(ent_vocab), opt['adj_norm'])
    adj_taxo_c = get_normalized_adj_matrix(taxo_dict['c'], len(ent_vocab), opt['adj_norm'])
    _log.info('[%s] taxo triples=%d (%.2f, %.2f of total)' % (time.ctime(), len(taxo_dict['p']), len(taxo_dict['p'])/len(train_triples), len(taxo_dict['c'])/len(train_triples)))
    _log.info('adj_taxo_p nnz=%d, shape=%s, adj_taxo_c nnz=%d, shape=%s' % (adj_taxo_p.count_nonzero(), adj_taxo_p.shape, adj_taxo_c.count_nonzero(), adj_taxo_c.shape))

    # Build model
    model = TaxoTransE(len(ent_vocab), len(rel_vocab), norm=opt['dist_norm'],
                       dim=opt['emb_dim'], aggre_type=opt['aggre_type'])
    checkpoint = th.load(checkpoint_path)   # load checkpoint
    model.load_state_dict(checkpoint)
    model = model.to(device)
    _log.info('[%s] model checkpoint load SUCCESS' % (time.ctime()))

    # test
    model.eval()
    hits_1, hits_3, hits_10, mrr, metrics_rels = test(model, test_iter, len(ent_vocab), device, all_triples_map, adj_taxo_p, adj_taxo_c)
    _log.info('[%s] TEST on best model, hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), hits_1, hits_3, hits_10, mrr))
    taxo_rels = get_taxo_relations(opt['corpus_type'])
    taxo_rels = [rel_vocab[_] for _ in taxo_rels]
    (taxo_hits_1, taxo_hits_3, taxo_hits_10, taxo_mrr), (nontaxo_hits_1, nontaxo_hits_3, nontaxo_hits_10, nontaxo_mrr) = cal_taxo_nontaxo_metrics(metrics_rels, taxo_rels)
    _log.info('[%s] TEST on best model, Taxo - hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), taxo_hits_1, taxo_hits_3, taxo_hits_10, taxo_mrr))
    _log.info('[%s] TEST on best model, NonTaxo- hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), nontaxo_hits_1, nontaxo_hits_3, nontaxo_hits_10, nontaxo_mrr))

    if opt['corpus_type'] == 'WN18RR':
        wordnet_def = load_WN18RR_definition()
        case_study(model, test_iter, len(ent_vocab), device, all_triples_map, adj_taxo_p, adj_taxo_c, ent_vocab, rel_vocab, wordnet_def)
    else:
        case_study(model, test_iter, len(ent_vocab), device, all_triples_map, adj_taxo_p, adj_taxo_c, ent_vocab, rel_vocab)

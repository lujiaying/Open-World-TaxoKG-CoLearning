"""
Test Scripts
Author: Anonymous Siamese
Create Date: Oct 6, 2021
"""
import time
import os
import json
import random
from typing import Tuple

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sacred

from model.data_loader import get_concept_tok_tensor, BatchType, CompGCNCGCTripleDst, CompGCNOLPTripleDst
from model.data_loader_HAKEGCN import prepare_ingredients_HAKEGCN
from model.data_loader import HAKETrainDst
from model.TaxoRelGraph import TokenEncoder
from model.HAKE import HAKEGCNEncoder, HAKEGCNScorer
from model.RGCN import RGCN, DistMultDecoder
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics
from utils.metrics import cal_Shannon_diversity_index, cal_freshness_per_sample

# Sacred Setup
ex = sacred.Experiment('test_RGCN')


def produce_human_eval_cg_triples(tok_encoder: th.nn.Module, gcn_encoder: th.nn.Module,
                                  scorer: th.nn.Module, cg_iter: DataLoader,
                                  tok_vocab: dict, all_phrase2id: dict, test_G: dgl.DGLGraph,
                                  test_g_nid_map: dict, concept_vocab: dict, opt: dict,
                                  device: th.device, out_path: str) -> tuple:
    # pre-compute concept embs
    with th.no_grad():
        ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
        ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
        node_embs = ment_tok_embs[test_G.ndata['phrid']]  # (n_cnt, tok_emb)
        edge_embs = ment_tok_embs[test_G.edata['phrid']]  # (e_cng, tok_emb)
        node_embs = gcn_encoder(test_G, node_embs, edge_embs)
        cep_nids = [test_g_nid_map[cep] for cep in concept_vocab.keys()]
        cep_embs = node_embs[cep_nids]   # (cep_cnt, h)
    cep_cnt = cep_embs.size(0)

    id2phrase = {v: k for k, v in all_phrase2id.items()}
    id2concept = {v: k for k, v in concept_vocab.items()}
    fwrite = open(out_path, 'w')
    cnt = 0
    macro_freshness = []
    all_preds = []
    with th.no_grad():
        for (h_nids, r_pids, cep_ids_l) in cg_iter:
            head_embs = node_embs[h_nids.to(device)]      # (B, h)
            rel_embs = ment_tok_embs[r_pids.to(device)]   # (B, tok_h)
            # rel_embs = gcn_encoder.encode_relation(rel_embs)  # (B, h)
            B = head_embs.size(0)
            cep_embs_batch = cep_embs.unsqueeze(0).expand(B, cep_cnt, -1)  # (B, cep_cnt, emb)
            cep_embs_batch = cep_embs_batch.reshape(B*cep_cnt, -1)
            pred_scores = scorer((head_embs, rel_embs, cep_embs_batch), BatchType.TAIL_BATCH)   # (B, cep_cnt)
            pred_scores = pred_scores.argsort(dim=1, descending=True)
            for i in range(B):
                ent = test_G.ndata['phrid'][h_nids[i]]
                ent = id2phrase[ent.item()]
                gold_cepts = [id2concept[_] for _ in cep_ids_l[i]]
                pred_cepts = [id2concept[_] for _ in pred_scores[i, :5].tolist()]
                all_preds.append(pred_cepts)
                macro_freshness.append(cal_freshness_per_sample(gold_cepts, pred_cepts))
                out_line = '%s\t%s\t%s\n' % (ent, ','.join(gold_cepts), ','.join(pred_cepts))
                fwrite.write(out_line)
                cnt += 1
                if cnt > 400:
                    break
            if cnt > 400:
                break
    fwrite.close()
    macro_freshness = sum(macro_freshness) / len(macro_freshness)
    diversity = cal_Shannon_diversity_index(all_preds)
    print('CGC freshness=%.2f, diversity=%.2f' % (macro_freshness, diversity))
    return


def produce_human_eval_okg_triples(tok_encoder: th.nn.Module, gcn_encoder: th.nn.Module, scorer: th.nn.Module,
                                   olp_iter: DataLoader, tok_vocab: dict, all_phrase2id: dict,
                                   test_G: dgl.DGLGraph, test_g_nid_map: dict, candidate_vocab: dict,
                                   opt: dict, device: th.device, all_triple_ids_map: dict,
                                   out_path: str):
    with th.no_grad():
        ment_toks, ment_tok_lens = get_concept_tok_tensor(all_phrase2id, tok_vocab)
        ment_tok_embs = tok_encoder(ment_toks.to(device), ment_tok_lens.to(device))
        node_embs = ment_tok_embs[test_G.ndata['phrid']]  # (n_cnt, tok_emb)
        edge_embs = ment_tok_embs[test_G.edata['phrid']]  # (e_cng, tok_emb)
        node_embs = gcn_encoder(test_G, node_embs, edge_embs)  # (n_cnt, h)
        cand_nids = [test_g_nid_map[cand] for cand in candidate_vocab.keys()]
        cand_embs = node_embs[cand_nids]  # (pool_cnt, h)
    pool_cnt = cand_embs.size(0)

    id2ment = {v: k for k, v in candidate_vocab.items()}
    id2phr = {v: k for k, v in all_phrase2id.items()}
    fwrite = open(out_path, 'w')
    cnt = 0
    macro_freshness = []
    all_preds = []
    visited_hr = set()
    with th.no_grad():
        for (h_batch, r_batch, t_batch) in olp_iter:
            B = h_batch.size(0)
            cand_embs_batch = cand_embs.unsqueeze(0).expand(B, pool_cnt, -1)   # (B, pool_cnt, h)
            cand_embs_batch = cand_embs_batch.reshape(B*pool_cnt, -1)
            h_embs = cand_embs[h_batch.to(device)]
            r_embs = ment_tok_embs[r_batch.to(device)]    # (B, tok_emb)
            # r_embs = gcn_encoder.encode_relation(r_embs)  # (B, h)
            t_embs = cand_embs[t_batch.to(device)]
            h_mids = h_batch.tolist()
            r_pids = r_batch.tolist()
            t_mids = t_batch.tolist()
            # tail pred
            pred_tails = scorer((h_embs, r_embs, cand_embs_batch), BatchType.TAIL_BATCH)  # (B, ment_cnt)
            pred_tails = pred_tails.argsort(dim=1, descending=True)
            # head pred
            pred_heads = scorer((cand_embs_batch, r_embs, t_embs), BatchType.HEAD_BATCH)  # (B, ment_cnt)
            pred_heads = pred_heads.argsort(dim=1, descending=True)
            for i in range(B):
                h_phrase = id2ment[h_mids[i]]
                r_phrase = id2phr[r_pids[i]]
                t_phrase = id2ment[t_mids[i]]
                if (h_phrase, r_phrase) in visited_hr:
                    continue
                visited_hr.add((h_phrase, r_phrase))
                gold_t_phrases = all_triple_ids_map['h'][(h_mids[i], r_pids[i])]
                gold_t_phrases = [id2ment[_] for _ in gold_t_phrases]
                pred_t_phrases = pred_tails[i, :5].tolist()
                pred_t_phrases = [id2ment[_] for _ in pred_t_phrases]
                all_preds.append(pred_t_phrases)
                macro_freshness.append(cal_freshness_per_sample(gold_t_phrases, pred_t_phrases))
                out_line = '%s-> %s\t%s\t%s\n' % (h_phrase, r_phrase, ','.join(gold_t_phrases),
                                                  ','.join(pred_t_phrases))
                fwrite.write(out_line)
                gold_h_phrases = all_triple_ids_map['t'][(t_mids[i], r_pids[i])]
                gold_h_phrases = [id2ment[_] for _ in gold_h_phrases]
                pred_h_phrases = pred_heads[i, :5].tolist()
                pred_h_phrases = [id2ment[_] for _ in pred_h_phrases]
                all_preds.append(pred_h_phrases)
                macro_freshness.append(cal_freshness_per_sample(gold_h_phrases, pred_h_phrases))
                out_line = '%s <-%s\t%s\t%s\n' % (t_phrase, r_phrase, ','.join(gold_h_phrases),
                                                  ','.join(pred_h_phrases))
                fwrite.write(out_line)
                cnt += 2
                if cnt > 400:
                    break
            if cnt > 400:
                break
    fwrite.close()
    macro_freshness = sum(macro_freshness) / len(macro_freshness)
    diversity = cal_Shannon_diversity_index(all_preds)
    print('OLP freshness=%.2f, diversity=%.2f' % (macro_freshness, diversity))
    return


@ex.config
def my_config():
    config_path = ''
    checkpoint_path = ''
    human_eval_path = ''


@ex.automain
def test_model(config_path, checkpoint_path, human_eval_path, _run, _log):
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
    # Load corpus
    train_set_head_batch, train_set_tail_batch,\
        dev_cg_set, test_cg_set, dev_olp_set, test_olp_set,\
        all_triple_ids_map, olp_ment_vocab, tok_vocab, concept_vocab,\
        all_phrase2id, train_G, train_g_nid_map, test_G, test_g_nid_map\
        = prepare_ingredients_HAKEGCN(dataset_dir, opt['neg_method'], opt['neg_size'])
    _log.info('[%s] Load dataset Done, len=%d(tr), %d(CGC-dev)|%d(OLP-dev), %d(CGC-tst)|%d(OLP-tst)' % (time.ctime(),
              len(train_set_head_batch), len(dev_cg_set), len(dev_olp_set), len(test_cg_set), len(test_olp_set)))
    _log.info('Train G info=%s; Test G info=%s' % (train_G, test_G))
    test_cg_iter = DataLoader(test_cg_set, collate_fn=CompGCNCGCTripleDst.collate_fn,
                              batch_size=opt['batch_size'], shuffle=False)
    test_olp_iter = DataLoader(test_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    _log.info('corpus=%s, #Tok=%d, #Mention=%d, #Concept=%d' % (opt['dataset_type'], len(tok_vocab),
              len(all_phrase2id), len(concept_vocab)))
    # Build model
    tok_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim']).to(device)
    gcn_encoder = RGCN(opt['emb_dim'], opt['num_basis'], opt['g_self_dropout'],
                       opt['g_other_dropout']).to(device)
    scorer = DistMultDecoder(opt['emb_dim'], opt['emb_dropout']).to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    # Load best checkpoint
    checkpoint = th.load(checkpoint_path)
    tok_encoder.load_state_dict(checkpoint['tok_encoder'])
    tok_encoder = tok_encoder.to(device)
    tok_encoder.eval()
    gcn_encoder.load_state_dict(checkpoint['gcn_encoder'])
    gcn_encoder = gcn_encoder.to(device)
    gcn_encoder.eval()
    scorer.load_state_dict(checkpoint['scorer'])
    scorer = scorer.to(device)
    scorer.eval()

    test_G = test_G.to(device)
    if not os.path.exists(human_eval_path):
        os.makedirs(human_eval_path)
    cgc_out_path = '%s/cgc_preds.csv' % (human_eval_path)
    produce_human_eval_cg_triples(tok_encoder, gcn_encoder, scorer, test_cg_iter,
                                  tok_vocab, all_phrase2id, test_G, test_g_nid_map,
                                  concept_vocab, opt, device, cgc_out_path)
    olp_out_path = '%s/olp_preds.csv' % (human_eval_path)
    produce_human_eval_okg_triples(tok_encoder, gcn_encoder, scorer,
                                   test_olp_iter, tok_vocab, all_phrase2id,
                                   test_G, test_g_nid_map,
                                   olp_ment_vocab, opt, device, all_triple_ids_map,
                                   olp_out_path)

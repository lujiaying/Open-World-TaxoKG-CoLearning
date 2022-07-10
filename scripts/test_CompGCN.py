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
from utils.metrics import cal_Shannon_diversity_index, cal_freshness_per_sample, cal_Pielou_eveness_index

# Sacred Setup to keep everything in record
ex = sacred.Experiment('test_CompGCN')

N_diversity_CG = 25
N_diversity_OKG = 50


def produce_human_eval_cg_triples(compgcn_transe: th.nn.Module, test_iter: DataLoader,
                                  all_node_embs: th.tensor, all_edge_embs: th.tensor,
                                  node_vocab: dict, concept_vocab: dict, device: th.device,
                                  out_path: str):
    cid_nid_mapping = [-1 for _ in range(len(concept_vocab))]
    for cep, cid in concept_vocab.items():
        nid = node_vocab[cep]
        cid_nid_mapping[cid] = nid
    cep_embs = all_node_embs[cid_nid_mapping]   # (cep_cnt, emb)
    id2node_phr = {v: k for k, v in node_vocab.items()}
    id2concept = {v: k for k, v in concept_vocab.items()}
    fwrite = open(out_path, 'w')
    cnt = 0
    macro_freshness = []
    all_preds = []
    with th.no_grad():
        for hids, rids, cep_ids_l in test_iter:
            B = hids.size(0)
            h_embs = all_node_embs[hids.to(device)]
            r_embs = all_edge_embs[rids.to(device)]
            cep_pred = compgcn_transe.predict(h_embs, r_embs, None, cep_embs, False)  # (B, cand_cnt)
            cep_pred = cep_pred.argsort(dim=1, descending=True)
            for i in range(B):
                ent = id2node_phr[hids[i].item()]
                gold_cepts = [id2concept[_] for _ in cep_ids_l[i]]
                pred_cepts = [id2concept[_] for _ in cep_pred[i, :N_diversity_CG].tolist()]
                all_preds.append(pred_cepts)
                macro_freshness.append(cal_freshness_per_sample(gold_cepts, pred_cepts[:5]))
                out_line = '%s\t%s\t%s\n' % (ent, ','.join(gold_cepts), ','.join(pred_cepts))
                fwrite.write(out_line)
                cnt += 1
                if cnt > 400:
                    break
            if cnt > 400:
                break
    fwrite.close()
    macro_freshness = sum(macro_freshness) / len(macro_freshness)
    # diversity = cal_Shannon_diversity_index(all_preds)
    diversity = cal_Pielou_eveness_index(all_preds)
    print('CGC freshness=%.3f, diversity=%.3f' % (macro_freshness, diversity))
    return


def produce_human_eval_okg_triples(compgcn_transe: th.nn.Module, test_iter: DataLoader,
                                   all_node_embs: th.tensor, all_edge_embs: th.tensor, node_vocab: dict,
                                   edge_vocab: dict, mention_vocab: dict, all_oie_triples_map: dict,
                                   device: th.device, out_path: str):
    mid_nid_mapping = [-1 for _ in range(len(mention_vocab))]
    for m, mid in mention_vocab.items():
        nid = node_vocab[m]
        mid_nid_mapping[mid] = nid
    ment_embs = all_node_embs[mid_nid_mapping]  # (ment_cnt, emb)

    id2ment = {v: k for k, v in mention_vocab.items()}
    id2edge = {v: k for k, v in edge_vocab.items()}
    fwrite = open(out_path, 'w')
    cnt = 0
    macro_freshness = []
    all_preds = []
    visited_hr = set()
    with th.no_grad():
        for (sids, rids, oids) in test_iter:
            B = sids.size(0)
            sids = sids.to(device)
            rids = rids.to(device)
            oids = oids.to(device)
            subj_embs = ment_embs[sids]
            rel_embs = all_edge_embs[rids]
            obj_embs = ment_embs[oids]
            # tail pred
            pred_tails = compgcn_transe.predict(subj_embs, rel_embs, obj_embs, ment_embs, False)  # (B, ment_cnt)
            pred_tails = pred_tails.argsort(dim=1, descending=True)
            # head pred
            pred_heads = compgcn_transe.predict(subj_embs, rel_embs, obj_embs, ment_embs, True)  # (B, ment_cnt)
            pred_heads = pred_heads.argsort(dim=1, descending=True)
            for i in range(B):
                sid = sids[i].item()
                rid = rids[i].item()
                oid = oids[i].item()
                h_phrase = id2ment[sid]
                r_phrase = id2edge[rid]
                t_phrase = id2ment[oid]
                if (h_phrase, r_phrase) in visited_hr:
                    continue
                visited_hr.add((h_phrase, r_phrase))
                gold_t_phrases = all_oie_triples_map['h'][(sid, rid)]
                gold_t_phrases = [id2ment[_] for _ in gold_t_phrases]
                pred_t_phrases = pred_tails[i, :N_diversity_OKG].tolist()
                pred_t_phrases = [id2ment[_] for _ in pred_t_phrases]
                all_preds.append(pred_t_phrases)
                macro_freshness.append(cal_freshness_per_sample(gold_t_phrases, pred_t_phrases[:5]))
                out_line = '%s-> %s\t%s\t%s\n' % (h_phrase, r_phrase, ','.join(gold_t_phrases),
                                                  ','.join(pred_t_phrases[:5]))
                fwrite.write(out_line)
                gold_h_phrases = all_oie_triples_map['t'][(oid, rid)]
                gold_h_phrases = [id2ment[_] for _ in gold_h_phrases]
                pred_h_phrases = pred_heads[i, :N_diversity_OKG].tolist()
                pred_h_phrases = [id2ment[_] for _ in pred_h_phrases]
                all_preds.append(pred_h_phrases)
                macro_freshness.append(cal_freshness_per_sample(gold_h_phrases, pred_h_phrases[:5]))
                out_line = '%s <-%s\t%s\t%s\n' % (t_phrase, r_phrase, ','.join(gold_h_phrases),
                                                  ','.join(pred_h_phrases[:5]))
                fwrite.write(out_line)
                cnt += 2
                if cnt > 400:
                    break
            if cnt > 400:
                break
    fwrite.close()
    macro_freshness = sum(macro_freshness) / len(macro_freshness)
    # diversity = cal_Shannon_diversity_index(all_preds)
    diversity = cal_Pielou_eveness_index(all_preds)
    print('OLP freshness=%.3f, diversity=%.3f' % (macro_freshness, diversity))
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
    (train_set, dev_CGC_set, test_CGC_set,
     dev_OLP_set, test_OLP_set,
     tok_vocab, node_vocab, edge_vocab,
     mention_vocab, concept_vocab, all_oie_triples_map) = prepare_ingredients_CompGCN(dataset_dir)
    _log.info('[%s] #node_vocab=%d, #edge_vocab=%d, #ment_vocab=%d, #cep_vocab=%d' % (time.ctime(),
              len(node_vocab), len(edge_vocab), len(mention_vocab), len(concept_vocab)))
    test_CGC_iter = DataLoader(test_CGC_set, collate_fn=CompGCNCGCTripleDst.collate_fn,
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
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    # Load best checkpoint
    checkpoint = th.load(checkpoint_path)
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
    if not os.path.exists(human_eval_path):
        os.makedirs(human_eval_path)
    cgc_out_path = '%s/cgc_preds.csv' % (human_eval_path)
    produce_human_eval_cg_triples(compgcn_transe, test_CGC_iter, node_embs,
                                  edge_embs, node_vocab, concept_vocab, device,
                                  cgc_out_path)
    olp_out_path = '%s/olp_preds.csv' % (human_eval_path)
    produce_human_eval_okg_triples(compgcn_transe, test_OLP_iter, node_embs,
                                   edge_embs, node_vocab, edge_vocab, mention_vocab,
                                   all_oie_triples_map, device, olp_out_path)

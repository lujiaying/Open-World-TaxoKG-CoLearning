"""
Test Scripts
Author: Anonymous Siamese
Create Date: Aug 26, 2021
"""
import time
import os
import json
import random
from typing import Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sacred

from model.data_loader import prepare_ingredients_HAKE, get_concept_tok_tensor
from model.data_loader import collate_fn_CGCpairs, CompGCNOLPTripleDst
from model.data_loader import HAKETrainDst, BatchType
from model.TaxoRelGraph import TokenEncoder
from model.HAKE import HAKE
from .train_openHAKE import test_CGC_task, test_OLP_task
from utils.metrics import get_phrase_from_dictid, cal_Shannon_diversity_index, cal_freshness_per_sample, cal_Pielou_eveness_index

# Sacred Setup
ex = sacred.Experiment('test_HAKE')

N_diversity_CG = 30
N_diversity_OKG = 50


def produce_human_eval_cg_triples(tok_encoder: th.nn.Module, scorer: th.nn.Module, cg_iter: DataLoader,
                                  tok_vocab: dict, concept_vocab: dict, device: th.device, out_path: str):
    """
    To produce concepts for entities already in CG
    """
    id2tok = {v: k for k, v in tok_vocab.items()}
    id2cept = {v: k for k, v in concept_vocab.items()}
    all_concepts, all_cep_lens = get_concept_tok_tensor(concept_vocab, tok_vocab)
    all_concept_embs = tok_encoder(all_concepts.to(device), all_cep_lens.to(device))  # (cep_cnt, emb_d)
    fwrite = open(out_path, 'w')
    cnt = 0
    macro_freshness = []
    all_preds = []
    with th.no_grad():
        for (ent_batch, gold_ceps_batch, ent_lens) in cg_iter:
            ent_embs = tok_encoder(ent_batch.to(device), ent_lens.to(device))  # (B, emb_d)
            B = ent_batch.size(0)
            r_batch = ent_batch.new_tensor([tok_vocab["IsA"] for _ in range(B)]).unsqueeze(-1)   # (B, 1)
            r_lens = ent_lens.new_ones(B)   # (B, )
            r_embs = tok_encoder(r_batch.to(device), r_lens.to(device))  # (B, emb_d)
            all_concept_embs_batch = all_concept_embs.unsqueeze(0).expand(B, -1, -1)  # (B, ment_cnt, emb)
            pred_scores = scorer((ent_embs, r_embs, all_concept_embs_batch), BatchType.TAIL_BATCH)   # (B, cep_cnt)
            preds_idx = pred_scores.argsort(dim=1, descending=True)
            for i in range(B):
                ent = get_phrase_from_dictid(ent_batch[i], ent_lens[i], id2tok)
                if '<UNK>' in ent:
                    continue
                gold_cepts = [id2cept[_] for _ in gold_ceps_batch[i]]
                pred_cepts = [id2cept[_] for _ in preds_idx[i, :N_diversity_CG].tolist()]
                all_preds.append(pred_cepts)
                macro_freshness.append(cal_freshness_per_sample(gold_cepts, pred_cepts[:5]))
                out_line = '%s\t%s\t%s\n' % (ent, ','.join(gold_cepts), ','.join(pred_cepts[:5]))
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


def produce_human_eval_okg_triples(tok_encoder: th.nn.Module, scorer: th.nn.Module,
                                   oie_iter: DataLoader, tok_vocab: dict, mention_vocab: dict,
                                   rel_vocab: dict, device: th.device, all_oie_triples_map: dict,
                                   out_path: str) -> tuple:
    all_mentions, all_mention_lens = get_concept_tok_tensor(mention_vocab, tok_vocab)
    all_mention_embs = tok_encoder(all_mentions.to(device), all_mention_lens.to(device))  # (ment_cnt, emb)
    all_rels, all_rel_lens = get_concept_tok_tensor(rel_vocab, tok_vocab)
    all_rel_embs = tok_encoder(all_rels.to(device), all_rel_lens.to(device))  # (rel_cnt, emb)

    id2ment = {v: k for k, v in mention_vocab.items()}
    id2rel = {v: k for k, v in rel_vocab.items()}
    fwrite = open(out_path, 'w')
    cnt = 0
    macro_freshness = []
    all_preds = []
    visited_hr = set()
    with th.no_grad():
        for (h_batch, r_batch, t_batch) in oie_iter:
            B = h_batch.size(0)
            all_mention_embs_batch = all_mention_embs.unsqueeze(0).expand(B, -1, -1)  # (B, ment_cnt, emb)
            h_embs = all_mention_embs[h_batch.to(device)]
            r_embs = all_rel_embs[r_batch.to(device)]
            t_embs = all_mention_embs[t_batch.to(device)]
            h_mids = h_batch.tolist()
            r_rids = r_batch.tolist()
            t_mids = t_batch.tolist()
            # tail pred
            pred_tails = scorer((h_embs, r_embs, all_mention_embs_batch), BatchType.TAIL_BATCH)  # (B, ment_cnt)
            pred_tails = pred_tails.argsort(dim=1, descending=True)
            # head pred
            pred_heads = scorer((all_mention_embs_batch, r_embs, t_embs), BatchType.HEAD_BATCH)  # (B, ment_cnt)
            pred_heads = pred_heads.argsort(dim=1, descending=True)
            for i in range(B):
                h_phrase = id2ment[h_mids[i]]
                r_phrase = id2rel[r_rids[i]]
                t_phrase = id2ment[t_mids[i]]
                if (h_phrase, r_phrase) in visited_hr:
                    continue
                visited_hr.add((h_phrase, r_phrase))
                gold_t_phrases = all_oie_triples_map['h'][(h_mids[i], r_rids[i])]
                gold_t_phrases = [id2ment[_] for _ in gold_t_phrases]
                pred_t_phrases = pred_tails[i, :N_diversity_OKG].tolist()
                pred_t_phrases = [id2ment[_] for _ in pred_t_phrases]
                all_preds.append(pred_t_phrases)
                macro_freshness.append(cal_freshness_per_sample(gold_t_phrases, pred_t_phrases[:5]))
                out_line = '%s-> %s\t%s\t%s\n' % (h_phrase, r_phrase, ','.join(gold_t_phrases),
                                                  ','.join(pred_t_phrases[:5]))
                fwrite.write(out_line)
                gold_h_phrases = all_oie_triples_map['t'][(t_mids[i], r_rids[i])]
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
    do_human_eval = False


@ex.automain
def test_model(config_path, checkpoint_path, do_human_eval, _run, _log):
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
    # device = th.device('cpu')
    # Load corpus
    train_set_head_batch, train_set_tail_batch,\
        dev_cg_set, test_cg_set, dev_olp_set, test_olp_set, concept_vocab,\
        tok_vocab, train_mention_vocab, train_rel_vocab, all_mention_vocab,\
        all_rel_vocab, all_oie_triples_map = prepare_ingredients_HAKE(dataset_dir, opt['negative_size'])
    test_cg_iter = DataLoader(test_cg_set, collate_fn=collate_fn_CGCpairs, batch_size=opt['batch_size'], shuffle=False)
    test_olp_iter = DataLoader(test_olp_set, collate_fn=CompGCNOLPTripleDst.collate_fn,
                               batch_size=opt['batch_size'], shuffle=False)
    # Build model
    tok_encoder = TokenEncoder(len(tok_vocab), opt['emb_dim']).to(device)
    scorer = HAKE(opt['emb_dim'], opt['gamma'], opt['mod_w'], opt['pha_w']).to(device)
    _log.info('[%s] Model build Done. Use device=%s' % (time.ctime(), device))
    # do test
    checkpoint = th.load(checkpoint_path)
    tok_encoder.load_state_dict(checkpoint['tok_encoder'])
    tok_encoder = tok_encoder.to(device)
    tok_encoder.eval()
    scorer.load_state_dict(checkpoint['scorer'])
    scorer = scorer.to(device)
    scorer.eval()
    if do_human_eval is not False:
        if not os.path.exists(do_human_eval):
            os.makedirs(do_human_eval)
        cgc_out_path = '%s/cgc_preds.csv' % (do_human_eval)
        produce_human_eval_cg_triples(tok_encoder, scorer, test_cg_iter, tok_vocab,
                                      concept_vocab, device, cgc_out_path)
        olp_out_path = '%s/olp_preds.csv' % (do_human_eval)
        produce_human_eval_okg_triples(tok_encoder, scorer, test_olp_iter, tok_vocab,
                                       all_mention_vocab, all_rel_vocab, device,
                                       all_oie_triples_map, olp_out_path)
        exit(0)
    MAP, CGC_MRR, P1, P3, P10 = test_CGC_task(tok_encoder, scorer, test_cg_iter,
                                              tok_vocab, concept_vocab, device)
    _run.log_scalar("test.CGC.MAP", MAP)
    _run.log_scalar("test.CGC.MRR", CGC_MRR)
    _run.log_scalar("test.CGC.P@1", P1)
    _run.log_scalar("test.CGC.P@3", P3)
    _run.log_scalar("test.CGC.P@10", P10)
    _log.info('[%s] CGC TEST, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' % (time.ctime(), MAP, CGC_MRR, P1, P3, P10))
    OLP_MRR, H10, H30, H50 = test_OLP_task(tok_encoder, scorer, test_olp_iter, tok_vocab, all_mention_vocab,
                                           all_rel_vocab, device, all_oie_triples_map)
    _run.log_scalar("test.OLP.MRR", OLP_MRR)
    _run.log_scalar("test.OLP.Hits@10", H10)
    _run.log_scalar("test.OLP.Hits@30", H30)
    _run.log_scalar("test.OLP.Hits@50", H50)
    _log.info('[%s] OLP TEST, MRR=%.3f, Hits@10,30,50=%.3f,%.3f,%.3f' % (time.ctime(), OLP_MRR, H10, H30, H50))

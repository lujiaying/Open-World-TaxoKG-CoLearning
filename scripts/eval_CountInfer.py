"""
Evaluate CountInfer
Author: Jiaying Lu
Create Date: Aug 24, 2021
"""
from collections import defaultdict
import time
import random

import tqdm
import numpy as np
import torch as th
import sacred
# from sacred.observers import FileStorageObserver

from model.data_loader import load_cg_pairs, load_oie_triples
from model.CountInfer import CountInfer, NaiveCountInfer
from utils.metrics import cal_AP_atk, cal_reciprocal_rank, cal_OLP_metrics_nontensor


# Sacred Setup to keep everything in record
ex = sacred.Experiment('base-CountInfer')
# ex.observers.append(FileStorageObserver("logs/CountInfer"))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'seed': 27,
           'dataset_type': '',     # MSCG-ReVerb, ..., SEMusic-OPIEC
           'dataset_dir': {
               'MSCG-ReVerb': "data/CGC-OLP-BENCH/MSCG-ReVerb",
               'SEMedical-ReVerb': "data/CGC-OLP-BENCH/SEMedical-ReVerb",
               'SEMusic-ReVerb': "data/CGC-OLP-BENCH/SEMusic-ReVerb",
               'MSCG-OPIEC': "data/CGC-OLP-BENCH/MSCG-OPIEC",
               'SEMedical-OPIEC': "data/CGC-OLP-BENCH/SEMedical-OPIEC",
               'SEMusic-OPIEC': "data/CGC-OLP-BENCH/SEMusic-OPIEC",
               },
           'model_type': 'CountInfer'
            }


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))
    dataset_dir = opt['dataset_dir'][opt['dataset_type']]
    _log.info('Experiments on %s' % (dataset_dir))
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    oie_train_set = load_oie_triples(oie_train_path)
    oie_dev_path = '%s/oie_triples.dev.txt' % (dataset_dir)
    oie_dev_set = load_oie_triples(oie_dev_path)
    oie_test_path = '%s/oie_triples.test.txt' % (dataset_dir)
    oie_test_set = load_oie_triples(oie_test_path)
    cgc_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cgc_train_set = load_cg_pairs(cgc_train_path)
    cgc_dev_path = '%s/cg_pairs.dev.txt' % (dataset_dir)
    cgc_dev_set = load_cg_pairs(cgc_dev_path)
    cgc_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    cgc_test_set = load_cg_pairs(cgc_test_path)
    if opt['model_type'] == 'CountInfer':
        model = CountInfer(cgc_train_set, oie_train_set)
    elif opt['model_type'] == 'NaiveCountInfer':
        model = NaiveCountInfer(cgc_train_set, oie_train_set)
        model.update_all_concepts(cgc_train_set, cgc_dev_set, cgc_test_set)
        model.update_all_mentions(oie_train_set, oie_dev_set, oie_test_set)
    else:
        _log.error('Model Type=%s Invalid' % (opt['model_type']))
        exit(-1)
    all_triples_map = {'h': defaultdict(set),
                       't': defaultdict(set)}
    for (h, r, t) in (oie_train_set + oie_dev_set + oie_test_set):
        all_triples_map['h'][(h, r)].add(t)
        all_triples_map['t'][(t, r)].add(h)
    _log.info('Model Load Train Resource Done')

    topk = 15
    MAP = []     # MAP@15
    MRR = []
    P1 = []
    P3 = []
    P10 = []
    for ent, gold_ceps in tqdm.tqdm(cgc_test_set.items()):
        taxo_results = model.infer_taxonomy(ent)
        preds = [_[0] for _ in taxo_results]
        AP = cal_AP_atk(gold_ceps, preds, k=topk)
        MAP.append(AP)
        RR = cal_reciprocal_rank(gold_ceps, preds)
        MRR.append(RR)
        gold_ceps = set(gold_ceps)
        p1 = len(gold_ceps.intersection(set(preds[:1]))) / 1.0
        p3 = len(gold_ceps.intersection(set(preds[:3]))) / 3.0
        p10 = len(gold_ceps.intersection(set(preds[:10]))) / 10.0
        P1.append(p1)
        P3.append(p3)
        P10.append(p10)
    MAP = sum(MAP) / len(MAP)
    MRR = sum(MRR) / len(MRR)
    P1 = sum(P1) / len(P1)
    P3 = sum(P3) / len(P3)
    P10 = sum(P10) / len(P10)
    _log.info('CGC evaluate, MAP=%.3f, MRR=%.3f, P@1,3,10=%.3f,%.3f,%.3f' %
              (MAP, MRR, P1, P3, P10))
    MRR = []
    H10 = []
    H30 = []
    H50 = []
    for (h, r, t) in tqdm.tqdm(oie_test_set):
        t_preds, h_preds = model.infer_relation(h, r, t)
        rr, h10, h30, h50 = cal_OLP_metrics_nontensor(t_preds, h, r, t, True, all_triples_map)
        MRR.append(rr)
        H10.append(h10)
        H30.append(h30)
        H50.append(h50)
        rr, h10, h30, h50 = cal_OLP_metrics_nontensor(h_preds, h, r, t, False, all_triples_map)
        MRR.append(rr)
        H10.append(h10)
        H30.append(h30)
        H50.append(h50)
    MRR = sum(MRR) / len(MRR)
    H10 = sum(H10) / len(H10)
    H30 = sum(H30) / len(H30)
    H50 = sum(H50) / len(H50)
    _log.info('OLP evaluate, MRR=%.3f, H@10,30,50=%.3f,%.3f,%.3f' % (MRR, H10, H30, H50))

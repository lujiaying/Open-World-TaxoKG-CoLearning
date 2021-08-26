"""
Evaluate CountInfer
Author: Jiaying Lu
Create Date: Aug 24, 2021
"""
import sacred
from sacred.observers import FileStorageObserver

from model.data_loader import load_cg_pairs, load_oie_triples
from model.CountInfer import CountInfer


# Sacred Setup to keep everything in record
ex = sacred.Experiment('base-CountInfer')
# ex.observers.append(FileStorageObserver("logs/CountInfer"))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'dataset_type': '',     # MSCG-ReVerb, ..., SEMusic-OPIEC
           'dataset_dir': {
               'MSCG-ReVerb': "data/CGC-OLP-BENCH/MSCG-ReVerb",
               'SEMedical-ReVerb': "data/CGC-OLP-BENCH/SEMedical-ReVerb",
               'SEMusic-ReVerb': "data/CGC-OLP-BENCH/SEMusic-ReVerb",
               'MSCG-OPIEC': "data/CGC-OLP-BENCH/MSCG-OPIEC",
               'SEMedical-OPIEC': "data/CGC-OLP-BENCH/SEMedical-OPIEC",
               'SEMusic-OPIEC': "data/CGC-OLP-BENCH/SEMusic-OPIEC",
               },
            }


@ex.automain
def main(opt, _run, _log):
    dataset_dir = opt['dataset_dir'][opt['dataset_type']]
    cgc_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    oie_train_set = load_oie_triples(oie_train_path)
    model = CountInfer(load_cg_pairs(cgc_train_path), oie_train_set)
    _log.info('Model Load Train Resource Done')

    cgc_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    cgc_test_set = load_cg_pairs(cgc_test_path)
    cnt = 0
    for ent, concepts in cgc_test_set.items():
        taxo_results = model.infer_taxonomy(ent)
        print('CGC ground truth:', concepts)
        print(taxo_results[:15])
        cnt += 1
        if cnt > 2:
            break
    oie_test_path = '%s/oie_triples.test.txt' % (dataset_dir)
    oie_test_set = load_oie_triples(oie_test_path)
    cnt = 0
    for (h, r, t) in oie_test_set:
        print('oie ground truth:', h, r, t)
        t_score = model.infer_relation(h, r, t)
        print('t score', t_score)
        cnt += 1
        if cnt > 4:
            break

"""
Analysis entity, relation distribution
Author: Anonymous Siamese
Create Date: Oct 13, 2021
"""

from collections import Counter

import tqdm
import numpy as np


def analysis_taxoKG_triples(aligend_cg_path: str, aligned_okg_path: str):
    cept_counter = Counter()
    ent_counter = Counter()
    rel_counter = Counter()
    with open(aligend_cg_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            ent = line_list[0]
            ent_counter[ent] += 1
            for cept in line_list[1:]:
                cept_counter[cept] += 1
    with open(aligned_okg_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            if len(line_list) < 3:
                continue
            s, o, p = line_list[:3]
            ent_counter[s] += 1
            ent_counter[o] += 1
            rel_counter[p] += 1
    cept_cnts = list(cept_counter.values())
    bins = [0, 2, 10, 40, max(cept_cnts)]
    hist, _ = np.histogram(cept_cnts, bins)
    print('concept distribution of bins(%s): %s' % (bins, ' | '.join([str(_/sum(hist)) for _ in hist])))
    ent_cnts = list(ent_counter.values())
    # bins = [0, 10, 30, 50, 100, max(ent_cnts)]
    bins = [0, 2, 10, 40, max(ent_cnts)]
    hist, _ = np.histogram(ent_cnts, bins)
    print('entity distribution of bins(%s): %s' % (bins, ' | '.join([str(_/sum(hist)) for _ in hist])))
    rel_cnts = list(rel_counter.values())
    # bins = [0, 10, 30, 50, 100, max(rel_cnts)]
    bins = [0, 2, 10, 40, max(rel_cnts)]
    hist, _ = np.histogram(rel_cnts, bins)
    print('relation distribution of bins(%s): %s' % (bins, ' | '.join([str(_/sum(hist)) for _ in hist])))


def analysis_unseen_entity_concept_relation(dataset_dir: str):
    test_entity_set = set()
    test_concept_set = set()
    test_relation_set = set()
    with open('%s/cg_pairs.test.txt' % (dataset_dir)) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            ent = line_list[0]
            test_entity_set.add(ent)
            for cept in line_list[1:]:
                test_concept_set.add(cept)
    with open('%s/oie_triples.test.txt' % (dataset_dir)) as fopen:
        for line in tqdm.tqdm(fopen):
            s, p, o = line.strip().split('\t')
            test_entity_set.add(s)
            test_entity_set.add(o)
            test_relation_set.add(p)
    train_entity_set = set()
    train_concept_set = set()
    train_relation_set = set()
    with open('%s/cg_pairs.train.txt' % (dataset_dir)) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            ent = line_list[0]
            train_entity_set.add(ent)
            for cept in line_list[1:]:
                train_concept_set.add(cept)
    with open('%s/oie_triples.train.txt' % (dataset_dir)) as fopen:
        for line in tqdm.tqdm(fopen):
            s, p, o = line.strip().split('\t')
            train_entity_set.add(s)
            train_entity_set.add(o)
            train_relation_set.add(p)
    seen_ent_cnt = len(test_entity_set.intersection(train_entity_set))
    seen_cept_cnt = len(test_concept_set.intersection(train_concept_set))
    seen_rel_cnt = len(test_relation_set.intersection(train_relation_set))
    print('%s - entity=%.3f, concept=%.3f, relation=%.3f' % (dataset_dir,
          seen_ent_cnt/len(test_entity_set), seen_cept_cnt/len(test_concept_set),
          seen_rel_cnt/len(test_relation_set)))


if __name__ == '__main__':
    aligned_concept_path = 'data/MSConceptGraph/data-concept-instance-relations.ReVerb-aligned.txt'
    aligned_openie_path = 'data/ReVerb/reverb_clueweb_tuples.Probase-aligned.txt'
    # analysis_taxoKG_triples(aligned_concept_path, aligned_openie_path)
    aligned_concept_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.ReVerb-aligned.txt'
    aligned_openie_path = 'data/ReVerb/reverb_clueweb_tuples.SemEvalMedical-aligned.txt'
    # analysis_taxoKG_triples(aligned_concept_path, aligned_openie_path)
    aligned_concept_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.ReVerb-aligned.txt'
    aligned_openie_path = 'data/ReVerb/reverb_clueweb_tuples.SemEvalMusic-aligned.txt'
    # analysis_taxoKG_triples(aligned_concept_path, aligned_openie_path)

    aligned_concept_path = 'data/MSConceptGraph/instance-concepts.OPIEC-aligned.txt'
    aligned_openie_path = 'data/OPIEC/OPIEC-Linked-triples.Probase-aligned.txt'
    # analysis_taxoKG_triples(aligned_concept_path, aligned_openie_path)
    aligned_concept_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.OPIEC-aligned.txt'
    aligned_openie_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMedical-aligned.txt'
    # analysis_taxoKG_triples(aligned_concept_path, aligned_openie_path)
    aligned_concept_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.OPIEC-aligned.txt'
    aligned_openie_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMusic-aligned.txt'
    # analysis_taxoKG_triples(aligned_concept_path, aligned_openie_path)

    benchmark_data_dir = 'data/CGC-OLP-BENCH/MSCG-OPIEC'
    benchmark_data_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    benchmark_data_dir = 'data/CGC-OLP-BENCH/SEMusic-OPIEC'
    benchmark_data_dir = 'data/CGC-OLP-BENCH/MSCG-ReVerb'
    benchmark_data_dir = 'data/CGC-OLP-BENCH/SEMedical-ReVerb'
    benchmark_data_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    # analysis_unseen_entity_concept_relation(benchmark_data_dir)

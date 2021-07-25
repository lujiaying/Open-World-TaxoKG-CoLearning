"""
Align entity in Concept Graph and OpenIE facts
Author: Jiaying Lu
Create Date: Jun 29, 2021
"""

import os
import time
from collections import defaultdict
from typing import Tuple, Dict
import random

import tqdm
import numpy as np


def load_Probase(Probase_path: str, do_analysis: bool = False) -> dict:
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    concept_set = set()
    pair_cnt = 0
    with open(Probase_path) as fopen:
        tik = time.perf_counter()
        lines = fopen.readlines()
        elapsed_time = time.perf_counter() - tik
        print('Readlines Probase elapsed_time=%s' % (elapsed_time))
        for line in tqdm.tqdm(lines):
            line_list = line.strip().split('\t')
            pair_cnt += 1
            p = line_list[0]
            c = line_list[1]
            concept_set.add(p)
            concept_pairs[c].add(p)
        del lines
    print('Probase # of pairs=%d, unique # of instances=%d' % (pair_cnt, len(concept_pairs)))
    print('unique # of concepts=%d' % (len(concept_set)))
    if do_analysis:
        analysis_concept_pairs(concept_pairs)
    return concept_pairs


def analysis_concept_pairs(concept_pairs: Dict[str, set], full_analysis: bool = True):
    # concept_pairs: {ent1: {c1, c2, ...}, ent2: {}, ...}
    triple_cnt = 0
    avg_parent = []
    reverse_concept_pairs = defaultdict(set)
    for c, ps in concept_pairs.items():
        triple_cnt += len(ps)
        avg_parent.append(len(ps))
        for p in ps:
            reverse_concept_pairs[p].add(c)
    avg_parent = sum(avg_parent) / len(avg_parent)
    all_concepts = set()
    avg_child = []
    for p, cs in reverse_concept_pairs.items():
        avg_child.append(len(cs))
        all_concepts.add(p)
        all_concepts.update(cs)
    avg_child = sum(avg_child) / len(avg_child)
    print('#entity=%d, #concept=%d, #triple=%d' % (len(concept_pairs), len(reverse_concept_pairs),
                                                   triple_cnt))
    print('Avg #Parent=%.2f, Avg #Child=%.2f' % (avg_parent, avg_child))
    if not full_analysis:
        return
    # cal avg level
    # first find leaf concepts
    # then find concepts with all children exist
    print('#concept=%d' % (len(all_concepts)))
    concept_level = dict()
    to_remove = set()
    for ent in concept_pairs:
        if ent not in reverse_concept_pairs:
            concept_level[ent] = 0.0   # ent only serves as instance
            to_remove.add(ent)
    all_concepts.difference_update(to_remove)
    print('#pure instance=%d' % (len(concept_level)))
    iteration_cnt = 0
    while len(all_concepts) > 0:
        to_remove = set()
        for ent in all_concepts:
            # if all children are in concept_level dict,
            # then remove ent from all_concepts
            if all(child in concept_level for child in reverse_concept_pairs[ent]):
                level = max(concept_level[child] for child in reverse_concept_pairs[ent]) + 1.0
                concept_level[ent] = level
                to_remove.add(ent)
        all_concepts.difference_update(to_remove)
        iteration_cnt += 1
        if iteration_cnt > 20:
            print('deadlock!! remains all_concepts=%d' % (len(all_concepts)))
            # concept graph may contain mutual dependent concept-subconcept pair, e.g. A: B, C;   C: A, D
            # calculate approx level for these
            for ent in all_concepts:
                level = max(concept_level.get(child, 1.0) for child in reverse_concept_pairs[ent]) + 1.0
                concept_level[ent] = level
            # for ent in all_concepts:
            #     unresloved_children = [c for c in reverse_concept_pairs[ent] if c not in concept_level]
            #     print('%s: %s not resloved (total %d)' % (ent, unresloved_children, len(reverse_concept_pairs[ent])))
            break
    concept_level = {k: v for k, v in concept_level.items() if v > 0.0}
    print('Avg level=%.2f' % (sum(concept_level.values()) / len(concept_level)))
    print('Max level=%d' % (max(concept_level.values())))
    # concept instances distribution
    instance_cnts = [len(cs) for cs in reverse_concept_pairs.values()]
    bins = [0, 2, 10, 20, 30, 40, 50, max(instance_cnts)]
    hist, _ = np.histogram(instance_cnts, bins)
    print('concept distribution of bins(%s): %s' % (bins, ' | '.join([str(_) for _ in hist])))


def load_SemEval(data_path: str, gold_path: str, do_analysis: bool = False) -> dict:
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    with open(data_path) as fopen1, open(gold_path) as fopen2:
        for data_line in tqdm.tqdm(fopen1):
            gold_line = fopen2.readline()
            hyponym = data_line.strip().split('\t')[0]
            hypernyms = gold_line.strip().split('\t')
            hypernyms = [h for h in hypernyms if h != hyponym]
            concept_pairs[hyponym].update(hypernyms)
    if do_analysis:
        analysis_concept_pairs(concept_pairs)
    return concept_pairs


def load_merged_SemEval(file_path: str, do_analysis: bool = False) -> dict:
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    with open(file_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            c = line_list[0]
            ps = line_list[1:]
            concept_pairs[c] = ps
    if do_analysis:
        analysis_concept_pairs(concept_pairs)
    return concept_pairs


def analysis_SemEval(train_gold_path: str, test_gold_path: str):
    train_hypernyms = set()
    with open(train_gold_path) as fopen:
        for line in fopen:
            hypernyms = line.strip().split('\t')
            train_hypernyms.update(hypernyms)
    test_hypernyms = set()
    with open(test_gold_path) as fopen:
        for line in fopen:
            hypernyms = line.strip().split('\t')
            test_hypernyms.update(hypernyms)
    print('%d train overlap #%d test = %d' % (len(train_hypernyms), len(test_hypernyms),
                                              len(train_hypernyms.intersection(test_hypernyms))))


def align_Probase_ReVerb(Probase_path: str, ReVerb_path: str, output_path: str, CM_type: str = "Probase"):
    if CM_type == "Probase":
        concept_pairs = load_Probase(Probase_path)
    elif CM_type == "SemEval":
        concept_pairs = load_merged_SemEval(Probase_path)
    else:
        print('ERROR invalid CM_type=%s' % (CM_type))
        exit(0)
    with open(ReVerb_path) as fopen, open(output_path, 'w') as fwrite:
        tik = time.perf_counter()
        lines = fopen.readlines()
        elapsed_time = time.perf_counter() - tik
        print('Readlines ReVerb, elapsed_time=%s' % (elapsed_time))
        for line in tqdm.tqdm(lines):
            line_list = line.strip().split('\t')
            arg1_norm = line_list[4]   # norm is already lowercased
            rel_norm = line_list[5]
            arg2_norm = line_list[6]
            if arg1_norm not in concept_pairs and arg2_norm not in concept_pairs:
                continue
            fwrite.write('%s\t%s\t%s\n' % (arg1_norm, rel_norm, arg2_norm))
        del lines


def _get_lemma_wikilink(tok_list: list) -> Tuple[str, str]:
    lemma = []
    wikilink = tok_list[0]['w_link']['wiki_link']
    for _ in tok_list:
        lemma.append(_['lemma'] if _['lemma'] is not None else _['word'])
    return ' '.join(lemma), wikilink


def align_Probase_OPIEC(Probase_path: str, OPIEC_path: str, OPIEC_aligned_path: str, CM_type: str = "Probase"):
    from avro.datafile import DataFileReader
    from avro.io import DatumReader
    if CM_type == "Probase":
        concept_pairs = load_Probase(Probase_path)
    elif CM_type == "SemEval":
        concept_pairs = load_merged_SemEval(Probase_path)
    else:
        print('ERROR invalid CM_type=%s' % (CM_type))
        exit(0)
    print('Concept Graph=%s Loaded.' % (Probase_path))

    matched_hyponyms = set()
    fwrite = open(OPIEC_aligned_path, 'w')
    if os.path.isfile(OPIEC_path):
        all_paths = [OPIEC_path]
    else:
        all_paths = []
        for fname in os.listdir(OPIEC_path):
            if not fname.endswith('avro'):
                continue
            all_paths.append('%s/%s' % (OPIEC_path, fname))
    for file_path in tqdm.tqdm(all_paths):
        reader = DataFileReader(open(file_path, "rb"), DatumReader())
        for triple in reader:
            # print('subj: %s' % (triple['subject']))
            # print('rel: %s' % (triple['relation']))
            # print('obj: %s' % (triple['object']))
            subj_lem, subj_wikilink = _get_lemma_wikilink(triple['subject'])
            rel_lem, rel_wikilink = _get_lemma_wikilink(triple['relation'])
            obj_lem, obj_wikilink = _get_lemma_wikilink(triple['object'])
            subj_lem = subj_lem.lower()
            obj_lem = obj_lem.lower()
            if subj_lem not in concept_pairs and obj_lem not in concept_pairs:
                continue
            if subj_lem in concept_pairs:
                matched_hyponyms.add(subj_lem)
            if obj_lem in concept_pairs:
                matched_hyponyms.add(obj_lem)
            line = '\t'.join([subj_lem, rel_lem, obj_lem, subj_wikilink, rel_wikilink, obj_wikilink])
            fwrite.write(line + '\n')
        reader.close()
    fwrite.close()
    print('Total %d hyponyms, matched %d hyponyms' % (len(concept_pairs), len(matched_hyponyms)))


def produce_OPIEC_mention_wiki_map(OPIEC_path: str, output_path: str):
    from avro.datafile import DataFileReader
    from avro.io import DatumReader
    wiki_mention_dict = defaultdict(set)
    if os.path.isfile(OPIEC_path):
        all_paths = [OPIEC_path]
    else:
        all_paths = []
        for fname in os.listdir(OPIEC_path):
            if not fname.endswith('avro'):
                continue
            all_paths.append('%s/%s' % (OPIEC_path, fname))
    for file_path in tqdm.tqdm(all_paths):
        reader = DataFileReader(open(file_path, "rb"), DatumReader())
        for triple in reader:
            subj_lem, subj_wikilink = _get_lemma_wikilink(triple['subject'])
            # rel_lem, rel_wikilink = _get_lemma_wikilink(triple['relation'])
            obj_lem, obj_wikilink = _get_lemma_wikilink(triple['object'])
            if subj_wikilink != '':
                wiki_mention_dict[subj_wikilink].add(subj_lem)
            if obj_wikilink != '':
                wiki_mention_dict[obj_wikilink].add(obj_lem)
        reader.close()
    with open(output_path, 'w') as fwrite:
        for wiki, mentions in tqdm.tqdm(wiki_mention_dict.items()):
            fwrite.write('%s\t%s\n' % (wiki, '\t'.join(mentions)))


def merge_SemEval_train_dev_test_sets(dataset_dir: str, dataset_name: str):
    concept_pairs = defaultdict(set)   # c: {p1, p2, ...}
    for split in ['training', 'test', 'trial']:
        data_path = '%s/%s/data/%s.%s.data.txt' % (dataset_dir, split, dataset_name, split)
        gold_path = '%s/%s/gold/%s.%s.gold.txt' % (dataset_dir, split, dataset_name, split)
        concept_pairs.update(load_SemEval(data_path, gold_path))
    out_path = '%s/%s.merged_pairs.txt' % (dataset_dir, dataset_name)
    with open(out_path, 'w') as fwrite:
        for c, ps in concept_pairs.items():
            c = c.lower()
            ps = [p.lower() for p in ps]
            fwrite.write('%s\t%s\n' % (c, '\t'.join(ps)))


def load_OPIEC_wiki_mention_map(OPIEC_mention_map_path: str) -> Tuple[dict, dict]:
    wiki_mention_map = defaultdict(set)
    mention_wiki_map = defaultdict(set)
    with open(OPIEC_mention_map_path) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            wiki = line_list[0]
            mentions = set([_.lower() for _ in line_list[1:]])
            wiki_mention_map[wiki] = mentions
            for m in mentions:
                mention_wiki_map[m].add(wiki)
    return wiki_mention_map, mention_wiki_map


def align_SemEval_OPIEC_mentions(SemEval_path: str, OPIEC_mention_map_path: str, aligned_SemEval_path: str):
    """
    Store filtered SemEval concept pairs
    find SemEval hyponyms that exsit in OPIEC mentions
    """
    wiki_mention_map, mention_wiki_map = load_OPIEC_wiki_mention_map(OPIEC_mention_map_path)
    concept_maps = defaultdict(set)
    with open(SemEval_path) as fopen, open(aligned_SemEval_path, 'w') as fwrite:
        for line in fopen:
            line_list = line.strip().split('\t')
            c = line_list[0]
            if c not in mention_wiki_map:
                continue
            fwrite.write(line)
            ps = set(line_list[1:])
            concept_maps[c] = ps
    # analysis
    analysis_concept_pairs(concept_maps)


def align_Probase_OPIEC_mentions(Probase_path: str, OPIEC_mention_map_path: str, aligned_Probase_path: str):
    """
    Store filtered concept pairs
    """
    wiki_mention_map, mention_wiki_map = load_OPIEC_wiki_mention_map(OPIEC_mention_map_path)
    all_concept_maps = load_Probase(Probase_path)
    concept_maps = defaultdict(set)
    with open(aligned_Probase_path, 'w') as fwrite:
        for c, ps in all_concept_maps.items():
            if c not in mention_wiki_map:
                continue
            concept_maps[c] = ps
            fwrite.write('%s\t%s\n' % (c, '\t'.join(ps)))
    analysis_concept_pairs(concept_maps)


def analysis_openie_triples(OPIEC_triple_path: str):
    print('start analysis openie triples in %s' % (OPIEC_triple_path))
    mention_set = set()
    relation_set = set()
    line_cnt = 0
    with open(OPIEC_triple_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip('\n').split('\t')
            # subj_m, rel_m, obj_m, subj, rel, obj = line.strip('\n').split('\t')
            if len(line_list) < 3:
                continue
            subj_m = line_list[0]
            rel_m = line_list[1]
            obj_m = line_list[2]
            mention_set.add(subj_m)
            mention_set.add(obj_m)
            relation_set.add(rel_m)
            line_cnt += 1
    print('#mention, #rel, #triple: ')
    print('| %s | %s | %s |' % (len(mention_set), len(relation_set), line_cnt))


def store_filtered_ConceptPairs_ReVerb(CM_path: str, CM_type: str, ReVerb_path: str, out_path: str):
    mention_set = set()
    with open(ReVerb_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            arg1_norm = line_list[4]   # norm is already lowercased
            arg2_norm = line_list[6]
            mention_set.add(arg1_norm)
            mention_set.add(arg2_norm)
    print('OpenIE Triple=%s loaded' % (ReVerb_path))
    if CM_type == "Probase":
        concept_pairs = load_Probase(CM_path)
    elif CM_type == "SemEval":
        concept_pairs = load_merged_SemEval(CM_path)
    else:
        print('ERROR invalid CM_type=%s' % (CM_type))
        exit(0)
    print('Concept Graph=%s Loaded.' % (CM_path))
    to_remove = set()
    for c in concept_pairs:
        if c not in mention_set:
            to_remove.add(c)
    concept_pairs = {k: v for k, v in concept_pairs.items() if k not in to_remove}
    analysis_concept_pairs(concept_pairs)
    with open(out_path, 'w') as fwrite:
        for c, ps in concept_pairs.items():
            fwrite.write('%s\t%s\n' % (c, '\t'.join(ps)))


def split_train_dev_test(aligned_concept_path: str, aligned_openie_path: str,
                         out_dir: str, setting: str = 'rich'):
    def _write_concept_pairs_to_file(concept_pairs: Dict[str, set], out_path: str):
        with open(out_path, 'w') as fwrite:
            for c, ps in concept_pairs.items():
                fwrite.write('%s\t%s\n' % (c, '\t'.join(ps)))
        return
    random.seed(1105)
    if setting == 'rich':
        concept_entity_threshold = 40  # freq >= 40
    elif setting == 'limited':
        concept_entity_threshold = 3   # freq >= 1
        OLP_rel_freq_threshold = 2
        OLP_rel_char_threshold = 2
        OLP_ment_freq_threshold = 2
        OLP_ment_char_threshold = 3
    else:
        print('invalid arg setting=%s' % (setting))
        exit(-1)
    # filter out invalid concept pairs
    all_concept_pairs = load_merged_SemEval(aligned_concept_path)
    reverse_concept_pairs = defaultdict(set)  # p: {c1, c2}
    for c, ps in all_concept_pairs.items():
        for p in ps:
            reverse_concept_pairs[p].add(c)
    kept_concepts = set()
    for p, cs in reverse_concept_pairs.items():
        if len(cs) >= concept_entity_threshold:
            kept_concepts.add(p)
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    for c, ps in all_concept_pairs.items():
        kept_ps = set()
        for p in ps:
            if p in kept_concepts:
                kept_ps.add(p)
        if len(kept_ps) <= 0:
            continue
        concept_pairs[c] = kept_ps
    print('After filtering, #ent=%d, #cep=%d' % (len(concept_pairs), len(kept_concepts)))
    train_ent_cnt = int(len(concept_pairs) * 0.55)
    test_ent_cnt = int(len(concept_pairs) * 0.35)
    print('pre-computed train_ent_cnt=%d, test_ent_cnt=%d' % (train_ent_cnt, test_ent_cnt))
    # dev_ent_cnt = len(concept_pairs) - train_ent_cnt - test_ent_cnt
    train_concept_pairs = defaultdict(set)
    test_concept_pairs = defaultdict(set)
    dev_concept_pairs = defaultdict(set)
    all_entities = list(concept_pairs.keys())
    for ent in all_entities:
        if len(train_concept_pairs) < train_ent_cnt:
            train_concept_pairs[ent] = concept_pairs[ent]
        elif len(test_concept_pairs) < test_ent_cnt:
            test_concept_pairs[ent] = concept_pairs[ent]
        else:
            dev_concept_pairs[ent] = concept_pairs[ent]
    print('CG - train set')
    analysis_concept_pairs(train_concept_pairs, False)
    print('CG - dev set')
    analysis_concept_pairs(dev_concept_pairs, False)
    print('CG - test set')
    analysis_concept_pairs(test_concept_pairs, False)
    # write to files
    cg_train_path = '%s/cg_pairs.train.txt' % (out_dir)
    cg_dev_path = '%s/cg_pairs.dev.txt' % (out_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (out_dir)
    _write_concept_pairs_to_file(train_concept_pairs, cg_train_path)
    _write_concept_pairs_to_file(dev_concept_pairs, cg_dev_path)
    _write_concept_pairs_to_file(test_concept_pairs, cg_test_path)
    # filter out triples that not align with kept entities
    # split openie triples according to ratio
    kept_lines = []
    # first round to collect info
    ment_freq_dict = defaultdict(int)
    rel_freq_dict = defaultdict(int)
    with open(aligned_openie_path) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            if len(line_list) < 3:
                continue
            subj, rel, obj = line_list[0], line_list[1], line_list[2]
            if len(subj) < OLP_ment_char_threshold or len(rel) < OLP_rel_char_threshold \
                    or len(obj) < OLP_ment_char_threshold:
                continue
            if subj not in concept_pairs and obj not in concept_pairs:
                continue
            ment_freq_dict[subj] += 1
            ment_freq_dict[obj] += 1
            rel_freq_dict[rel] += 1
    ment_freq_dict = {k: v for k, v in ment_freq_dict.items() if v >= OLP_ment_freq_threshold}
    rel_freq_dict = {k: v for k, v in rel_freq_dict.items() if v >= OLP_rel_freq_threshold}
    # second round to write
    with open(aligned_openie_path) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            if len(line_list) < 3:
                continue
            subj, rel, obj = line_list[0], line_list[1], line_list[2]
            if len(subj) < OLP_ment_char_threshold or len(rel) < OLP_rel_char_threshold \
                    or len(obj) < OLP_ment_char_threshold:
                continue
            if subj not in concept_pairs and obj not in concept_pairs:
                continue
            if subj not in ment_freq_dict or obj not in ment_freq_dict:
                continue
            if rel not in rel_freq_dict:
                continue
            kept_lines.append('%s\t%s\t%s\n' % (subj, rel, obj))
    total_triple_cnt = len(kept_lines)
    train_cnt = int(total_triple_cnt * 0.80)
    test_cnt = int(total_triple_cnt * 0.15)
    dev_cnt = total_triple_cnt - train_cnt - test_cnt
    indices = list(range(total_triple_cnt))
    random.shuffle(indices)
    test_indices = set(indices[train_cnt:train_cnt+test_cnt])
    dev_indices = set(indices[-dev_cnt:])
    oie_train_path = '%s/oie_triples.train.txt' % (out_dir)
    oie_dev_path = '%s/oie_triples.dev.txt' % (out_dir)
    oie_test_path = '%s/oie_triples.test.txt' % (out_dir)
    fwrite_train = open(oie_train_path, 'w')
    fwrite_dev = open(oie_dev_path, 'w')
    fwrite_test = open(oie_test_path, 'w')
    for idx, line in enumerate(kept_lines):
        if idx in dev_indices:
            fwrite_dev.write(line)
        elif idx in test_indices:
            fwrite_test.write(line)
        else:
            fwrite_train.write(line)
    fwrite_train.close()
    fwrite_dev.close()
    fwrite_test.close()
    print('OIE - train set')
    analysis_openie_triples(oie_train_path)
    print('OIE - dev set')
    analysis_openie_triples(oie_dev_path)
    print('OIE - test set')
    analysis_openie_triples(oie_test_path)


if __name__ == '__main__':
    # Notes: ConceptGraphs are stored as lower case

    # OPIEC_path = 'data/OPIEC/OPIEC-Linked-example.avro'
    # OPIEC_aligned_path = 'data/OPIEC/OPIEC-Linked-example.Probase-aligned.txt'
    OPIEC_path = 'data/OPIEC/OPIEC-Linked-triples'
    OPIEC_Probase_aligned_path = 'data/OPIEC/OPIEC-Linked-triples.Probase-aligned.txt'
    # align_Probase_OPIEC(Probase_path, OPIEC_path, OPIEC_Probase_aligned_path)
    OPIEC_wiki_mention_path = 'data/OPIEC/OPIEC-Linked-triples.Wiki-mentions.txt'
    # produce_OPIEC_mention_wiki_map(OPIEC_path, OPIEC_wiki_mention_path)

    SemEval_medical_train_data = 'data/SemEval2018-Task9/training/data/2A.medical.training.data.txt'
    SemEval_medical_train_gold = 'data/SemEval2018-Task9/training/gold/2A.medical.training.gold.txt'
    SemEval_medical_test_data = 'data/SemEval2018-Task9/test/data/2A.medical.test.data.txt'
    SemEval_medical_test_gold = 'data/SemEval2018-Task9/test/gold/2A.medical.test.gold.txt'
    # load_SemEval(SemEval_medical_test_data, SemEval_medical_test_gold, True)
    # analysis_SemEval(SemEval_medical_train_gold, SemEval_medical_test_gold)
    SemEval_music_train_data = 'data/SemEval2018-Task9/training/data/2B.music.training.data.txt'
    SemEval_music_train_gold = 'data/SemEval2018-Task9/training/gold/2B.music.training.gold.txt'
    SemEval_music_test_data = 'data/SemEval2018-Task9/test/data/2B.music.test.data.txt'
    SemEval_music_test_gold = 'data/SemEval2018-Task9/test/gold/2B.music.test.gold.txt'
    # load_SemEval(SemEval_music_test_data, SemEval_music_test_gold, True)
    # analysis_SemEval(SemEval_music_train_gold, SemEval_music_test_gold)

    # Produce dataset for own task
    # SemEval
    # Step 1: Merge data gold to get concept pairs
    # merge_SemEval_train_dev_test_sets('data/SemEval2018-Task9', '2B.music')
    # merge_SemEval_train_dev_test_sets('data/SemEval2018-Task9', '2A.medical')
    # merge_SemEval_train_dev_test_sets('data/SemEval2018-Task9', '1A.english')
    # Step 2: align entity with openIE triples, keep aligned ones and analysis
    OPIEC_wiki_mention_path = 'data/OPIEC/OPIEC-Linked-triples.Wiki-mentions.txt'
    SemEval_music_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.txt'
    aligned_SemEval_music_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.OPIEC-aligned.txt'
    # align_SemEval_OPIEC_mentions(SemEval_music_path, OPIEC_wiki_mention_path, aligned_SemEval_music_path)
    SemEval_medical_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.txt'
    aligned_SemEval_medical_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.OPIEC-aligned.txt'
    # align_SemEval_OPIEC_mentions(SemEval_medical_path, OPIEC_wiki_mention_path, aligned_SemEval_medical_path)
    # Step3: filter out openIE triples
    OPIEC_path = 'data/OPIEC/OPIEC-Linked-triples'
    OPIEC_music_aligned_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMusic-aligned.txt'
    # align_Probase_OPIEC(SemEval_music_path, OPIEC_path, OPIEC_music_aligned_path, CM_type="SemEval")
    OPIEC_medical_aligned_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMedical-aligned.txt'
    # align_Probase_OPIEC(SemEval_medical_path, OPIEC_path, OPIEC_medical_aligned_path, CM_type="SemEval")

    # Probase / MSConceptGraph
    OPIEC_wiki_mention_path = 'data/OPIEC/OPIEC-Linked-triples.Wiki-mentions.txt'
    Probase_path = 'data/MSConceptGraph/data-concept-instance-relations.txt'
    aligned_Probase_path = 'data/MSConceptGraph/instance-concepts.OPIEC-aligned.txt'
    # align_Probase_OPIEC_mentions(Probase_path, OPIEC_wiki_mention_path, aligned_Probase_path)

    # Aligned concept graph STAT
    # load_merged_SemEval(aligned_Probase_path, do_analysis=True)
    # load_merged_SemEval(aligned_SemEval_medical_path, do_analysis=True)
    # load_merged_SemEval(aligned_SemEval_music_path, do_analysis=True)
    # OpenIE triple STAT
    # analysis_openie_triples(OPIEC_Probase_aligned_path)
    # analysis_openie_triples(OPIEC_medical_aligned_path)
    # analysis_openie_triples(OPIEC_music_aligned_path)

    # ReVerb as Open Knowledge
    Probase_path = 'data/MSConceptGraph/data-concept-instance-relations.txt'
    # ReVerb_path = 'data/ReVerb/reverb_wikipedia_tuples-1.1.txt'
    ReVerb_path = 'data/ReVerb/reverb_clueweb_tuples-1.1.txt'
    ReVerb_out_path = 'data/ReVerb/reverb_clueweb_tuples.Probase-aligned.txt'
    # align_Probase_ReVerb(Probase_path, ReVerb_path, Reverb_out_path)
    # analysis_openie_triples(ReVerb_out_path)
    Probase_ReVerb_aligned_path = 'data/MSConceptGraph/data-concept-instance-relations.ReVerb-aligned.txt'
    # store_filtered_ConceptPairs_ReVerb(Probase_path, 'Probase', ReVerb_path, Probase_ReVerb_aligned_path)
    SemEval_medical_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.txt'
    ReVerb_medical_out_path = 'data/ReVerb/reverb_clueweb_tuples.SemEvalMedical-aligned.txt'
    # align_Probase_ReVerb(SemEval_medical_path, ReVerb_path, ReVerb_medical_out_path, CM_type="SemEval")
    # analysis_openie_triples(ReVerb_medical_out_path)
    SemEval_medical_ReVerb_aligned_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.ReVerb-aligned.txt'
    # store_filtered_ConceptPairs_ReVerb(SemEval_medical_path, 'SemEval', ReVerb_path, SemEval_medical_ReVerb_aligned_path)
    SemEval_music_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.txt'
    ReVerb_music_out_path = 'data/ReVerb/reverb_clueweb_tuples.SemEvalMusic-aligned.txt'
    # align_Probase_ReVerb(SemEval_music_path, ReVerb_path, ReVerb_music_out_path, CM_type='SemEval')
    # analysis_openie_triples(ReVerb_music_out_path)
    SemEval_music_ReVerb_aligned_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.ReVerb-aligned.txt'
    # store_filtered_ConceptPairs_ReVerb(SemEval_music_path, 'SemEval', ReVerb_path, SemEval_music_ReVerb_aligned_path)
    # Aligned concept graph STAT for ReVerb x ConceptGraphs
    # load_merged_SemEval(Probase_ReVerb_aligned_path, do_analysis=True)
    # load_merged_SemEval(SemEval_medical_ReVerb_aligned_path, do_analysis=True)
    # load_merged_SemEval(SemEval_music_ReVerb_aligned_path, do_analysis=True)

    # Split train dev test
    aligned_concept_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.ReVerb-aligned.txt'
    aligned_openie_path = 'data/ReVerb/reverb_clueweb_tuples.SemEvalMedical-aligned.txt'
    out_dir = 'data/CGC-OLP-BENCH/SEMedical-ReVerb'
    split_train_dev_test(aligned_concept_path, aligned_openie_path, out_dir, 'limited')
    aligned_concept_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.ReVerb-aligned.txt'
    aligned_openie_path = 'data/ReVerb/reverb_clueweb_tuples.SemEvalMusic-aligned.txt'
    out_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    split_train_dev_test(aligned_concept_path, aligned_openie_path, out_dir, 'limited')
    aligned_concept_path = 'data/MSConceptGraph/data-concept-instance-relations.ReVerb-aligned.txt'
    aligned_openie_path = 'data/ReVerb/reverb_clueweb_tuples.Probase-aligned.txt'
    out_dir = 'data/CGC-OLP-BENCH/MSCG-ReVerb'
    # split_train_dev_test(aligned_concept_path, aligned_openie_path, out_dir, 'rich')
    aligned_concept_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.OPIEC-aligned.txt'
    aligned_openie_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMedical-aligned.txt'
    out_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    # split_train_dev_test(aligned_concept_path, aligned_openie_path, out_dir, 'limited')
    aligned_concept_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.OPIEC-aligned.txt'
    aligned_openie_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMusic-aligned.txt'
    out_dir = 'data/CGC-OLP-BENCH/SEMusic-OPIEC'
    # split_train_dev_test(aligned_concept_path, aligned_openie_path, out_dir, 'limited')
    aligned_concept_path = 'data/MSConceptGraph/instance-concepts.OPIEC-aligned.txt'
    aligned_openie_path = 'data/OPIEC/OPIEC-Linked-triples.Probase-aligned.txt'
    out_dir = 'data/CGC-OLP-BENCH/MSCG-OPIEC'
    # split_train_dev_test(aligned_concept_path, aligned_openie_path, out_dir, 'rich')

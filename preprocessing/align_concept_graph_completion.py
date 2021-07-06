"""
Align entity in Concept Graph and OpenIE facts
Author: Jiaying Lu
Create Date: Jun 29, 2021
"""

import os
import time
from collections import defaultdict
from typing import Tuple, Dict

import tqdm
from avro.datafile import DataFileReader
from avro.io import DatumReader


def load_Probase(Probase_path: str) -> dict:
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
    return concept_pairs


def analysis_concept_pairs(concept_pairs: Dict[str, set]):
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
    print('#hyponym=%d, #hypernym=%d, #triple=%d' % (len(concept_pairs), len(reverse_concept_pairs),
                                                     triple_cnt))
    print('Avg #Parent=%.2f, Avg #Child=%.2f' % (avg_parent, avg_child))
    # cal avg level
    # first find leaf concepts
    # then find concepts with all children exist
    print('#concept=%d' % (len(all_concepts)))
    concept_level = dict()
    to_remove = set()
    for c in all_concepts:
        if c not in reverse_concept_pairs:
            concept_level[c] = 1.0
            to_remove.add(c)
    all_concepts.difference_update(to_remove)
    print('#leaf concept=%d' % (len(concept_level)))
    iteration_cnt = 0
    while len(all_concepts) > 0:
        remains = set()
        for ent in all_concepts:
            # if all children are in concept_level dict,
            # then remove ent from all_concepts
            if all(child in concept_level for child in reverse_concept_pairs[ent]):
                level = max(concept_level[child] for child in reverse_concept_pairs[ent]) + 1.0
                concept_level[ent] = level
            else:
                remains.add(ent)
        all_concepts = remains
        iteration_cnt += 1
        if iteration_cnt > 10:
            print('deadlock!! remains all_concepts=%d' % (len(all_concepts)))
            '''
            for ent in all_concepts:
                print('%s: %s' % (ent, reverse_concept_pairs[ent]))
            '''
            break
    print('Avg level=%.2f' % (sum(concept_level.values()) / len(concept_level)))


def load_SemEval(data_path: str, gold_path: str, do_analysis: bool = False) -> dict:
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    with open(data_path) as fopen1, open(gold_path) as fopen2:
        for data_line in tqdm.tqdm(fopen1):
            gold_line = fopen2.readline()
            hyponym = data_line.strip().split('\t')[0]
            hypernyms = gold_line.strip().split('\t')
            concept_pairs[hyponym].update(hypernyms)
    if do_analysis:
        analysis_concept_pairs(concept_pairs)
    return concept_pairs


def load_merged_SemEval(file_path: str) -> dict:
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    with open(file_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            c = line_list[0]
            ps = line_list[1:]
            concept_pairs[c] = ps
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


def align_Probase_ReVerb(Probase_path: str, ReVerb_path: str):
    concept_pairs = load_Probase(Probase_path)
    grounded_concepts = defaultdict(int)
    grounded_concepts_entities = defaultdict(set)
    with open(ReVerb_path) as fopen:
        tik = time.perf_counter()
        lines = fopen.readlines()
        elapsed_time = time.perf_counter() - tik
        print('Readlines ReVerb, elapsed_time=%s' % (elapsed_time))
        for line in tqdm.tqdm(lines):
            line_list = line.strip().split('\t')
            arg1_norm = line_list[4]
            # rel_norm = line_list[5]
            arg2_norm = line_list[6]
            if arg1_norm in concept_pairs:
                for p in concept_pairs[arg1_norm]:
                    grounded_concepts[p] += 1
                    grounded_concepts_entities[p].add(arg1_norm)
            if arg2_norm in concept_pairs:
                for p in concept_pairs[arg2_norm]:
                    grounded_concepts[p] += 1
                    grounded_concepts_entities[p].add(arg2_norm)
        del lines
    print('Grounded ReVerb concepts=%d' % (len(grounded_concepts)))
    # cnt >= 50, <50
    manyshot_concepts = len([(k, v) for k, v in grounded_concepts.items() if v >= 50])
    print('>=50 triples concepts=%d' % (manyshot_concepts))
    fewshot_concepts = len([(k, v) for k, v in grounded_concepts.items() if v < 50])
    print('<50 triples concepts=%d' % (fewshot_concepts))
    # entity_cnt >=50, <50
    manyshot_concepts = len([(k, v) for k, v in grounded_concepts_entities.items() if len(v) >= 50])
    print('>=50 entities concepts=%d' % (manyshot_concepts))
    fewshot_concepts = len([(k, v) for k, v in grounded_concepts_entities.items() if len(v) < 50])
    print('<50 entities concepts=%d' % (fewshot_concepts))


def _get_lemma_wikilink(tok_list: list) -> Tuple[str, str]:
    lemma = []
    wikilink = tok_list[0]['w_link']['wiki_link']
    for _ in tok_list:
        lemma.append(_['lemma'] if _['lemma'] is not None else _['word'])
    return ' '.join(lemma), wikilink


def align_Probase_OPIEC(Probase_path: str, OPIEC_path: str, OPIEC_aligned_path: str, CM_type: str = "Probase"):
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
            fwrite.write('%s\t%s\n' % (c, '\t'.join(ps)))


def load_OPIEC_wiki_mention_map(OPIEC_mention_map_path: str) -> Tuple[dict, dict]:
    wiki_mention_map = defaultdict(set)
    mention_wiki_map = defaultdict(set)
    with open(OPIEC_mention_map_path) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            wiki = line_list[0]
            mentions = line_list[1:]
            wiki_mention_map[wiki] = set(mentions)
            for m in mentions:
                mention_wiki_map[m].add(wiki)
    return wiki_mention_map, mention_wiki_map


def align_SemEval_OPIEC_mentions(SemEval_path: str, OPIEC_mention_map_path: str, aligned_SemEval_path: str):
    # find SemEval hyponyms that exsit in OPIEC mentions
    # Store filtered SemEval concept pairs
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


if __name__ == '__main__':
    Probase_path = 'data/MSConceptGraph/data-concept-instance-relations.txt'
    ReVerb_path = 'data/ReVerb/reverb_clueweb_tuples-1.1.txt'
    # ReVerb_out_path = 'data/ReVerb/tuples_grounded_MSConceptGraph.txt'
    # ReVerb_path = 'data/ReVerb/reverb_wikipedia_tuples-1.1.txt'
    # align_Probase_ReVerb(Probase_path, ReVerb_path)

    # OPIEC_path = 'data/OPIEC/OPIEC-Linked-example.avro'
    # OPIEC_aligned_path = 'data/OPIEC/OPIEC-Linked-example.Probase-aligned.txt'
    # OPIEC_path = 'data/OPIEC/OPIEC-Linked-triples'
    # OPIEC_aligned_path = 'data/OPIEC/OPIEC-Linked-triples.Probase-aligned.txt'
    # align_Probase_OPIEC(Probase_path, OPIEC_path, OPIEC_aligned_path)
    OPIEC_wiki_mention_path = 'data/OPIEC/OPIEC-Linked-triples.Wiki-mentions.txt'
    # produce_OPIEC_mention_wiki_map(OPIEC_path, OPIEC_wiki_mention_path)

    SemEval_medical_train_gold = 'data/SemEval2018-Task9/training/gold/2A.medical.training.gold.txt'
    SemEval_medical_test_data = 'data/SemEval2018-Task9/test/data/2A.medical.test.data.txt'
    SemEval_medical_test_gold = 'data/SemEval2018-Task9/test/gold/2A.medical.test.gold.txt'
    # load_SemEval(SemEval_medical_data, SemEval_medical_gold, True)
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
    # Step 2: align entity with openIE triples, keep aligned ones and analysis
    OPIEC_wiki_mention_path = 'data/OPIEC/OPIEC-Linked-triples.Wiki-mentions.txt'
    SemEval_music_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.txt'
    aligned_SemEval_music_path = 'data/SemEval2018-Task9/2B.music.merged_pairs.OPIEC-aligned.txt'
    # align_SemEval_OPIEC_mentions(SemEval_music_path, OPIEC_wiki_mention_path, aligned_SemEval_music_path)
    SemEval_medical_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.txt'
    aligned_SemEval_medical_path = 'data/SemEval2018-Task9/2A.medical.merged_pairs.OPIEC-aligned.txt'
    # align_SemEval_OPIEC_mentions(SemEval_medical_path, OPIEC_wiki_mention_path, aligned_SemEval_medical_path)

    OPIEC_path = 'data/OPIEC/OPIEC-Linked-triples'
    OPIEC_aligned_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMusic-aligned.txt'
    # align_Probase_OPIEC(SemEval_music_path, OPIEC_path, OPIEC_aligned_path, CM_type="SemEval")
    OPIEC_aligned_path = 'data/OPIEC/OPIEC-Linked-triples.SemEvalMedical-aligned.txt'
    # align_Probase_OPIEC(SemEval_medical_path, OPIEC_path, OPIEC_aligned_path, CM_type="SemEval")

    # Probase / MSConceptGraph
    OPIEC_wiki_mention_path = 'data/OPIEC/OPIEC-Linked-triples.Wiki-mentions.txt'
    Probase_path = 'data/MSConceptGraph/data-concept-instance-relations.txt'
    aligned_Probase_path = 'data/MSConceptGraph/instance-concepts.OPIEC-aligned.txt'
    align_Probase_OPIEC_mentions(Probase_path, OPIEC_wiki_mention_path, aligned_Probase_path)

"""
Extract triples from fewrel
Author: Jiaying Lu
Create Date: Apr 26, 2021
"""

import json
import re
import bz2
import tqdm
import numpy as np


def extract_entities(in_path: str, out_path: str):
    result = {}
    total_cnt = 0
    with open(in_path) as fopen:
        data = json.load(fopen)
    for rel, fact_list in tqdm.tqdm(data.items()):
        for fact_d in fact_list:
            for triple_type in ['h', 't']:
                ent_name = fact_d[triple_type][0]
                ent_id = fact_d[triple_type][1]
                if ent_id not in result:
                    result[ent_id] = {'name': ent_name, 'freq': 0}
                result[ent_id]['freq'] += 1
                total_cnt += 2
    print('total entity cnt=%d, unique entity cnt=%d' % (total_cnt, len(result)))
    with open(out_path, 'w') as fwrite:
        json.dump(result, fwrite)


def extract_entities_test_set(in_path: str, out_path: str):
    result = {}
    total_cnt = 0
    with open(in_path) as fopen:
        data = json.load(fopen)
    for episode in tqdm.tqdm(data):
        fact_d = episode['meta_test']
        for triple_type in ['h', 't']:
            ent_name = fact_d[triple_type][0]
            ent_id = fact_d[triple_type][1]
            if ent_id not in result:
                result[ent_id] = {'name': ent_name, 'freq': 0}
            result[ent_id]['freq'] += 1
            total_cnt += 2
        for fact_ds in episode['meta_train']:
            for fact_d in fact_ds:
                for triple_type in ['h', 't']:
                    ent_name = fact_d[triple_type][0]
                    ent_id = fact_d[triple_type][1]
                    if ent_id not in result:
                        result[ent_id] = {'name': ent_name, 'freq': 0}
                    result[ent_id]['freq'] += 1
                    total_cnt += 2
    print('total entity cnt=%d, unique entity cnt=%d' % (total_cnt, len(result)))
    with open(out_path, 'w') as fwrite:
        json.dump(result, fwrite)


def merge_all_entities(in_path_list: list, out_path: str):
    result = {}
    for path in in_path_list:
        with open(path) as fopen:
            data = json.load(fopen)
        for ent_id, ent_info in data.items():
            if ent_id not in result:
                result[ent_id] = ent_info
            else:
                result[ent_id]['freq'] += ent_info['freq']
    with open(out_path, 'w') as fwrite:
        json.dump(result, fwrite)


def do_analysis(in_path: str):
    with open(in_path) as fopen:
        data = json.load(fopen)
    # freq_result = {}
    freqs = []
    for ent_id, ent_dict in data.items():
        freq = ent_dict['freq']
        freqs.append(freq)
    print(np.histogram(freqs, [1, 2, 3, 4, 5, 10000]))


def do_overlap_analysis(train_set: str, val_set: str, test_set: str):
    with open(train_set) as fopen:
        train_ents = json.load(fopen)
    train_ents = set(train_ents.keys())
    with open(val_set) as fopen:
        val_ents = json.load(fopen)
    val_ents = set(val_ents.keys())
    with open(test_set) as fopen:
        test_ents = json.load(fopen)
    test_ents = set(test_ents.keys())
    print('train x val: %d' % (len(train_ents.intersection(val_ents))))
    print('train x test: %d' % (len(train_ents.intersection(test_ents))))
    print('val x test: %d' % (len(val_ents.intersection(test_ents))))
    print('train x val x test: %d' % (len(train_ents.intersection(val_ents).intersection(test_ents))))


def extract_relation_info_from_wikidata(relations_path: str, wikidata_path: str, out_path: str,
                                        entry_type: str = 'property'):
    with open(relations_path, 'r') as fopen:
        relations = json.load(fopen)
    pids = set(relations.keys())
    hit_cnt = 0
    line_starts = '{"type":"%s"' % (entry_type)
    pat = re.compile(r'\{"type":"%s","id":"(\w+)",' % (entry_type))
    with bz2.open(wikidata_path, 'rt') as fopen, open(out_path, 'w') as fwrite:
        line = fopen.readline()
        for line in tqdm.tqdm(fopen):
            line = line.strip(',\n')
            if not line.startswith(line_starts):
                continue
            re_res = pat.match(line)
            if not re_res:
                continue
            ent_id = re_res.group(1)
            # info = json.loads(line)
            # ent_id = info['id']
            if ent_id not in pids:
                continue
            hit_cnt += 1
            fwrite.write(line + '\n')
            if hit_cnt % int(len(pids)/10) == 0:
                print('hit cnt=%d outof %d' % (hit_cnt, len(pids)))
            if hit_cnt >= len(pids):
                break


if __name__ == '__main__':
    # in_path = 'data/FewRel/val_wiki.json'
    # out_path = 'data/FewRel/entities/val_wiki.json'
    # in_path = 'data/FewRel/train_wiki.json'
    # out_path = 'data/FewRel/entities/train_wiki.json'
    # extract_entities(in_path, out_path)
    """
    for name in ['-5-1', '-5-5', '-10-1', '-10-5']:
        in_path = 'data/FewRel/test_wiki_input%s.json' % (name)
        out_path = 'data/FewRel/entities/test_wiki_input%s.json' % (name)
        extract_entities_test_set(in_path, out_path)
    """
    entity_files = ['data/FewRel/entities/train_wiki.json', 'data/FewRel/entities/val_wiki.json',
                    'data/FewRel/entities/test_wiki_input-5-1.json', 'data/FewRel/entities/test_wiki_input-5-5.json',
                    'data/FewRel/entities/test_wiki_input-10-1.json', 'data/FewRel/entities/test_wiki_input-10-5.json']
    out_path = 'data/FewRel/entities/wiki_all.json'
    # merge_all_entities(entity_files, out_path)

    # do_analysis('data/FewRel/entities/test_wiki_input-5-1.json')
    # do_overlap_analysis('data/FewRel/entities/train_wiki.json', 'data/FewRel/entities/val_wiki.json',
    #                     'data/FewRel/entities/test_wiki_input-10-1.json')

    relations_path = 'data/FewRel/pid2name.json'
    wikidata_path = 'data/wikidata-entities-all.Apr28.json.bz2'
    out_path = 'data/FewRel/Wikidata/relations-all-info.jsonlines'
    # extract_relation_info_from_wikidata(relations_path, wikidata_path, out_path)
    entities_path = 'data/FewRel/entities/wiki_all.json'
    out_path = 'data/FewRel/Wikidata/entities-all-info.jsonlines'
    # extract_relation_info_from_wikidata(entities_path, wikidata_path, out_path,
    #                                     entry_type='item')

"""
Extract triples from fewrel
Author: Anonymous Siamese
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
    keyerror_cnt = 0
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
            # ent_id = info['id']
            if ent_id not in pids:
                continue
            info = json.loads(line)
            # filter out irrelavant info
            res = {}
            res['id'] = info['id']
            # res['label'] = info['labels']['en']['value']
            res['claims'] = {}
            if 'P31' in info['claims']:   # P31: instance_of
                # res['claims']['P31'] = [_['mainsnak']['datavalue']['value']['id'] for _ in info['claims']['P31']]
                res['claims']['P31'] = []
                for _ in info['claims']['P31']:
                    try:
                        tail_ent_id = _['mainsnak']['datavalue']['value']['id']
                        res['claims']['P31'].append(tail_ent_id)
                    except KeyError:
                        keyerror_cnt += 1
                        print('KeyError: %s P31 item is: %s' % (ent_id, _))
            if 'P279' in info['claims']:  # P279 subclass_of
                # res['claims']['P279'] = [_['mainsnak']['datavalue']['value']['id'] for _ in info['claims']['P279']]
                res['claims']['P279'] = []
                for _ in info['claims']['P279']:
                    try:
                        tail_ent_id = _['mainsnak']['datavalue']['value']['id']
                        res['claims']['P279'].append(tail_ent_id)
                    except KeyError:
                        keyerror_cnt += 1
                        print('KeyError: %s P279 item is: %s' % (ent_id, _))
            hit_cnt += 1
            fwrite.write(json.dumps(res) + '\n')
            if hit_cnt % int(len(pids)/10) == 0:
                print('hit cnt=%d outof %d' % (hit_cnt, len(pids)))
            if hit_cnt >= len(pids):
                break


def get_itemid_from_URI(URI: str) -> str:
    """
    get_itemid_from_URI("http://www.wikidata.org/entity/Q209041") -> Q209041
    """
    return URI.split('/')[-1]


def generate_entity_taxonomy(entity_1hop_file: str, root_4hop_file: str, out_file: str):
    # analysis connectivity
    connect_to_root = set()
    with open(root_4hop_file) as fopen:
        _4hops = json.load(fopen)
    for _ in _4hops:
        for key in ['hop4', 'hop3', 'hop2', 'hop1']:
            itemid = get_itemid_from_URI(_[key])
            connect_to_root.add(itemid)
    total_cnt = 0
    hit_cnt = 0
    with open(entity_1hop_file) as fopen:
        for line in fopen:  # jsonlines
            res = json.loads(line.strip())
            total_cnt += 1
            hit_flag = False
            if 'P31' in res['claims']:
                for itemid in res['claims']['P31']:
                    if itemid in connect_to_root:
                        hit_flag = True
                        break
            if not hit_flag and 'P279' in res['claims']:
                for itemid in res['claims']['P279']:
                    if itemid in connect_to_root:
                        hit_flag = True
                        break
            if hit_flag:
                hit_cnt += 1
    print('total %d entities in FewRel, %d (%.2f) can connect to root' % (total_cnt, hit_cnt,
            hit_cnt/total_cnt))


def extract_all_entitiy_with_edges(wikidata_path: str, out_path: str):
    # edge_labels = ['P31', 'P279']    # instance_of, subclass_of
    item_type = 'item'
    line_starts = '{"type":"%s"' % (item_type)
    with bz2.open(wikidata_path, 'rt') as fopen, open(out_path, 'w') as fwrite:
        line = fopen.readline()
        for line in tqdm.tqdm(fopen):
            line = line.strip(',\n')
            if not line.startswith(line_starts):
                continue
            info = json.loads(line)
            # filter out irrelavant info
            res = {}
            res['id'] = info['id']
            try:
                res['label'] = info['labels']['en']['value']
            except KeyError:
                res['label'] = ''
            res['claims'] = {}
            if 'P31' in info['claims']:   # P31: instance_of
                # res['claims']['P31'] = [_['mainsnak']['datavalue']['value']['id'] for _ in info['claims']['P31']]
                res['claims']['P31'] = []
                for _ in info['claims']['P31']:
                    try:
                        tail_ent_id = _['mainsnak']['datavalue']['value']['id']
                        res['claims']['P31'].append(tail_ent_id)
                    except KeyError:
                        print('KeyError: %s P31 item is: %s' % (res['id'], _))
            if 'P279' in info['claims']:  # P279 subclass_of
                # res['claims']['P279'] = [_['mainsnak']['datavalue']['value']['id'] for _ in info['claims']['P279']]
                res['claims']['P279'] = []
                for _ in info['claims']['P279']:
                    try:
                        tail_ent_id = _['mainsnak']['datavalue']['value']['id']
                        res['claims']['P279'].append(tail_ent_id)
                    except KeyError:
                        print('KeyError: %s P279 item is: %s' % (res['id'], _))
            fwrite.write(json.dumps(res) + '\n')


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
    out_path = 'data/FewRel/Wikidata/entities-every-P31-P279.jsonlines'
    # extract_all_entitiy_with_edges(wikidata_path, out_path)

    entity_1hop_file = 'data/FewRel/Wikidata/entities-all-info.jsonlines'
    root_4hop_file = 'data/FewRel/Wikidata/4hops_subclass_of_entityQ35120.json'
    # generate_entity_taxonomy(entity_1hop_file, root_4hop_file, '')

"""
Analysis entity percentage of IsA relation
Author: Jiaying Lu
Create Date: Jul 9, 2021
"""
import re
from collections import defaultdict

import gzip
import tqdm


def analysis_Yago(type_path: str, fact_path: str, file_type: str = 'normal'):
    entity_set = set()
    if file_type == 'normal':
        fopen = open(fact_path)
    elif file_type == 'gz':
        fopen = gzip.open(fact_path, 'rt')
    else:
        print('invalid file_type')
        exit(-1)
    for line in tqdm.tqdm(fopen):
        line_list = line.strip().split('\t')
        head = line_list[0]
        tail = line_list[2]
        entity_set.add(head)
        entity_set.add(tail)
    fopen.close()
    entity_isA_set = set()
    if file_type == 'normal':
        fopen = open(type_path)
    elif file_type == 'gz':
        fopen = gzip.open(type_path, 'rt')
    else:
        print('invalid file_type')
        exit(-1)
    for line in tqdm.tqdm(fopen):
        line_list = line.strip().split('\t')
        head = line_list[0]
        tail = line_list[2]
        entity_isA_set.add(head)
        entity_isA_set.add(tail)
    fopen.close()
    isA_cnt = len(entity_isA_set)
    all_cnt = len(entity_set)
    print('#IsA=%d over %d, %.2f%%' % (isA_cnt, all_cnt, isA_cnt / all_cnt * 100))


def analysis_DBpedia(DBpedia_type_path: str, DBpedia_fact_path: str):
    regex = re.compile(r'[^"\s]\S*|".+?"@en')
    error_cnt = 0
    entity_set = set()
    with open(DBpedia_fact_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = regex.findall(line.strip())
            if len(line_list) != 4:
                error_cnt += 1
                continue
            head = line_list[0]
            tail = line_list[2]
            if head.startswith('<http://dbpedia.org'):
                entity_set.add(head)
            if tail.startswith('<http://dbpedia.org'):
                entity_set.add(tail)
    entity_isA_set = set()
    with open(DBpedia_type_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split(' ')
            head = line_list[0]
            entity_isA_set.add(head)
    print('before union all_cnt=%d' % (len(entity_set)))
    isA_cnt = len(entity_isA_set)
    # entity_set = entity_set.union(entity_isA_set)   # takes too much time
    all_cnt = len(entity_set)
    print('#IsA=%d over %d, %.2f%%' % (isA_cnt, all_cnt, isA_cnt / all_cnt * 100))


def analysis_Freebase(data_path: str):
    fopen = gzip.open(data_path, 'rt')
    regex = re.compile(r'[^"\s]\S*|".+?"@en')
    taxonomy_rels = ['<http://rdf.freebase.com/ns/type.type.instance>', '<http://rdf.freebase.com/ns/type.object.type>', '<http://rdf.freebase.com/ns/type.property.expected_type>']
    entity_set = set()
    entity_isA_set = set()
    for line in tqdm.tqdm(fopen):
        line_list = regex.findall(line.strip())
        head = line_list[0]
        rel = line_list[1]
        tail = line_list[2]
        entity_set.add(head)
        if tail.startswith('<http://'):
            entity_set.add(head)
        if rel in taxonomy_rels:
            entity_isA_set.add(head)
            if tail.startswith('<http://'):
                entity_isA_set.add(tail)
    fopen.close()
    isA_cnt = len(entity_isA_set)
    all_cnt = len(entity_set)
    print('#IsA=%d over %d, %.2f%%' % (isA_cnt, all_cnt, isA_cnt / all_cnt * 100))


if __name__ == '__main__':
    Yago_en_type_path = 'data/Yago/en/yago-wd-full-types.nt'
    Yago_en_fact_path = 'data/Yago/en/yago-wd-facts.nt'
    # analysis_Yago(Yago_en_type_path, Yago_en_fact_path)
    Yago_full_type_path = 'data/Yago/full/yago-wd-full-types.nt.gz'
    Yago_full_fact_path = 'data/Yago/full/yago-wd-facts.nt.gz'
    # analysis_Yago(Yago_full_type_path, Yago_full_fact_path, 'gz')

    DBpedia_type_path = 'data/DBpedia/instance-types_lang=en_specific.ttl'
    DBpedia_fact_path = 'data/DBpedia/infobox-properties_lang=en.ttl'
    # analysis_DBpedia(DBpedia_type_path, DBpedia_fact_path)

    Freebase_dump_path = 'data/Freebase/freebase-rdf-latest.gz'
    analysis_Freebase(Freebase_dump_path)

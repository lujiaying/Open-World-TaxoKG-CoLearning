"""
Analysis entity percentage of IsA relation
Author: Anonymous Siamese
Create Date: Jul 9, 2021
"""
import re
from collections import defaultdict
import json
import itertools
import random

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


def get_avg_rel_per_node(dataset_dir: str, dataset_type: str):
    train_path = '%s/train.txt' % (dataset_dir)
    dev_path = '%s/valid.txt' % (dataset_dir)
    test_path = '%s/test.txt' % (dataset_dir)
    if dataset_type == 'WN18RR':
        taxo_rel = ['_hypernym', '_instance_hypernym', '_member_meronym', '_synset_domain_topic_of', '_has_part', '_member_of_domain_usage']
    elif dataset_type == 'FB15k':
        taxo_rel = ['/location/location/contains', '/user/ktrueman/default_domain/international_organization/member_states', '/award/award_category/category_of', '/award/award_category/disciplines_or_subjects', '/education/educational_institution/school_type', '/people/profession/specialization_of', '/people/person/profession', '/film/film/genre', '/tv/tv_program/genre', '/music/genre/parent_genre', '/tv/tv_program/tv_producer./tv/tv_producer_term/producer_type', '/tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type', '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category']
    elif dataset_type == 'YAGO3-10':
        taxo_rel = ['IsLocatedIn', 'hasGender', 'isAffiliatedTo', 'hasMusicalRole']
    ent_all_rel_cnt = defaultdict(int)
    ent_taxo_rel_cnt = defaultdict(int)
    with open(train_path) as fopen:
        for line in tqdm.tqdm(fopen):
            h, r, t = line.strip().split('\t')
            ent_all_rel_cnt[h] += 1
            ent_all_rel_cnt[t] += 1
            if r in taxo_rel:
                ent_taxo_rel_cnt[h] += 1
                ent_taxo_rel_cnt[t] += 1
    with open(dev_path) as fopen:
        for line in tqdm.tqdm(fopen):
            h, r, t = line.strip().split('\t')
            ent_all_rel_cnt[h] += 1
            ent_all_rel_cnt[t] += 1
            if r in taxo_rel:
                ent_taxo_rel_cnt[h] += 1
                ent_taxo_rel_cnt[t] += 1
    with open(test_path) as fopen:
        for line in tqdm.tqdm(fopen):
            h, r, t = line.strip().split('\t')
            ent_all_rel_cnt[h] += 1
            ent_all_rel_cnt[t] += 1
            if r in taxo_rel:
                ent_taxo_rel_cnt[h] += 1
                ent_taxo_rel_cnt[t] += 1
    ent_cnt = len(ent_all_rel_cnt)
    avg_rel_per_ent = sum(ent_all_rel_cnt.values()) / ent_cnt
    avg_taxo_rel_per_ent = sum(ent_taxo_rel_cnt.values()) / ent_cnt
    print('%s: taxo/all = %.2f / %.2f' % (dataset_type, avg_taxo_rel_per_ent, avg_rel_per_ent))


def align_taxoas_and_kgs(taxo_path: str, taxo_type: str, kg_dir: str, kg_type: str,
                         alignment_both_end: bool = False, output_path: str = ''):
    taxo_ent_set = set()
    taxo_ori_tuples = set()
    if taxo_type == 'MSCG':
        with open(taxo_path) as fopen:
            for line in tqdm.tqdm(fopen):
                line_list = line.strip().split('\t')
                cep, ent = line_list[0], line_list[1]
                taxo_ent_set.add(ent)
                taxo_ent_set.add(cep)
                taxo_ori_tuples.add((ent, cep))
    elif taxo_type == 'SemEval':
        with open(taxo_path) as fopen:
            for line in tqdm.tqdm(fopen):
                line_list = line.strip().split('\t')
                ent, ceps = line_list[0], line_list[1:]
                ent = ent.lower()
                taxo_ent_set.add(ent)
                for cep in ceps:
                    cep = cep.lower()
                    taxo_ent_set.add(cep)
                    taxo_ori_tuples.add((ent, cep))
    train_path = '%s/train.txt' % (kg_dir)
    dev_path = '%s/valid.txt' % (kg_dir)
    test_path = '%s/test.txt' % (kg_dir)
    kg_ent_set = set()
    kg_ori_taxo_tuples = set()
    if output_path != '':
        fwrite = open(output_path, 'w')
    if kg_type == 'YAGO3-10':
        taxo_rel = ['IsLocatedIn', 'hasGender', 'isAffiliatedTo', 'hasMusicalRole']
        with open(train_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                h = ' '.join(h.lower().split('_'))
                t = ' '.join(t.lower().split('_'))
                kg_ent_set.add(h)
                kg_ent_set.add(t)
                if r in taxo_rel:
                    kg_ori_taxo_tuples.add((h, t))
                if output_path != '':
                    fwrite.write('%s\t%s\t%s\n' % (h, r, t))
        with open(dev_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                h = ' '.join(h.lower().split('_'))
                t = ' '.join(t.lower().split('_'))
                kg_ent_set.add(h)
                kg_ent_set.add(t)
                if r in taxo_rel:
                    kg_ori_taxo_tuples.add((h, t))
                if output_path != '':
                    fwrite.write('%s\t%s\t%s\n' % (h, r, t))
        with open(test_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                h = ' '.join(h.lower().split('_'))
                t = ' '.join(t.lower().split('_'))
                kg_ent_set.add(h)
                kg_ent_set.add(t)
                if r in taxo_rel:
                    kg_ori_taxo_tuples.add((h, t))
                if output_path != '':
                    fwrite.write('%s\t%s\t%s\n' % (h, r, t))
    elif kg_type == 'FB15k':
        taxo_rel = ['/location/location/contains', '/user/ktrueman/default_domain/international_organization/member_states', '/award/award_category/category_of', '/award/award_category/disciplines_or_subjects', '/education/educational_institution/school_type', '/people/profession/specialization_of', '/people/person/profession', '/film/film/genre', '/tv/tv_program/genre', '/music/genre/parent_genre', '/tv/tv_program/tv_producer./tv/tv_producer_term/producer_type', '/tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type', '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category']
        entityid2name_path = '%s/entity2wikidata.json' % (kg_dir)
        with open(entityid2name_path) as fopen:
            id2name_rawmap = json.load(fopen)
        id2name_map = {}
        for k, _ in id2name_rawmap.items():
            name = _['label'].lower()
            id2name_map[k] = name
        with open(train_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                if h in id2name_map:
                    kg_ent_set.add(id2name_map[h])
                if t in id2name_map:
                    kg_ent_set.add(id2name_map[t])
                if r in taxo_rel and h in id2name_map and t in id2name_map:
                    kg_ori_taxo_tuples.add((id2name_map[h], id2name_map[t]))
                if output_path != '' and h in id2name_map and t in id2name_map:
                    fwrite.write('%s\t%s\t%s\n' % (id2name_map[h], r, id2name_map[t]))
        with open(dev_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                if h in id2name_map:
                    kg_ent_set.add(id2name_map[h])
                if t in id2name_map:
                    kg_ent_set.add(id2name_map[t])
                if r in taxo_rel and h in id2name_map and t in id2name_map:
                    kg_ori_taxo_tuples.add((id2name_map[h], id2name_map[t]))
                if output_path != '' and h in id2name_map and t in id2name_map:
                    fwrite.write('%s\t%s\t%s\n' % (id2name_map[h], r, id2name_map[t]))
        with open(test_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                if h in id2name_map:
                    kg_ent_set.add(id2name_map[h])
                if t in id2name_map:
                    kg_ent_set.add(id2name_map[t])
                if r in taxo_rel and h in id2name_map and t in id2name_map:
                    kg_ori_taxo_tuples.add((id2name_map[h], id2name_map[t]))
                if output_path != '' and h in id2name_map and t in id2name_map:
                    fwrite.write('%s\t%s\t%s\n' % (id2name_map[h], r, id2name_map[t]))
    elif kg_type == 'WN18RR':
        taxo_rel = ['_hypernym', '_instance_hypernym', '_member_meronym', '_synset_domain_topic_of', '_has_part', '_member_of_domain_usage']
        definition_path = '%s/wordnet-mlj12-definitions.txt' % (kg_dir)
        id2name_map = {}
        id2rawname_map = {}
        name2rawname_map = defaultdict(set)
        with open(definition_path) as fopen:
            for line in fopen:
                entid, raw_name, descp = line.strip().split('\t')
                name = raw_name.lstrip('_').split('_')[:-2]
                name = ' '.join(name)
                id2name_map[entid] = name
                id2rawname_map[entid] = raw_name
                name2rawname_map[name].add(raw_name)
        with open(train_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                if h in id2name_map:
                    kg_ent_set.add(id2name_map[h])
                if t in id2name_map:
                    kg_ent_set.add(id2name_map[t])
                if r in taxo_rel and h in id2name_map and t in id2name_map:
                    kg_ori_taxo_tuples.add((id2name_map[h], id2name_map[t]))
                if output_path != '' and h in id2rawname_map and t in id2rawname_map:
                    fwrite.write('%s\t%s\t%s\n' % (id2rawname_map[h], r, id2rawname_map[t]))
        with open(dev_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                if h in id2name_map:
                    kg_ent_set.add(id2name_map[h])
                if t in id2name_map:
                    kg_ent_set.add(id2name_map[t])
                if r in taxo_rel and h in id2name_map and t in id2name_map:
                    kg_ori_taxo_tuples.add((id2name_map[h], id2name_map[t]))
                if output_path != '' and h in id2rawname_map and t in id2rawname_map:
                    fwrite.write('%s\t%s\t%s\n' % (id2rawname_map[h], r, id2rawname_map[t]))
        with open(test_path) as fopen:
            for line in tqdm.tqdm(fopen):
                h, r, t = line.strip().split('\t')
                if h in id2name_map:
                    kg_ent_set.add(id2name_map[h])
                if t in id2name_map:
                    kg_ent_set.add(id2name_map[t])
                if r in taxo_rel and h in id2name_map and t in id2name_map:
                    kg_ori_taxo_tuples.add((id2name_map[h], id2name_map[t]))
                if output_path != '' and h in id2rawname_map and t in id2rawname_map:
                    fwrite.write('%s\t%s\t%s\n' % (id2rawname_map[h], r, id2rawname_map[t]))
    # cal overlap between entities
    print('taxo(%s) #ent=%d, kg(%s) #ent=%d' % (taxo_type, len(taxo_ent_set), kg_type, len(kg_ent_set)))
    aligned_ent_cnt = len(taxo_ent_set.intersection(kg_ent_set))
    print('aligned #ent=%d' % (aligned_ent_cnt))
    # cal increased taxo tuples
    print('kg(%s) original taxo #tuple=%d' % (kg_type, len(kg_ori_taxo_tuples)))
    increased_taxo_tuple_cnt = 0
    for ent, cep in taxo_ori_tuples:
        if ent not in kg_ent_set:
            continue
        if alignment_both_end and cep not in kg_ent_set:
            continue
        if (ent, cep) not in kg_ori_taxo_tuples:
            increased_taxo_tuple_cnt += 1
            if output_path != '':
                if kg_type != 'WN18RR':
                    fwrite.write('%s\tIsA\t%s\n' % (ent, cep))
                else:
                    ent_rawnames = list(name2rawname_map[ent])
                    cep_rawnames = list(name2rawname_map[cep])
                    for ent_rawname, cep_rawname in itertools.product(ent_rawnames, cep_rawnames):
                        fwrite.write('%s\tIsA\t%s\n' % (ent_rawname, cep_rawname))
    print('align with taxo(%s) increased #tuple=%d, total=%d' % (taxo_path, increased_taxo_tuple_cnt, increased_taxo_tuple_cnt + len(kg_ori_taxo_tuples)))


def inference_over_taxonomy(data_path: str, data_type: str, relation_triple_outpath: str, hierarchical_out_path: str):
    random.seed(27)
    if data_type == 'WN18RR':
        # taxo_rel = ['_hypernym', '_instance_hypernym', '_member_meronym', '_synset_domain_topic_of', '_has_part', '_member_of_domain_usage']
        taxo_cp_rel = ['_hypernym', '_instance_hypernym', '_synset_domain_topic_of']
        taxo_pc_rel = ['_member_meronym', '_has_part', '_member_of_domain_usage']
    elif data_type == 'FB15k':
        # cp: child - r - parent
        taxo_cp_rel = ['/award/award_category/category_of', '/award/award_category/disciplines_or_subjects', '/education/educational_institution/school_type', '/people/profession/specialization_of', '/people/person/profession', '/film/film/genre', '/tv/tv_program/genre', '/music/genre/parent_genre', '/tv/tv_program/tv_producer./tv/tv_producer_term/producer_type', '/tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type', '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category']
        # pc: parent - r - child
        taxo_pc_rel = ['/location/location/contains', '/user/ktrueman/default_domain/international_organization/member_states']
        # taxo_rel = ['/location/location/contains', '/user/ktrueman/default_domain/international_organization/member_states', '/award/award_category/category_of', '/award/award_category/disciplines_or_subjects', '/education/educational_institution/school_type', '/people/profession/specialization_of', '/people/person/profession', '/film/film/genre', '/tv/tv_program/genre', '/music/genre/parent_genre', '/tv/tv_program/tv_producer./tv/tv_producer_term/producer_type', '/tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type', '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category']
    elif data_type == 'YAGO3-10':
        taxo_cp_rel = ['IsLocatedIn', 'hasGender', 'isAffiliatedTo', 'hasMusicalRole']
        taxo_pc_rel = []
    taxo_cp_rel.append('IsA')
    # sample 100 entities
    # cond: at least one taxo edge, one non-taxo edge
    ent_children_dict = defaultdict(set)   # {p:{c1,c2,}, ...}
    ent_nontaxo_dict = defaultdict(set)    # {h:{(r1,t1),(r2,t2)}, ...}
    with open(data_path) as fopen:
        for line in fopen:
            h, r, t = line.strip().split('\t')
            if r in taxo_cp_rel:
                ent_children_dict[t].add(h)
            elif r in taxo_pc_rel:
                ent_children_dict[h].add(t)
            else:
                ent_nontaxo_dict[h].add((r, t))
    ent_candidates = list(set(ent_children_dict.keys()).intersection(set(ent_nontaxo_dict.keys())))
    ent_candidates = sorted(ent_candidates)
    print('#ent_candidates=%d' % (len(ent_candidates)))
    sampled_ents = random.sample(ent_candidates, k=100)
    # infer relation triples by children
    fwrite = open(relation_triple_outpath, 'w')
    all_freshness = []
    for ent in sampled_ents:
        children = ent_children_dict[ent]
        children_cnt = len(children)
        all_children_rts = defaultdict(float)
        # print('ent=%s, children=%s' % (ent, children))
        for child in children:
            for rt in ent_nontaxo_dict[child]:
                t = rt[1]
                if t == ent:
                    continue
                all_children_rts[rt] += 1.0
        all_children_rts = [(rt, v/children_cnt) for rt, v in all_children_rts.items()]
        if len(all_children_rts) <= 0:
            freshness = 0.0
            fwrite.write('%s\tNone\n' % (ent))
        else:
            all_children_rts = sorted(all_children_rts, key=lambda _: -_[1])[:5]
            # print('all_children_rts:', all_children_rts)
            fwrite.write('%s\t%s\n' % (ent, all_children_rts))
            all_children_rts_no_prob = set(_[0] for _ in all_children_rts)
            ent_rts = ent_nontaxo_dict[ent]
            intersec = all_children_rts_no_prob.intersection(ent_rts)
            freshness = 1.0 - len(intersec) / len(all_children_rts_no_prob)
            # print('freshness=%.2f' % (freshness))
        all_freshness.append(freshness)
    print('relational freshness=%.2f' % (100 * sum(all_freshness) / len(all_freshness)))
    fwrite.close()
    # infer children by relations
    fwrite = open(hierarchical_out_path, 'w')
    rt_ent_dict = defaultdict(set)   # for child candidates
    for ent, rts in ent_nontaxo_dict.items():
        for rt in rts:
            rt_ent_dict[rt].add(ent)
    all_ent_rt_dict = dict()  # store all ent-rt
    with open(data_path) as fopen:
        for line in fopen:
            h, r, t = line.strip().split('\t')
            if (r, t) in rt_ent_dict:
                rt_ent_dict[(r, t)].add(h)
                all_ent_rt_dict[h] = set()
    with open(data_path) as fopen:
        for line in fopen:
            h, r, t = line.strip().split('\t')
            if r in taxo_cp_rel or r in taxo_pc_rel:
                continue
            if h in all_ent_rt_dict:
                all_ent_rt_dict[h].add((r, t))
    all_freshness = []
    for ent in tqdm.tqdm(sampled_ents):
        # find child candidates
        ent_rts = all_ent_rt_dict[ent]
        child_candidates = set()
        for rt in ent_rts:
            children = rt_ent_dict[rt]
            child_candidates.update(children)
        child_candidates.discard(ent)
        child_candidates = {c: 0.0 for c in child_candidates}
        # calculate prob
        for child in child_candidates:
            child_rts = all_ent_rt_dict[child]
            union_rts = ent_rts.union(child_rts)
            nrm = 0.0
            denrm = 0.0
            for rt in union_rts:
                p_child = 1.0 if child in rt_ent_dict[rt] else 0.0
                p_ent = 1.0 if ent in rt_ent_dict[rt] else 0.0
                nrm += (p_child * p_ent)
                denrm += (1 - (1 - p_child) * (1 - p_ent))
            child_candidates[child] = nrm / denrm
        child_candidates = sorted(child_candidates.items(), key=lambda _: -_[1])[:5]
        pred_children = set(_[0] for _ in child_candidates)
        intersec = pred_children.intersection(ent_children_dict[ent])
        if len(pred_children) <= 0:
            freshness = 0.0
        else:
            freshness = 1.0 - len(intersec) / len(pred_children)
        all_freshness.append(freshness)
        fwrite.write('%s\t%s\n' % (ent, child_candidates))
    print('hierarchical freshness=%.2f' % (100 * sum(all_freshness) / len(all_freshness)))
    fwrite.close()


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
    # analysis_Freebase(Freebase_dump_path)
    
    dataset_dir = 'data/WN18RR'
    dataset_type = 'WN18RR'
    dataset_dir = 'data/FB15k-237'
    dataset_type = 'FB15k'
    dataset_dir = 'data/YAGO3-10'
    dataset_type = 'YAGO3-10'
    # get_avg_rel_per_node(dataset_dir, dataset_type)

    taxo_path = 'data/ProBase/data-concept/data-concept-instance-relations.txt'
    taxo_type = 'MSCG'
    # taxo_path = 'data/SemEval2018-Task9/1A.english.merged_pairs.txt'
    # taxo_type = 'SemEval'
    # kg_dir = 'data/YAGO3-10'
    # kg_type = 'YAGO3-10'
    # kg_dir = 'data/FB15k-237'
    # kg_type = 'FB15k'
    kg_dir = 'data/WN18RR'
    kg_type = 'WN18RR'
    # align_taxoas_and_kgs(taxo_path, taxo_type, kg_dir, kg_type)
    # align_taxoas_and_kgs(taxo_path, taxo_type, kg_dir, kg_type, alignment_both_end=True)
    # output_path = 'data/TaxoKG_proposal/SemEval-YAGO3-10.aligned.txt'
    # output_path = 'data/TaxoKG_proposal/MSCG-YAGO3-10.aligned.txt'
    # output_path = 'data/TaxoKG_proposal/SemEval-FB15k.aligned.txt'
    # output_path = 'data/TaxoKG_proposal/MSCG-FB15k.aligned.txt'
    # output_path = 'data/TaxoKG_proposal/SemEval-WN18RR.aligned.txt'
    output_path = 'data/TaxoKG_proposal/MSCG-WN18RR.aligned.txt'
    # align_taxoas_and_kgs(taxo_path, taxo_type, kg_dir, kg_type, alignment_both_end=True, output_path=output_path)

    # data_path = 'data/TaxoKG_proposal/SemEval-FB15k.aligned.txt'
    data_path = 'data/TaxoKG_proposal/MSCG-FB15k.aligned.txt'
    data_type = 'FB15k'
    relational_out_path = 'data/TaxoKG_proposal/MSCG-FB15k.pred_relational_triple.txt'
    hierarchical_out_path = 'data/TaxoKG_proposal/MSCG-FB15k.pred_hierarchical_triple.txt'
    # data_path = 'data/TaxoKG_proposal/SemEval-YAGO3-10.aligned.txt'
    data_path = 'data/TaxoKG_proposal/MSCG-YAGO3-10.aligned.txt'
    data_type = 'YAGO3-10'
    relational_out_path = 'data/TaxoKG_proposal/MSCG-YAGO3-10.pred_relational_triple.txt'
    hierarchical_out_path = 'data/TaxoKG_proposal/MSCG-YAGO3-10.pred_hierarchical_triple.txt'
    # data_path = 'data/TaxoKG_proposal/SemEval-WN18RR.aligned.txt'
    data_path = 'data/TaxoKG_proposal/MSCG-WN18RR.aligned.txt'
    data_type = 'WN18RR'
    relational_out_path = 'data/TaxoKG_proposal/MSCG-WN18RR.pred_relational_triple.txt'
    hierarchical_out_path = 'data/TaxoKG_proposal/MSCG-WN18RR.pred_hierarchical_triple.txt'
    inference_over_taxonomy(data_path, data_type, relational_out_path, hierarchical_out_path)

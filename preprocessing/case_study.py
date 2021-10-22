from collections import Counter

import tqdm


def analysis_concept_neighbors(dataset_dir: str, concept: str):
    cg_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    concept_entities = set()
    with open(cg_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            ent = line_list[0]
            flag = False
            for cept in line_list[1:]:
                if cept == concept:
                    flag = True
            if flag:
                concept_entities.add(ent)
    olp_path = '%s/oie_triples.train.txt' % (dataset_dir)
    rels = Counter()
    ros = Counter()  # (r, o)
    srs = Counter()  # (s, r)
    with open(olp_path) as fopen:
        for line in tqdm.tqdm(fopen):
            s, r, o = line.strip().split('\t')
            if s not in concept_entities and o not in concept_entities:
                continue
            rels[r] += 1
            if s in concept_entities:
                ros[(r, o)] += 1
            if o in concept_entities:
                srs[(s, r)] += 1
    print('Concept=%s in dataset=%s' % (concept, dataset_dir))
    # print('most common r:', rels.most_common(20))
    print('most common (r,o):', ros.most_common(30))
    print('most common (s,r):', srs.most_common(30))


def analysis_relation_neighbors(dataset_dir: str, sro: tuple, sro_type: str):
    """
    Args:
        sro_type: str, 'r', 'sr', 'ro'
    """
    olp_path = '%s/oie_triples.train.txt' % (dataset_dir)
    ents = set()  # subj for input ro
    with open(olp_path) as fopen:
        for line in tqdm.tqdm(fopen):
            s, r, o = line.strip().split('\t')
            if sro_type == 'r' and r == sro:
                ents.add(s)
                ents.add(o)
            if sro_type == 'ro' and (r, o) == sro:
                ents.add(s)
    cg_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cept_counter = Counter()
    with open(cg_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            ent = line_list[0]
            if ent in ents:
                for c in line_list[1:]:
                    cept_counter[c] += 1
    print('[%s]sro=%s in dataset=%s' % (sro_type, sro, dataset_dir))
    print('most common concept:', cept_counter.most_common(20))


if __name__ == '__main__':
    benchmark_data_dir = 'data/CGC-OLP-BENCH/MSCG-ReVerb'
    # analysis_concept_neighbors(benchmark_data_dir, 'technique')
    # analysis_relation_neighbors(benchmark_data_dir, ('be in', 'canada'), 'ro')  # no results
    # analysis_relation_neighbors(benchmark_data_dir, ('be marry to'), 'r')
    benchmark_data_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    # analysis_concept_neighbors(benchmark_data_dir, 'rock music')
    # analysis_relation_neighbors(benchmark_data_dir, ('sound for'), 'r')
    analysis_relation_neighbors(benchmark_data_dir, ('listen to'), 'r')
    benchmark_data_dir = 'data/CGC-OLP-BENCH/SEMedical-ReVerb'
    # analysis_concept_neighbors(benchmark_data_dir, 'disease')
    # analysis_relation_neighbors(benchmark_data_dir, ('die from'), 'r')

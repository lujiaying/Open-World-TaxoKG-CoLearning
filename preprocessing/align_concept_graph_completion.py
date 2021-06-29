"""
Align entity in Concept Graph and OpenIE facts
Author: Jiaying Lu
Create Date: Jun 29, 2021
"""

import time
from collections import defaultdict
from functools import partial

import tqdm


def align_Probase_ReVerb(Probase_path: str, ReVerb_path: str):
    concept_pairs = defaultdict(set)   # c: {parent1, parent2}
    pair_cnt = 0
    tik = time.perf_counter()
    with open(Probase_path) as fopen:
        while True:
            lines = fopen.readlines(204800)
            if not lines:
                break
            for line in lines:
                line_list = line.strip().split('\t')
                pair_cnt += 1
                p = line_list[0]
                c = line_list[1]
                concept_pairs[c].add(p)
        """
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            pair_cnt += 1
            p = line_list[0]
            c = line_list[1]
            concept_pairs[c].add(p)
        """
    elapsed_time = time.perf_counter() - tik
    print('Processed Probase elapsed_time=%s' % (elapsed_time))
    print('Probase # of pairs=%d, unique # of child concept=%d' % (pair_cnt, len(concept_pairs)))

    grounded_subjects = defaultdict(int)
    grounded_objects = defaultdict(int)
    grounded_concepts = defaultdict(int)
    line_cnt = 0
    with open(ReVerb_path) as fopen:
        tik = time.perf_counter()
        while True:
            lines = fopen.readlines(204800)
            if not lines:
                break
            for line in lines:
                line_cnt += 1
                line_list = line.strip().split('\t')
                arg1_norm = line_list[4]
                # rel_norm = line_list[5]
                arg2_norm = line_list[6]
                if arg1_norm in concept_pairs:
                    grounded_subjects[arg1_norm] += 1
                    for p in concept_pairs[arg1_norm]:
                        grounded_concepts[p] += 1
                if arg2_norm in concept_pairs:
                    grounded_objects[arg2_norm] += 1
                    for p in concept_pairs[arg2_norm]:
                        grounded_concepts[p] += 1
            elapsed_time = time.perf_counter() - tik
            print('Processed %d lines, elapsed_time=%s' % (line_cnt, elapsed_time))
            tik = time.perf_counter()
        """
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            arg1_norm = line_list[4]
            # rel_norm = line_list[5]
            arg2_norm = line_list[6]
            if arg1_norm in concept_pairs:
                grounded_subjects[arg1_norm] += 1
                for p in concept_pairs[arg1_norm]:
                    grounded_concepts[p] += 1
            if arg2_norm in concept_pairs:
                grounded_objects[arg2_norm] += 1
                for p in concept_pairs[arg2_norm]:
                    grounded_concepts[p] += 1
        """
    print('Grounded ReVerb concepts=%d, sub=%d, obj=%d' % (len(grounded_concepts), len(grounded_subjects), len(grounded_objects)))
    # cnt > 50, many-shot
    manyshot_concepts = len([(k, v) for k, v in grounded_concepts.items() if v >= 50])
    manyshot_subjects = len([(k, v) for k, v in grounded_subjects.items() if v >= 50])
    manyshot_objects = len([(k, v) for k, v in grounded_objects.items() if v >= 50])
    print('Many shots concepts=%d, sub=%d, obj=%d' % (manyshot_concepts, manyshot_subjects, manyshot_objects))


if __name__ == '__main__':
    Probase_path = 'data/Probase/data-concept/data-concept-instance-relations.txt'
    ReVerb_path = 'data/ReVerb/reverb_clueweb_tuples-1.1.txt'
    align_Probase_ReVerb(Probase_path, ReVerb_path)

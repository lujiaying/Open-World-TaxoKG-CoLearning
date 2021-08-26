"""
The counting based inference, adapted from proposal
Author: Jiaying Lu
Create Date: Aug 24, 2021
"""
from collections import defaultdict
from typing import Tuple, Dict, List


class CountInfer():
    def __init__(self, CGC_pairs: Dict[str, set], OIE_triples: List[Tuple]):
        self.CGC_pairs = CGC_pairs
        self.reverse_CGC_pairs = defaultdict(set)
        for c, ps in CGC_pairs.items():
            for p in ps:
                self.reverse_CGC_pairs[p].add(c)
        # self.OIE_triples = OIE_triples
        self.concepts = set()
        for concepts in CGC_pairs.values():
            for c in concepts:
                self.concepts.add(c)
        self.h_rt = defaultdict(set)
        self.t_rh = defaultdict(set)
        for (h, r, t) in OIE_triples:
            self.h_rt[h].add((r, t))
            self.t_rh[t].add((r, h))

    def infer_taxonomy(self, ent: str) -> List[Tuple[str, float]]:
        concept_scores = []
        ent_rt_set = self.h_rt[ent]
        ent_rh_set = self.t_rh[ent]
        for c in self.concepts:
            c_rt_set = self.h_rt[c]
            rt_intersection = len(ent_rt_set.intersection(c_rt_set))
            rt_union = len(ent_rt_set.union(c_rt_set))
            rt_score = rt_intersection / (rt_union - rt_intersection)
            c_rh_set = self.t_rh[c]
            rh_intersection = len(ent_rh_set.intersection(c_rh_set))
            rh_union = len(ent_rh_set.union(c_rh_set))
            rh_score = rh_intersection / (rh_union - rh_intersection)
            concept_scores.append((c, 0.5 * (rt_score + rh_score)))
        return sorted(concept_scores, key=lambda _: -_[1])

    def infer_relation(self, h: str, r: str, t: str) -> List[Tuple[str, float]]:
        h_ceps = self.CGC_pairs.get(h, set())
        t_score_parent = 0.0
        for h_cep in h_ceps:
            if (r, t) in self.h_rt[h_cep]:
                t_score_parent += 1.0
        t_score_parent = t_score_parent / len(h_ceps) if len(h_ceps) > 0 else 0.0
        h_children = self.reverse_CGC_pairs[h]
        t_score_child = 0.0
        for h_child in h_children:
            if (r, t) in self.h_rt[h_child]:
                t_score_child += 1.0
        t_score_child = t_score_child / len(h_children) if len(h_children) > 0 else 0.0
        t_score = 0.5 * (t_score_parent + t_score_child)
        return t_score

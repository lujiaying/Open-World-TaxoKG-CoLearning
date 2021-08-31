"""
The counting based inference
Author: Jiaying Lu
Create Date: Aug 24, 2021
"""
from collections import defaultdict, Counter
from typing import Tuple, Dict, List
import random


class CountInfer():
    """
    Adpated from TaxoKG proposal
    """
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


class NaiveCountInfer():
    """
    Infer due to dataset bias
    """
    def __init__(self, CGC_pairs: Dict[str, set], OIE_triples: List[Tuple]):
        self.CGC_pairs = CGC_pairs
        self.reverse_CGC_pairs = defaultdict(set)
        for c, ps in CGC_pairs.items():
            for p in ps:
                self.reverse_CGC_pairs[p].add(c)
        # resource for infer_taxonomy
        self.concept_relations = defaultdict(Counter)   # {c1: {(h,r): 3, (r,t): 2,}, c2: {}}
        self.h_rt = defaultdict(set)
        self.t_hr = defaultdict(set)
        for (h, r, t) in OIE_triples:
            if h in CGC_pairs:
                concepts = CGC_pairs[h]
                for c in concepts:
                    self.concept_relations[c][(r, t)] += 1.0
            if t in CGC_pairs:
                concepts = CGC_pairs[t]
                for c in concepts:
                    self.concept_relations[c][(h, r)] += 1.0
            self.h_rt[h].add((r, t))
            self.t_hr[t].add((h, r))
        # add remaining concepts
        for c in self.reverse_CGC_pairs:
            if c not in self.concept_relations:
                self.concept_relations[c] = {}
        # resource for infer_relation
        self.r_t_taxos = defaultdict(lambda: defaultdict(Counter))   # {r1: {t1: {c1: 3, c2: 4}, t2: {}}, r2: {}}
        self.r_h_taxos = defaultdict(lambda: defaultdict(Counter))   # {r1: {h1: {c1: 3, c2: 4}, h2: {}}, r2: {}}
        for (h, r, t) in OIE_triples:
            if t in self.CGC_pairs:
                for p in self.CGC_pairs[t]:
                    self.r_t_taxos[r][t][p] += 1.0
            if t in self.reverse_CGC_pairs:
                for c in self.reverse_CGC_pairs[t]:
                    self.r_t_taxos[r][t][c] += 1.0
            if h in self.CGC_pairs:
                for p in self.CGC_pairs[h]:
                    self.r_h_taxos[r][h][p] += 1.0
            if h in self.reverse_CGC_pairs:
                for c in self.reverse_CGC_pairs[t]:
                    self.r_h_taxos[r][h][c] += 1.0

    def update_all_concepts(self, train_CGC_pairs: dict, dev_CGC_pairs: dict, test_CGC_pairs: dict):
        self.all_concepts = set()
        for ps in train_CGC_pairs.values():
            for p in ps:
                self.all_concepts.add(p)
        for ps in dev_CGC_pairs.values():
            for p in ps:
                self.all_concepts.add(p)
        for ps in test_CGC_pairs.values():
            for p in ps:
                self.all_concepts.add(p)

    def update_all_mentions(self, train_OIE_triples: list, dev_OIE_triples: list, test_OIE_triples: list):
        self.all_mentions = set()
        for (s, r, o) in train_OIE_triples:
            self.all_mentions.add(s)
            self.all_mentions.add(o)
        for (s, r, o) in dev_OIE_triples:
            self.all_mentions.add(s)
            self.all_mentions.add(o)
        for (s, r, o) in test_OIE_triples:
            self.all_mentions.add(s)
            self.all_mentions.add(o)

    def infer_taxonomy(self, ent: str) -> List[Tuple[str, float]]:
        """
        according to surrounding relations
        """
        concept_scores = {}
        ent_rt_set = self.h_rt[ent]
        ent_hr_set = self.t_hr[ent]
        for c, c_rels in self.concept_relations.items():
            ent_rt_scores = [c_rels.get(_, 0) for _ in ent_rt_set]
            ent_hr_scores = [c_rels.get(_, 0) for _ in ent_hr_set]
            concept_scores[c] = sum(ent_rt_scores+ent_hr_scores)
        for c in self.all_concepts:
            if c not in concept_scores:
                concept_scores[c] = 0.0
        concept_scores = [(k, v) for k, v in concept_scores.items()]
        random.shuffle(concept_scores)
        return sorted(concept_scores, key=lambda _: -_[1])

    def infer_relation(self, h: str, r: str, t: str) -> Tuple[dict]:
        """
        according to surrounding taxonomies
        """
        tail_preds = {}
        h_parents = self.CGC_pairs.get(h, set())
        h_children = self.reverse_CGC_pairs.get(h, set())
        h_taxos = h_parents.union(h_children)
        for candidate_t, taxo_dict in self.r_t_taxos[r].items():
            score = 0.0
            for h_taxo in h_taxos:
                score += taxo_dict.get(h_taxo, 0.0)
            tail_preds[candidate_t] = score
        for candidate_t in self.all_mentions:
            if candidate_t not in tail_preds:
                tail_preds[candidate_t] = 0.0
        # basically repeat above process
        head_preds = {}
        t_parents = self.CGC_pairs.get(t, set())
        t_children = self.reverse_CGC_pairs.get(t, set())
        t_taxos = t_parents.union(t_children)
        for candidate_h, taxo_dict in self.r_h_taxos[r].items():
            score = 0.0
            for t_taxo in t_taxos:
                score += taxo_dict.get(t_taxo, 0.0)
            head_preds[candidate_h] = score
        for candidate_h in self.all_mentions:
            if candidate_h not in head_preds:
                head_preds[candidate_h] = 0.0
        return tail_preds, head_preds

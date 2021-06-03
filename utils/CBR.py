"""
Case-based reasoning
From Das et al., A Simple Approach to Case-Based Reasoning in Knowledge Bases,
Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion
Author: Jiaying Lu
Create Date: Jun 1, 2021
"""

from collections import Counter, defaultdict
import numpy as np
import tqdm

from .baselines import load_dataset, generate_corrupted_triples, evaluate_mrr_hits


class TaxoCBR:
    """
    1. Retrieve: find similar entities by Taxo or KNN; then reasoning paths
    2. Reuse: gather paths, store in descending order
    3. Revise: instantiate abstract paths, ground results
    4. Retain: ..
    """
    def __init__(self, train_set: list, dev_set: list, test_set: list, taxo_rels: list, all_paths: dict):
        self.all_triples = set()   # used for corruption
        self.ent_vocab = set()     # used for corruption
        self.hypernym_dict = defaultdict(set)
        self.hyponym_dict = defaultdict(set)
        self.ent2id = {}
        self.rel2id = {'<PAD>': 0}
        self.h_rels = defaultdict(set)   # {h: {(r, t), ()}, }
        self.t_rels = defaultdict(set)   # {t: [(r, h), ()], }
        self.all_paths = all_paths
        self.taxo_rels = taxo_rels
        for (h, r, t) in train_set:
            self.all_triples.add((h, r, t))
            self.ent_vocab.add(h)
            self.ent_vocab.add(t)
            if r in taxo_rels:
                self.hypernym_dict[h].add(t)
                self.hyponym_dict[t].add(h)
            if r not in self.rel2id:
                self.rel2id[r] = len(self.rel2id)
            if h not in self.ent2id:
                self.ent2id[h] = len(self.ent2id)
            if t not in self.ent2id:
                self.ent2id[t] = len(self.ent2id)
            self.h_rels[h].add((r, t))
            self.t_rels[t].add((r, h))
        self.ent_embs = self._cal_rel_based_embedding(self.ent2id, self.rel2id, train_set)
        # to generate corrupted triples, need access to dev, test sets
        for (h, r, t) in (dev_set + test_set):
            self.all_triples.add((h, r, t))
            self.ent_vocab.add(h)
            self.ent_vocab.add(t)

    def _get_hypernyms_siblings(self, ent: str) -> list:
        hypernyms = self.hypernym_dict.get(ent, set())
        if not hypernyms:
            return []
        results = []
        for hypernym in hypernyms:
            siblings = self.hyponym_dict.get(hypernym, set())
            siblings.discard(ent)
            results.append((hypernym, siblings))
        return results

    def _get_hyponyms(self, ent: str) -> list:
        results = self.hyponym_dict.get(ent, set())
        return list(results)

    def _cal_rel_based_embedding(self, ent2id: dict, rel2id: dict, train_set: list) -> dict:
        """
        Each entity is a 2m-hot vector. m is size of relations
        vector entry set to 1 if has at least one edge, otherwise set to 0.
        """
        m = len(rel2id)
        ent_embs = np.zeros((len(ent2id), 2*m))   # <PAD>:0
        for (h, r, t) in train_set:
            ent_embs[ent2id[h], rel2id[r]] = 1
            ent_embs[ent2id[t], m+rel2id[r]] = 1
        ent_embs = np.sqrt(ent_embs)
        l2norm = np.linalg.norm(ent_embs, axis=1)   # shape: len(ent2id)
        l2norm[0] += np.finfo(np.float).eps
        ent_embs = ent_embs / l2norm.reshape(l2norm.shape[0], 1)
        print('entity embedding shape (%d,%d) calc done' % (len(ent2id), 2*m))
        return ent_embs

    def _cal_emb_similarity(self, ent1: str, ent2: str) -> float:
        ent1id = self.ent2id.get(ent1, 0)
        ent2id = self.ent2id.get(ent2, 0)
        return np.dot(self.ent_embs[ent1id], self.ent_embs[ent2id])

    def _retrieve_similar_ents(self, ent: str, rel: str) -> dict:
        results = Counter()
        if rel not in self.taxo_rels:
            # rel is non-taxo rel, using hypernym, sibling, hyponym
            for hypernym, siblings in self._get_hypernyms_siblings(ent):
                results[hypernym] += 1
                results.update(siblings)
            hyponyms = self._get_hyponyms(ent)
            results.update(hyponyms)
        else:
            # rel is taxo rel, using K-NN search
            pass
        return results

    def _find_paths(self, ents: dict, rel: str, is_head: bool) -> dict:
        """
        is_head: true denotes `ents` are heads, false denotes `ents` are tails
        """
        # gather all triples in form (ent, rel, ent?)
        triples = defaultdict(float)
        if is_head:
            # tail prediction, thus search tail
            for (ent, weight) in ents.items():
                for (r, t) in self.h_rels[ent]:
                    if r == rel:
                        triples[(ent, r, t)] += weight
        else:
            # head prediction, thus search head
            # TODO: finish
            pass
        # print('triples from similar ents: ', triples)
        # gather candidate paths
        rules = defaultdict(float)
        for (h, r, t), weight in triples.items():
            for path in self.all_paths[h]:
                if path[-1] == t:
                    # abstract paths into abstract pahts (rules)
                    rel_path = path[0::2]
                    if len(rel_path) == 1 and rel_path[0] == rel:
                        continue
                    rules[rel_path] += weight
        # print('around rules: ', rules)
        return rules

    def _predict_by_paths(self, ent: str, rules: dict, is_head: bool) -> dict:
        preds = defaultdict(float)
        if is_head:
            for path in self.all_paths[ent]:
                rel_path = path[0::2]
                if rel_path in rules:
                    weight = rules[rel_path]
                    tail = path[-1]
                    preds[tail] += weight
        else:
            # TODO: finish
            pass
        # print(preds)
        return preds

    def do_eval(self, test_set: list, max_rule: int = 0):
        """
        max_rule: 0 indicates unlimited.
        """
        all_hit1, all_hit3, all_hit10 = 0, 0, 0
        all_mrr = 0.0
        taxo_cbr_found_cnt = 0
        for (h, r, t) in tqdm.tqdm(test_set):
            cor_hs, cor_ts = generate_corrupted_triples((h, r, t),
                                                        self.ent_vocab,
                                                        self.all_triples)
            # -- tail prediction --
            # Retrieve step
            # print('triple to predict: ', h, r, t)
            similar_ents = self._retrieve_similar_ents(h, r)
            # print('similar ents for h:', similar_ents)
            # Reuse step
            rules = self._find_paths(similar_ents, r, is_head=True)
            if max_rule > 0 and len(rules) > 0:
                rules = sorted(rules.items(), key=lambda _: _[1], reverse=True)[:max_rule]
                rules = dict(rules)
            # Revise step
            pred_tails = self._predict_by_paths(h, rules, is_head=True)
            if len(pred_tails) > 0:
                taxo_cbr_found_cnt += 1
            cor_ts = filter(lambda _: _[2] in pred_tails, cor_ts)
            cor_ts = sorted(cor_ts, key=lambda _: pred_tails[_[2]], reverse=True)
            # print('ranked cor_ts: ', cor_ts[:15])
            hit1, hit3, hit10, mrr = evaluate_mrr_hits((h, r, t), cor_ts)
            # print('MRR=%.3f' % (mrr))
            # print('hits@1,3,10 =%.3f, %.3f, %.3f' % (hit1, hit3, hit10))
            all_hit1 += hit1
            all_hit3 += hit3
            all_hit10 += hit10
            all_mrr += mrr
        all_hit1 /= taxo_cbr_found_cnt   # TODO: to modify after all rels
        all_hit3 /= taxo_cbr_found_cnt
        all_hit10 /= taxo_cbr_found_cnt
        all_mrr /= taxo_cbr_found_cnt
        print('hyperparams max_rule=%d' % (max_rule))
        print('taxo_cbr found %d/%d' % (taxo_cbr_found_cnt, len(test_set)))
        print('MRR=%.3f' % (all_mrr))
        print('hits@1,3,10 =%.3f, %.3f, %.3f' % (all_hit1, all_hit3, all_hit10))


def produce_3hop_path(all_triples: list, out_path: str):
    all_paths = set()
    all_ents = set()
    one_hop_dict = defaultdict(set)
    for (h, r, t) in all_triples:
        if h not in all_ents:
            all_ents.add(h)
        if t not in all_ents:
            all_ents.add(t)
        one_hop_dict[h].add((r, t))
        all_paths.add((h, r, t))   # add one hop paths
    for ent in tqdm.tqdm(all_ents):
        # two, three hop paths
        for (r1, ent1) in one_hop_dict[ent]:
            for (r2, ent2) in one_hop_dict[ent1]:
                # add two hop paths
                if len({ent, ent1, ent2}) == 3:
                    all_paths.add((ent, r1, ent1, r2, ent2))
                for (r3, ent3) in one_hop_dict[ent2]:
                    # add three hop paths
                    if len({ent, ent1, ent2, ent3}) == 4:
                        all_paths.add((ent, r1, ent1, r2, ent2, r3, ent3))
    with open(out_path, 'w') as fwrite:
        for path in all_paths:
            fwrite.write('\t'.join(path) + '\n')
    return


def load_all_path(fname: str) -> dict:
    all_paths = defaultdict(set)
    cnt = 0
    with open(fname) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            st_node = line_list[0]
            all_paths[st_node].add(tuple(line_list[1:]))
            cnt += 1
    print('load all path done, unique entity=%d, total_cnt=%d' % (len(all_paths), cnt))
    return all_paths


if __name__ == '__main__':
    dataset = 'CN100K'    # to set
    max_rule = 15         # to set

    if dataset == 'WN18RR':
        WN18RR_dir = 'data/WN18RR'
        taxo_rels = ['_hypernym', '_instance_hypernym']
        train_set, dev_set, test_set = load_dataset(WN18RR_dir, 'WN18RR')
        max3hop_path_fname = 'data/WN18RR/train_3hop_paths.txt'
    elif dataset == 'CN100K':
        CN100k_dir = 'data/CN-100K'
        taxo_rels = ['IsA']
        train_set, dev_set, test_set = load_dataset(CN100k_dir, 'CN100k')
        max3hop_path_fname = 'data/CN-100K/train_3hop_paths.txt'

    # produce_3hop_path(train_set, max3hop_path_fname)
    all_paths = load_all_path(max3hop_path_fname)
    model = TaxoCBR(train_set, dev_set, test_set, taxo_rels, all_paths)
    model.do_eval(test_set, max_rule=max_rule)

    """
    ent = '03001627'   # chair
    # ent = '01503061'   # bird
    hypernym_sibling = model._get_hypernym_sibling(ent)
    print('taxo related ents for %s: ' % (ent))
    for hypernym, siblings in hypernym_sibling:
        print('hypernym: %s -- %.4f' % (hypernym, model._cal_emb_similarity(ent, hypernym)))
        for sib in siblings:
            print('sibling: %s -- %.4f' % (sib, model._cal_emb_similarity(ent, sib)))
    hyponyms = model._get_hyponym(ent)
    print('hyponym:  ')
    for hyponym in hyponyms:
        print('%s -- %.4f' % (hyponym, model._cal_emb_similarity(ent, hyponym)))
    """

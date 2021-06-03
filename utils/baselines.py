"""
Heuristic Baselines
Author: Jiaying Lu
Create Date: May 19, 2021
"""

import pickle
from collections import Counter, defaultdict

import tqdm


def generate_corrupted_triples(test_triple: set, vocab: dict, shown_triples: set) -> tuple:
    h, r, t = test_triple
    corrupted_h_list = [test_triple]
    corrupted_t_list = [test_triple]
    for ent in vocab:
        if (ent, r, t) != test_triple and (ent, r, t) not in shown_triples:
            corrupted_h_list.append((ent, r, t))
        if (h, r, ent) != test_triple and (h, r, ent) not in shown_triples:
            corrupted_t_list.append((h, r, ent))
    return corrupted_h_list, corrupted_t_list


def evaluate_mrr_hits(gold_triple: tuple, ranked_list: list) -> tuple:
    # hit @1, 3, 10
    hit1 = 1.0 if gold_triple in ranked_list[:1] else 0.0
    hit3 = 1.0 if gold_triple in ranked_list[:3] else 0.0
    hit10 = 1.0 if gold_triple in ranked_list[:10] else 0.0
    # mrr
    if gold_triple not in ranked_list:
        mrr = 0.0
    else:
        rank = ranked_list.index(gold_triple) + 1
        mrr = 1.0 / rank
    return hit1, hit3, hit10, mrr


class MostFrequent:
    def __init__(self, train_set: list, dev_set: list):
        self.hfreq_dict = Counter()   # h: freq
        self.tfreq_dict = Counter()   # t: freq
        self.vocab = Counter()
        self.rel_dict = dict()    # rel: {'h':Counter, 't': Counter}
        self.all_triples = set()
        for (h, r, t) in (train_set + dev_set):
            self.all_triples.add((h, r, t))
            self.vocab[h] += 1
            self.vocab[t] += 1
            self.hfreq_dict[h] += 1
            self.tfreq_dict[t] += 1
            if r not in self.rel_dict:
                self.rel_dict[r] = {'h': Counter(), 't': Counter()}
            self.rel_dict[r]['h'][h] += 1
            self.rel_dict[r]['t'][t] += 1
        print('total h=%d, t=%d, rel=%d' % (
              len(self.hfreq_dict), len(self.tfreq_dict),
              len(self.rel_dict)))

    def predict(self, test_set: list):
        all_hit1, all_hit3, all_hit10 = 0, 0, 0
        all_mrr = 0.0
        for test_triple in tqdm.tqdm(test_set):
            cor_hs, cor_ts = generate_corrupted_triples(test_triple, self.vocab,
                                                        self.all_triples)
            # print('test_triple=%s, len(cor_hs)=%d, len(cor_ts)=%d' % (test_triple, len(cor_hs), len(cor_ts)))
            # two factor rank, rel-specific freq, overall freq
            cor_hs = sorted(cor_hs, key=lambda _: (self.rel_dict[_[1]]['h'][_[0]], self.hfreq_dict[_[0]]), reverse=True)
            hit1, hit3, hit10, mrr = evaluate_mrr_hits(test_triple, cor_hs)
            # print('sorted top10:%s' % (cor_hs[:10]))
            all_hit1 += hit1
            all_hit3 += hit3
            all_hit10 += hit10
            all_mrr += mrr
            cor_ts = sorted(cor_ts, key=lambda _: (self.rel_dict[_[1]]['t'][_[2]], self.tfreq_dict[_[2]]), reverse=True)
            # print('sorted top10:%s' % (cor_ts[:10]))
            hit1, hit3, hit10, mrr = evaluate_mrr_hits(test_triple, cor_ts)
            all_hit1 += hit1
            all_hit3 += hit3
            all_hit10 += hit10
            all_mrr += mrr
        all_hit1 = all_hit1 / len(test_set) / 2.0
        all_hit3 = all_hit3 / len(test_set) / 2.0
        all_hit10 = all_hit10 / len(test_set) / 2.0
        all_mrr = all_mrr / len(test_set) / 2.0
        print('MRR=%.3f' % (all_mrr))
        print('hits@1,3,10 =%.3f, %.3f, %.3f' % (all_hit1, all_hit3, all_hit10))


class HardTaxonomy:
    def __init__(self, train_set: list, dev_set: list, taxo_rels: list):
        self.hfreq_dict = Counter()   # h: freq
        self.tfreq_dict = Counter()   # t: freq
        self.vocab = Counter()
        self.rel_dict = dict()    # rel: {'h':Counter, 't': Counter}
        self.all_triples = set()
        # compute Jaccard sim between two ents
        self.h_rels = {}   # h: {(r, t), }
        self.t_rels = {}   # t: {(r, h), }
        for (h, r, t) in (train_set + dev_set):
            self.all_triples.add((h, r, t))
            self.hfreq_dict[h] += 1
            self.tfreq_dict[t] += 1
            self.vocab[h] += 1
            self.vocab[t] += 1
            if r not in self.rel_dict:
                self.rel_dict[r] = {'h': Counter(), 't': Counter()}
            self.rel_dict[r]['h'][h] += 1
            self.rel_dict[r]['t'][t] += 1
            if h not in self.h_rels:
                self.h_rels[h] = set()
            self.h_rels[h].add((r, t))
            if t not in self.t_rels:
                self.t_rels[t] = set()
            self.t_rels[t].add((r, h))
        """
        self.h_Jaccard = self._compute_Jaccard_scores(self.h_rels)
        with open('data/baselines/HardTaxo-CN100k-hJaccard.pkl', 'wb') as fwrite:
            pickle.dump(self.h_Jaccard, fwrite)
        with open('data/baselines/HardTaxo-CN100k-hJaccard.pkl', 'rb') as fopen:
            self.h_Jaccard = pickle.load(fopen)
        """
        # self.t_Jaccard = self._compute_Jaccard_scores(self.t_rels)
        self.taxo_rels = taxo_rels
        print('Jaccard score computed in __init__')

    def _compute_Jaccard_scores(self, ent_rels: dict) -> dict:
        Jaccard_matrix = {}  # h1: {h2: J12, }
        all_ents = list(ent_rels.keys())
        for idx, h1 in enumerate(tqdm.tqdm(all_ents)):
            for h2 in all_ents[idx+1:]:
                h1_tuples = ent_rels[h1]
                h2_tuples = ent_rels[h2]
                intersect = len(h1_tuples.intersection(h2_tuples))
                union = len(h1_tuples.union(h2_tuples))
                score = intersect / union
                if h1 not in Jaccard_matrix:
                    Jaccard_matrix[h1] = {}
                Jaccard_matrix[h1][h2] = score
                if h2 not in Jaccard_matrix:
                    Jaccard_matrix[h2] = {}
                Jaccard_matrix[h2][h1] = score
        return Jaccard_matrix

    def _get_hypernyms(self, h: str) -> list:
        if h not in self.h_rels:
            return []
        rels = self.h_rels[h]
        hypernyms = []
        for r, t in rels:
            if r in self.taxo_rels:
                hypernyms.append(t)
        return hypernyms

    def _get_hyponyms(self, t: str) -> list:
        if t not in self.t_rels:
            return []
        rels = self.t_rels[t]
        hyponyms = []
        for r, h in rels:
            if r in self.taxo_rels:
                hyponyms.append(h)
        return hyponyms

    def _get_siblings(self, ent: str) -> dict:
        siblings = Counter()
        hypernyms = self._get_hypernyms(ent)
        for hyp in hypernyms:
            sibs = self._get_hyponyms(hyp)
            for sib in sibs:
                if sib == ent:
                    continue
                siblings[sib] += 1
        return siblings

    def _get_ent_by_r(self, ent: str, r: str, all_rels: set) -> list:
        if ent not in all_rels:
            return []
        rels = all_rels[ent]
        results = []
        for r_prime, ent_prime in rels:
            if r_prime == r:
                results.append(ent_prime)
        return results

    def predict(self, test_set: list):
        all_hit1, all_hit3, all_hit10 = 0, 0, 0
        all_mrr = 0.0
        test_h_has_sib = 0
        test_t_has_sib = 0
        for test_triple in tqdm.tqdm(test_set):
            cor_hs, cor_ts = generate_corrupted_triples(test_triple, self.vocab,
                                                        self.all_triples)
            h, r, t = test_triple
            if r in self.taxo_rels:
                """
                # <h, IsA, ?>
                # find similar h', then <h', IsA, t'> gives t' scores
                sim_score_d = defaultdict(float)
                hypernym_score_d = defaultdict(float)
                if h in self.h_Jaccard:
                    sim_score_d.update(self.h_Jaccard[h])
                for h_prime, s in sim_score_d.items():
                    if s <= 0.0:
                        continue
                    hypernyms = self._get_hypernyms(h_prime)
                    for t_prime in hypernyms:
                        hypernym_score_d[t_prime] += s
                cor_ts = sorted(cor_ts,
                                key=lambda _: (hypernym_score_d[_[2]], self.rel_dict[_[1]]['t'][_[2]], self.tfreq_dict[_[2]]),
                                reverse=True)
                # <?, IsA, t>
                # h similar to existing <h', IsA, t>
                hyponyms = self._get_hyponyms(t)
                hyponym_score_d = defaultdict(float)
                for h_prime in hyponyms:
                    if h_prime in self.h_Jaccard:
                        hyponym_score_d.update(self.h_Jaccard[h_prime])
                cor_hs = sorted(cor_hs,
                                key=lambda _: (hyponym_score_d[_[0]], self.rel_dict[_[1]]['h'][_[0]], self.hfreq_dict[_[0]]),
                                reverse=True)
                """
                # just use most frequent
                cor_ts = sorted(cor_ts, key=lambda _: (self.rel_dict[_[1]]['t'][_[2]], self.tfreq_dict[_[2]]), reverse=True)
                cor_hs = sorted(cor_hs, key=lambda _: (self.rel_dict[_[1]]['h'][_[0]], self.hfreq_dict[_[0]]), reverse=True)
            else:
                # <h, r, ?>
                # find siblings h', then suggested by siblings
                siblings = self._get_siblings(h)
                suggested = defaultdict(float)
                if len(siblings) > 0:
                    test_h_has_sib += 1
                # print('siblings for <h=%s, %s, %s>: %s' % (h, r, t, siblings))
                for h_prime, sib_score in siblings.items():
                    t_primes = self._get_ent_by_r(h_prime, r, self.h_rels)
                    for t_prime in t_primes:
                        suggested[t_prime] += (1.0 * (2**sib_score))
                # print('suggested:%s' % (sorted(suggested.items(), key=lambda _: _[1], reverse=True)[:20]))
                cor_ts = sorted(cor_ts,
                                key=lambda _: (suggested[_[2]], self.rel_dict[_[1]]['t'][_[2]], self.tfreq_dict[_[2]]),
                                reverse=True)
                # <?, r, t>
                # finding siblings t', then suggested by siblings
                siblings = self._get_siblings(t)
                if len(siblings) > 0:
                    test_t_has_sib += 1
                suggested = defaultdict(float)
                # print('siblings for <%s, %s, t=%s>: %s' % (h, r, t, siblings))
                # exit(0)
                for t_prime, sib_score in siblings.items():
                    h_primes = self._get_ent_by_r(t_prime, r, self.t_rels)
                    for h_prime in h_primes:
                        suggested[h_prime] += (1.0 * (2**sib_score))
                cor_hs = sorted(cor_hs,
                                key=lambda _: (suggested[_[0]], self.rel_dict[_[1]]['h'][_[0]], self.hfreq_dict[_[0]]),
                                reverse=True)
            hit1, hit3, hit10, mrr = evaluate_mrr_hits(test_triple, cor_hs)
            # print('sorted top10:%s' % (cor_hs[:10]))
            all_hit1 += hit1
            all_hit3 += hit3
            all_hit10 += hit10
            all_mrr += mrr
            cor_ts = sorted(cor_ts, key=lambda _: (self.rel_dict[_[1]]['t'][_[2]], self.tfreq_dict[_[2]]), reverse=True)
            # print('sorted top10:%s' % (cor_ts[:10]))
            hit1, hit3, hit10, mrr = evaluate_mrr_hits(test_triple, cor_ts)
            all_hit1 += hit1
            all_hit3 += hit3
            all_hit10 += hit10
            all_mrr += mrr
        print('test_h_has_sib=%d (%.3f), test_t_has_sib=%d (%.3f)' % (
              test_h_has_sib, test_h_has_sib/len(test_set),
              test_t_has_sib, test_t_has_sib/len(test_set)))
        all_hit1 = all_hit1 / len(test_set) / 2.0
        all_hit3 = all_hit3 / len(test_set) / 2.0
        all_hit10 = all_hit10 / len(test_set) / 2.0
        all_mrr = all_mrr / len(test_set) / 2.0
        print('MRR=%.3f' % (all_mrr))
        print('hits@1,3,10 =%.3f, %.3f, %.3f' % (all_hit1, all_hit3, all_hit10))


def load_dataset(dataset_dir: str, dataset_name: str) -> tuple:
    train_set = list()
    dev_set = list()
    test_set = list()
    if dataset_name == 'CN100k':
        with open('%s/train100k.txt' % (dataset_dir)) as fopen:
            for line in fopen:
                line_list = line.strip().split('\t')
                head, tail = line_list[1], line_list[2]
                relation = line_list[0]
                train_set.append((head, relation, tail))
        for fname in ['dev1.txt', 'dev2.txt']:
            with open('%s/%s' % (dataset_dir, fname)) as fopen:
                for line in fopen:
                    line_list = line.strip().split('\t')
                    head, tail = line_list[1], line_list[2]
                    relation = line_list[0]
                    dev_set.append((head, relation, tail))
        with open('%s/test.txt' % (dataset_dir)) as fopen:
            for line in fopen:
                line_list = line.strip().split('\t')
                if line_list[-1] == '0':
                    continue
                head, tail = line_list[1], line_list[2]
                relation = line_list[0]
                test_set.append((head, relation, tail))
    elif dataset_name == 'WN18RR':
        with open('%s/train.txt' % (dataset_dir)) as fopen:
            for line in fopen:
                line_list = line.strip().split('\t')
                head, tail = line_list[0], line_list[2]
                relation = line_list[1]
                train_set.append((head, relation, tail))
        with open('%s/valid.txt' % (dataset_dir)) as fopen:
            for line in fopen:
                line_list = line.strip().split('\t')
                head, tail = line_list[0], line_list[2]
                relation = line_list[1]
                dev_set.append((head, relation, tail))
        with open('%s/test.txt' % (dataset_dir)) as fopen:
            for line in fopen:
                line_list = line.strip().split('\t')
                head, tail = line_list[0], line_list[2]
                relation = line_list[1]
                test_set.append((head, relation, tail))
    else:
        print('dataset_name should in [CN100k, WN18RR]')
        exit(-1)
    return train_set, dev_set, test_set


if __name__ == '__main__':
    CN100k_dir = 'data/CN-100K'
    taxo_rels = ['IsA']
    train_set, dev_set, test_set = load_dataset(CN100k_dir, 'CN100k')
    WN18RR_dir = 'data/WN18RR'
    # train_set, dev_set, test_set = load_dataset(WN18RR_dir, 'WN18RR')
    # model = MostFrequent(train_set, dev_set)
    model = HardTaxonomy(train_set, dev_set, taxo_rels)
    model.predict(test_set)

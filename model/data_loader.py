"""
Data Loaders for WN18RR and CN100K
Author: Jiaying Lu
Create Date: Jun 7, 2021
"""
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy import sparse as spsp
import torch as th
from torch.utils import data
from allennlp.common.util import pad_sequence_to_length

from utils.baselines import load_dataset


PAD_idx = 0


class LinkPredDst(data.Dataset):
    def __init__(self, triples: list, ent_vocab: dict, rel_vocab: dict):
        self.triples = triples
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        (h, r, t) = self.triples[idx]
        h = self.ent_vocab.get(h, self.ent_vocab['<UNK>'])
        t = self.ent_vocab.get(t, self.ent_vocab['<UNK>'])
        r = self.rel_vocab.get(r)
        return h, r, t


def get_vocabs(train_triples: list) -> tuple:
    ent_vocab = {'<PAD>': PAD_idx, '<UNK>': PAD_idx+1}
    rel_vocab = {}
    for (h, r, t) in train_triples:
        if r not in rel_vocab:
            rel_vocab[r] = len(rel_vocab)
        if h not in ent_vocab:
            ent_vocab[h] = len(ent_vocab)
        if t not in ent_vocab:
            ent_vocab[t] = len(ent_vocab)
    return ent_vocab, rel_vocab


def sample_negative_triples(batch_h: th.LongTensor, batch_r: th.LongTensor, batch_t: th.LongTensor, ent_vocab: dict, known_triples: set) -> th.LongTensor:
    head_or_tail = th.randint(high=2, size=batch_h.size())   # shape=(batch,)
    random_entities = th.randint(high=len(ent_vocab), size=batch_h.size())
    broken_heads = th.where(head_or_tail == 1, random_entities, batch_h)
    broken_tails = th.where(head_or_tail == 0, random_entities, batch_t)
    negative_triples = th.stack((broken_heads, batch_r, broken_tails), dim=1)
    # sanity check
    for idx, triple in enumerate(negative_triples):
        h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
        if (h, r, t) not in known_triples:
            continue
        while (h, r, t) in known_triples:
            if head_or_tail[idx].item() == 1:
                # resample head
                h = random.randint(0, len(ent_vocab)-1)
            else:
                # resample tail
                t = random.randint(0, len(ent_vocab)-1)
        negative_triples[idx] = negative_triples.new_tensor((h, r, t))
    return negative_triples


def prepare_ingredients(dataset_dir: str, corpus_type: str)\
        -> Tuple[data.Dataset, data.Dataset, data.Dataset, dict, dict, set, dict]:
    train_triples, dev_triples, test_triples = load_dataset(dataset_dir, corpus_type)
    ent_vocab, rel_vocab = get_vocabs(train_triples)
    train_set = LinkPredDst(train_triples, ent_vocab, rel_vocab)
    dev_set = LinkPredDst(dev_triples, ent_vocab, rel_vocab)
    test_set = LinkPredDst(test_triples, ent_vocab, rel_vocab)
    # triples consit of ids for negative sampling and evaluation
    train_triple_ids = set([(ent_vocab[h], rel_vocab[r], ent_vocab[t]) for (h, r, t) in train_triples])
    all_triple_ids_map = {'h': defaultdict(set),
                          't': defaultdict(set)}  # resources for corrupted triples evaluation
    for (h, r, t) in (train_triples + dev_triples + test_triples):
        if h not in ent_vocab or t not in ent_vocab:
            continue
        all_triple_ids_map['h'][(ent_vocab[h], rel_vocab[r])].add(ent_vocab[t])
        all_triple_ids_map['t'][(ent_vocab[t], rel_vocab[r])].add(ent_vocab[h])
    return train_set, dev_set, test_set, ent_vocab, rel_vocab, train_triple_ids, all_triple_ids_map


def get_taxo_relations(corpus_type: str) -> list:
    if corpus_type == 'WN18RR':
        taxo_rels = ['_hypernym', '_instance_hypernym']
    elif corpus_type == 'CN100k':
        taxo_rels = ['IsA']
    else:
        print('get_taxo_relations() invalid corpus_type=%s' % (corpus_type))
        exit(-1)
    return taxo_rels


def get_taxo_parents_children(train_triple_ids: set, rel_vocab: dict, corpus_type: str) -> dict:
    taxo_rels = get_taxo_relations(corpus_type)
    taxo_rels = [rel_vocab[_] for _ in taxo_rels]
    taxo_dict = {'p': defaultdict(list), 'c': defaultdict(list)}
    for (h, r, t) in train_triple_ids:
        if r not in taxo_rels:
            continue
        if t not in taxo_dict['p'][h]:
            taxo_dict['p'][h].append(t)   # h's parent is t
        if h not in taxo_dict['c'][t]:
            taxo_dict['c'][t].append(h)   # t's child is h
    return taxo_dict


def scipy_sparse2_torch(adj: spsp.coo_matrix, size: tuple) -> th.sparse_coo_tensor:
    if isinstance(adj, spsp.csr_matrix):
        adj = adj.tocoo()
    i = th.LongTensor(np.vstack((adj.row, adj.col)))
    v = th.FloatTensor(adj.data)
    adj_th = th.sparse_coo_tensor(i, v, size)   # normalized adj matrix
    return adj_th


def prepare_batch_taxo_ents(batch_ent: th.LongTensor, taxo_dict: dict) -> Tuple[Tuple[th.LongTensor, th.LongTensor], Tuple[th.LongTensor, th.LongTensor]]:
    """
    Args:
        batch_ent: shape = (batch, )
    """
    batch_p = []
    batch_c = []
    lens_p = []
    lens_c = []
    for ent in batch_ent:
        p = taxo_dict['p'][ent.item()]
        c = taxo_dict['c'][ent.item()]
        batch_p.append(p)
        batch_c.append(c)
        lens_p.append(len(p))
        lens_c.append(len(c))
    max_len_p = max(lens_p)
    max_len_c = max(lens_c)
    batch_p = [pad_sequence_to_length(_, max_len_p, lambda: PAD_idx) for _ in batch_p]
    batch_c = [pad_sequence_to_length(_, max_len_c, lambda: PAD_idx) for _ in batch_c]
    batch_p = batch_ent.new_tensor(batch_p)   # (batch, max_len)
    batch_c = batch_ent.new_tensor(batch_c)   # (batch, max_len)
    return (batch_p, batch_ent.new_tensor(lens_p)), (batch_c, batch_ent.new_tensor(lens_c))


def get_normalized_adj_matrix(adj_dict: dict, ent_count: int, norm: str) -> spsp.csr_matrix:
    """
    Args:
        norm: str, options ['asym', 'sym']
            'asym': D^-1 * A
            'sym': D^-1/2 * A * D^-1/2
    """
    # construct sparse adj matrix,
    # option 1: from scipy
    # option 2: from pytorch, however 1.7 not support two sparse mat multiply
    row = []   # store coord
    col = []   # store coord
    data = []     # store entry values
    for h, t_set in adj_dict.items():
        for t in t_set:
            row.append(h)
            col.append(t)
            data.append(1.0)
    adj = spsp.coo_matrix((data, (row, col)), shape=(ent_count, ent_count))   # (ent_c, ent_c)
    D_inv = np.array(adj.sum(1)).squeeze()   # (ent_c, ) vector
    if norm == 'asym':
        D_inv = np.nan_to_num(np.power(D_inv, -1.0), posinf=0.0, neginf=0.0)
        D_inv = spsp.diags(D_inv)    # (ent_c, ent_c)
        adj = D_inv.dot(adj)       # (ent_c, ent_c)
    elif norm == 'sym':
        D_inv = np.nan_to_num(np.power(D_inv, -0.5), posinf=0.0, neginf=0.0)
        D_inv = spsp.diags(D_inv)  # (ent_c, ent_c)
        adj = D_inv.dot(adj).dot(D_inv)    # (ent_c, ent_c)
        # adj = adj.tocoo()   # csr -> coo
    else:
        print('ERROR get_normalized_adj_matrix invalid norm=%s' % (norm))
        exit(-1)
    return adj


def load_WN18RR_definition(fpath: str = "data/WN18RR/wordnet-mlj12-definitions.txt") -> dict:
    wordnet_def = {}
    with open(fpath) as fopen:
        for line in fopen:
            wid, word, explain = line.strip().split('\t')
            wordnet_def[wid] = word
    return wordnet_def


if __name__ == '__main__':
    WN18RR_dir = 'data/WN18RR'
    train_set, dev_set, test_set, ent_vocab, rel_vocab, all_triples = prepare_ingredients(WN18RR_dir, 'WN18RR')
    dev_iter = data.DataLoader(dev_set, batch_size=4, shuffle=True)
    for batch_h, batch_r, batch_t in dev_iter:
        print(batch_h.shape)
        print(batch_h)
        exit(0)

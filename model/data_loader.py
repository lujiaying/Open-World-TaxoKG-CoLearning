"""
Data Loaders for WN18RR and CN100K
Author: Jiaying Lu
Create Date: Jun 7, 2021
"""
import random

import torch as th
from torch.utils import data
from typing import Tuple

from utils.baselines import load_dataset


class LinkPredDst(data.Dataset):
    def __init__(self, triples: list, ent_vocab: dict, rel_vocab: dict):
        self.triples = triples
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        (h, r, t) = self.triples[idx]
        h = self.ent_vocab.get(h, self.ent_vocab['<PAD>'])
        t = self.ent_vocab.get(t, self.ent_vocab['<PAD>'])
        r = self.rel_vocab.get(r)
        return h, r, t


def get_vocabs(train_triples: list) -> tuple:
    ent_vocab = {'<PAD>': 0}
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
                h = random.randint(0, len(ent_vocab))
            else:
                # resample tail
                t = random.randint(0, len(ent_vocab))
        negative_triples[idx] = negative_triples.new_tensor((h, r, t))
    return negative_triples


def prepare_ingredients(dataset_dir: str, corpus_type: str)\
        -> Tuple[data.Dataset, data.Dataset, data.Dataset, dict, dict, set, set]:
    train_triples, dev_triples, test_triples = load_dataset(dataset_dir, corpus_type)
    ent_vocab, rel_vocab = get_vocabs(train_triples)
    train_set = LinkPredDst(train_triples, ent_vocab, rel_vocab)
    dev_set = LinkPredDst(dev_triples, ent_vocab, rel_vocab)
    test_set = LinkPredDst(test_triples, ent_vocab, rel_vocab)
    # triples consit of ids for negative sampling and evaluation
    train_triple_ids = set([(ent_vocab[h], rel_vocab[r], ent_vocab[t]) for (h, r, t) in train_triples])
    all_triple_ids = set()  # resources for corrupted triples evaluation
    for (h, r, t) in (train_triples + dev_triples + test_triples):
        if h not in ent_vocab or t not in ent_vocab:
            continue
        all_triple_ids.add((ent_vocab[h], rel_vocab[r], ent_vocab[t]))
    return train_set, dev_set, test_set, ent_vocab, rel_vocab, train_triple_ids, all_triple_ids


if __name__ == '__main__':
    WN18RR_dir = 'data/WN18RR'
    train_set, dev_set, test_set, ent_vocab, rel_vocab, all_triples = prepare_ingredients(WN18RR_dir, 'WN18RR')
    dev_iter = data.DataLoader(dev_set, batch_size=4, shuffle=True)
    for batch_h, batch_r, batch_t in dev_iter:
        print(batch_h.shape)
        print(batch_h)
        exit(0)

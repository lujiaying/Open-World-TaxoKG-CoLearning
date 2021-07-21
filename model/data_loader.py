"""
Data Loaders for CGC-OLP-BENCH
Author: Jiaying Lu
Create Date: Jul 14, 2021
"""
import random
from collections import defaultdict
from typing import Tuple, Dict, List

import torch as th
from torch.utils import data
from allennlp.common.util import pad_sequence_to_length

PAD_idx = 0


class CGCOLPTriplesDst(data.Dataset):
    def __init__(self, triples: list, tok_vocab: dict):
        self.triples = []
        for h, r, t in triples:
            h_num = [tok_vocab.get(_, PAD_idx) for _ in h.split(' ')]
            r_num = [tok_vocab.get(_, PAD_idx) for _ in r.split(' ')]
            t_num = [tok_vocab.get(_, PAD_idx) for _ in t.split(' ')]
            self.triples.append((h_num, r_num, t_num))

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[list, list, list]:
        (h, r, t) = self.triples[idx]
        return h, r, t


def collate_fn_triples(data: list) -> tuple:
    hs, rs, ts = zip(*data)
    h_lens = [len(_) for _ in hs]
    r_lens = [len(_) for _ in rs]
    t_lens = [len(_) for _ in ts]
    h_max_len = max(h_lens)
    r_max_len = max(r_lens)
    t_max_len = max(t_lens)
    h_batch = [pad_sequence_to_length(_, h_max_len, lambda: PAD_idx) for _ in hs]
    r_batch = [pad_sequence_to_length(_, r_max_len, lambda: PAD_idx) for _ in rs]
    t_batch = [pad_sequence_to_length(_, t_max_len, lambda: PAD_idx) for _ in ts]
    return th.LongTensor(h_batch), th.LongTensor(r_batch), th.LongTensor(t_batch), th.LongTensor(h_lens), th.LongTensor(r_lens), th.LongTensor(t_lens)


class CGCPairsDst(data.Dataset):
    def __init__(self, cg_pairs: Dict[str, set], tok_vocab: dict, concept_vocab: dict):
        self.pairs = []
        for ent, concepts in cg_pairs.items():
            ent_num = [tok_vocab.get(_, PAD_idx) for _ in ent.split(' ')]  # L-token length
            concepts_nums = [concept_vocab[_] for _ in concepts]   # k concepts length
            self.pairs.append((ent_num, concepts_nums))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[list, list]:
        (ent, ceps) = self.pairs[idx]
        return ent, ceps


def collate_fn_CGCpairs(data) -> Tuple[th.LongTensor, list]:
    ent_l, ceps_l = zip(*data)
    ent_lens = [len(_) for _ in ent_l]
    ent_max_len = max(ent_lens)
    ent_batch = [pad_sequence_to_length(_, ent_max_len, lambda: PAD_idx) for _ in ent_l]
    return th.LongTensor(ent_batch), ceps_l, th.LongTensor(ent_lens)


def load_cg_pairs(fpath: str) -> Dict[str, set]:
    concept_pairs = dict()   # ent: {cep1, cep2}
    with open(fpath) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            ent, concepts = line_list[0], line_list[1:]
            concept_pairs[ent] = concepts
    return concept_pairs


def cg_pairs_to_cg_triples(concept_pairs: Dict[str, set]) -> List[Tuple[str, str, str]]:
    triples = []
    for ent, cep_set in concept_pairs.items():
        for cep in cep_set:
            triples.append((ent, "IsA", cep))
    return triples


def analysis_concept_token_existence(dataset_dir: str):
    print('analysis_concept_token_existence for %s' % (dataset_dir))
    cg_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    # analysis overlapping of concept in train and test set
    concepts_train = set([c for ceps in load_cg_pairs(cg_train_path).values() for c in ceps])
    concepts_test = set([c for ceps in load_cg_pairs(cg_test_path).values() for c in ceps])
    intersc = concepts_train.intersection(concepts_test)
    print('#train_cep=%d, #test_cep=%d. #%d(%.4f test) in train_cep' % (len(concepts_train), len(concepts_test), len(intersc), len(intersc)/len(concepts_test)))
    # analysis whether all concept tokens have shown in train set (both CG and OKG)
    concepts_tok_test = set(tok for cep in concepts_test for tok in cep.split(' '))
    all_tok = set()
    for cep in concepts_train:
        for tok in cep.split(' '):
            all_tok.add(tok)
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    with open(oie_train_path) as fopen:
        for line in fopen:
            try:
                subj, rel, obj = line.strip().split('\t')
            except:
                print(line)
            for tok in ('%s %s %s' % (subj, rel, obj)).split(' '):
                all_tok.add(tok)
    intersc = all_tok.intersection(concepts_tok_test)
    print('#all_tok=%d, #test_cep_tok=%d. #%d(%.4f test) in all_tok' % (len(all_tok), len(concepts_tok_test), len(intersc), len(intersc)/len(concepts_tok_test)))


def load_oie_triples(fpath: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(fpath) as fopen:
        for line in fopen:
            subj, rel, obj = line.strip().split('\t')
            triples.append((subj, rel, obj))
    return triples


def get_vocabs(cg_triples_train: list, oie_triples_train: list) -> Tuple[dict, dict]:
    tok_vocab = {'<PAD>': PAD_idx, '<UNK>': PAD_idx+1}
    mention_vocab = {}
    for subj, rel, obj in oie_triples_train:
        if subj not in mention_vocab:
            mention_vocab[subj] = len(mention_vocab)
        if obj not in mention_vocab:
            mention_vocab[obj] = len(mention_vocab)
        for tok in (' '.join([subj, rel, obj])).split(' '):
            if tok not in tok_vocab:
                tok_vocab[tok] = len(tok_vocab)
    for ent, rel, cep in cg_triples_train:
        for tok in (' '.join([ent, rel, cep])).split(' '):
            if tok not in tok_vocab:
                tok_vocab[tok] = len(tok_vocab)
    return tok_vocab, mention_vocab


def prepare_ingredients_transE(dataset_dir: str) -> tuple:
    # Load Concept Graph
    cg_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cg_dev_path = '%s/cg_pairs.dev.txt' % (dataset_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    cg_pairs_train = load_cg_pairs(cg_train_path)
    cg_pairs_dev = load_cg_pairs(cg_dev_path)
    cg_pairs_test = load_cg_pairs(cg_test_path)
    concept_vocab = {}
    for cep_set in cg_pairs_train.values():
        for cep in cep_set:
            if cep not in concept_vocab:
                concept_vocab[cep] = len(concept_vocab)
    for cep_set in cg_pairs_dev.values():
        for cep in cep_set:
            if cep not in concept_vocab:
                concept_vocab[cep] = len(concept_vocab)
    for cep_set in cg_pairs_test.values():
        for cep in cep_set:
            if cep not in concept_vocab:
                concept_vocab[cep] = len(concept_vocab)
    cg_triples_train = cg_pairs_to_cg_triples(cg_pairs_train)
    cg_triples_dev = cg_pairs_to_cg_triples(cg_pairs_dev)
    cg_triples_test = cg_pairs_to_cg_triples(cg_pairs_test)
    # Load Open KG
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    oie_dev_path = '%s/oie_triples.dev.txt' % (dataset_dir)
    oie_test_path = '%s/oie_triples.test.txt' % (dataset_dir)
    oie_triples_train = load_oie_triples(oie_train_path)
    oie_triples_dev = load_oie_triples(oie_dev_path)
    oie_triples_test = load_oie_triples(oie_test_path)
    tok_vocab, mention_vocab = get_vocabs(cg_triples_train, oie_triples_train)
    train_set = CGCOLPTriplesDst(cg_triples_train+oie_triples_train, tok_vocab)
    dev_cg_set = CGCPairsDst(cg_pairs_dev, tok_vocab, concept_vocab)
    dev_oie_set = CGCOLPTriplesDst(oie_triples_dev, tok_vocab)
    test_cg_set = CGCPairsDst(cg_pairs_test, tok_vocab, concept_vocab)
    test_oie_set = CGCOLPTriplesDst(oie_triples_test, tok_vocab)
    return train_set, dev_cg_set, dev_oie_set, test_cg_set, test_oie_set, tok_vocab, mention_vocab, concept_vocab


def get_concept_tok_tensor(concept_vocab: dict, tok_vocab: dict) -> th.LongTensor:
    concepts = []
    cep_lens = []
    for cep in concept_vocab.keys():
        cep_num = [tok_vocab.get(_, PAD_idx) for _ in cep.split(' ')]
        concepts.append(cep_num)
        cep_lens.append(len(cep_num))
    max_len = max(cep_lens)
    concepts = [pad_sequence_to_length(_, max_len, lambda: PAD_idx) for _ in concepts]
    return th.LongTensor(concepts), th.LongTensor(cep_lens)


if __name__ == '__main__':
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/MSCG-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-OPIEC'
    dataset_dir = 'data/CGC-OLP-BENCH/MSCG-OPIEC'
    # analysis_concept_token_existence(dataset_dir)

    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    # train_set, tok_vocab, mention_vocab, concept_vocab = prepare_ingredients_transE(dataset_dir)
    # train_iter = data.DataLoader(train_set, collate_fn=collate_fn_triples, batch_size=4, shuffle=True)

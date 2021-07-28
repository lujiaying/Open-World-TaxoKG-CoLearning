"""
Data Loaders for CGC-OLP-BENCH
Author: Jiaying Lu
Create Date: Jul 14, 2021
"""
import random
from collections import defaultdict
from typing import Tuple, Dict, List

import tqdm
import torch as th
from torch.utils import data
from allennlp.common.util import pad_sequence_to_length
import networkx as nx
import dgl

PAD_idx = 0
UNK_idx = PAD_idx+1
SELF_LOOP = "SELF_LOOP"


class CGCOLPTriplesDst(data.Dataset):
    def __init__(self, triples: list, tok_vocab: dict):
        self.triples = []
        for h, r, t in triples:
            h_num = [tok_vocab.get(_, UNK_idx) for _ in h.split(' ')]
            r_num = [tok_vocab.get(_, UNK_idx) for _ in r.split(' ')]
            t_num = [tok_vocab.get(_, UNK_idx) for _ in t.split(' ')]
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
    return (th.LongTensor(h_batch), th.LongTensor(r_batch), th.LongTensor(t_batch),
            th.LongTensor(h_lens), th.LongTensor(r_lens), th.LongTensor(t_lens))


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


class OLPTriplesDst(data.Dataset):
    def __init__(self, triples: list, tok_vocab: dict, mention_vocab: dict, rel_vocab: dict):
        self.triples = []
        for h, r, t in triples:
            h_num = [tok_vocab.get(_, PAD_idx) for _ in h.split(' ')]
            r_num = [tok_vocab.get(_, PAD_idx) for _ in r.split(' ')]
            t_num = [tok_vocab.get(_, PAD_idx) for _ in t.split(' ')]
            h_mid = mention_vocab[h]
            r_rid = rel_vocab[r]
            t_mid = mention_vocab[t]
            self.triples.append((h_num, r_num, t_num, h_mid, r_rid, t_mid))

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[list, list, list, int, int, int]:
        return self.triples[idx]


def collate_fn_oie_triples(data: list) -> tuple:
    hs, rs, ts, h_mids, r_rids, t_mids = zip(*data)
    h_lens = [len(_) for _ in hs]
    r_lens = [len(_) for _ in rs]
    t_lens = [len(_) for _ in ts]
    h_max_len = max(h_lens)
    r_max_len = max(r_lens)
    t_max_len = max(t_lens)
    h_batch = [pad_sequence_to_length(_, h_max_len, lambda: PAD_idx) for _ in hs]
    r_batch = [pad_sequence_to_length(_, r_max_len, lambda: PAD_idx) for _ in rs]
    t_batch = [pad_sequence_to_length(_, t_max_len, lambda: PAD_idx) for _ in ts]
    return (th.LongTensor(h_batch), th.LongTensor(r_batch), th.LongTensor(t_batch),
            th.LongTensor(h_lens), th.LongTensor(r_lens), th.LongTensor(t_lens),
            h_mids, r_rids, t_mids)


class CGCEgoGraphDst(data.Dataset):
    def __init__(self, cg_pairs: Dict[str, set], oie_triples: List[Tuple[str, str, str]], tok_vocab: dict):
        self.graphs = []
        DG = nx.DiGraph()
        for subj, rel, obj in oie_triples:
            DG.add_edge(subj, obj, rel=rel)
        # add self-loops
        for n in DG:
            DG.add_edge(n, n, rel=SELF_LOOP)
        for ent in cg_pairs:
            if ent in DG:
                continue
            DG.add_edge(ent, ent, rel=SELF_LOOP)
        for ent, ceps in tqdm.tqdm(cg_pairs.items()):
            ego_graph = nx.generators.ego.ego_graph(DG, ent, radius=2, undirected=True)
            cep_tids = [[tok_vocab.get(t, UNK_idx) for t in c.split(' ')] for c in ceps]
            node_id_map = {ent: 0}  # {mention: nid}
            edge_tids = []  # [[tid1, tid2, ...], []]
            u_l = []
            v_l = []
            for n in ego_graph.nodes:
                if n not in node_id_map:
                    node_id_map[n] = len(node_id_map)
            for (u, v, rel) in ego_graph.edges.data('rel'):
                u_l.append(node_id_map[u])
                v_l.append(node_id_map[v])
                edge_tids.append([tok_vocab.get(t, UNK_idx) for t in rel.split(' ')])
            u_l = th.tensor(u_l)
            v_l = th.tensor(v_l)
            g = dgl.graph((u_l, v_l))
            node_tids = [[] for _ in range(len(node_id_map))]
            for ent, nid in node_id_map.items():
                node_tids[nid] = [tok_vocab.get(t, UNK_idx) for t in ent.split(' ')]
            # TODO: cep_tids use target tensor to replace; then BCEWithLogitsLoss can be applied
            self.graphs.append((g, node_tids, edge_tids, cep_tids))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        return self.graphs[idx]

    @staticmethod
    def collate_fn(data: list) -> tuple:
        g_l, node_tids_l, edge_tids_l, cep_tids_l = zip(*data)
        bg = dgl.batch(g_l)
        # print('batched graphs batch_size=%s, num of nodes=%s, num of edges=%s' % (bg.batch_size, bg.batch_num_nodes(), bg.batch_num_edges()))
        # 1D list for all nodes, edges, concepts
        # node
        node_tlens = [len(toks) for tids in node_tids_l for toks in tids]
        max_node_tlen = max(node_tlens)
        node_toks = [pad_sequence_to_length(toks, max_node_tlen, lambda: PAD_idx)
                     for tids in node_tids_l for toks in tids]
        node_toks = th.LongTensor(node_toks)
        node_tlens = th.LongTensor(node_tlens)
        # edge
        edge_tlens = [len(toks) for tids in edge_tids_l for toks in tids]
        max_edge_tlen = max(edge_tlens)
        edge_toks = [pad_sequence_to_length(toks, max_edge_tlen, lambda: PAD_idx)
                     for tids in edge_tids_l for toks in tids]
        edge_toks = th.LongTensor(edge_toks)
        edge_tlens = th.LongTensor(edge_tlens)
        # concept
        batch_num_concepts = [len(cep_tids) for cep_tids in cep_tids_l]
        cep_tlens = [len(toks) for tids in cep_tids_l for toks in tids]
        max_cep_tlen = max(cep_tlens)
        cep_toks = [pad_sequence_to_length(toks, max_cep_tlen, lambda: PAD_idx)
                    for tids in cep_tids_l for toks in tids]
        cep_toks = th.LongTensor(cep_toks)
        cep_tlens = th.LongTensor(cep_tlens)
        return bg, node_toks, node_tlens, edge_toks, edge_tlens, cep_toks, cep_tlens, batch_num_concepts

    @staticmethod
    def sample_neg_concepts(pos_ceps: list, cep_vocab: dict) -> list:
        pass



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
    cg_dev_path = '%s/cg_pairs.dev.txt' % (dataset_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    # analysis overlapping of concept in train and test set
    cg_pairs_train = load_cg_pairs(cg_train_path)
    concepts_train = set([c for ceps in cg_pairs_train.values() for c in ceps])
    concepts_dev = set([c for ceps in load_cg_pairs(cg_dev_path).values() for c in ceps])
    concepts_test = set([c for ceps in load_cg_pairs(cg_test_path).values() for c in ceps])
    concepts_all = concepts_train.union(concepts_dev).union(concepts_test)
    intersc = concepts_train.intersection(concepts_all)
    print('#train_cep=%d, #all_cep=%d. #%d(%.4f all) in train_cep' % (len(concepts_train), len(concepts_all),
                                                                      len(intersc), len(intersc)/len(concepts_all)))
    # analysis whether all concept tokens have shown in train set (both CG and OKG)
    concepts_tok_all = set(tok for cep in concepts_all for tok in cep.split(' '))
    all_tok = set()
    for ent, ceps in cg_pairs_train.items():
        for tok in ent.split(' '):
            all_tok.add(tok)
        for cep in ceps:
            for tok in cep.split(' '):
                all_tok.add(tok)
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    with open(oie_train_path) as fopen:
        for line in fopen:
            """
            try:
                subj, rel, obj = line.strip().split('\t')
            except:
                print(line)
            """
            subj, rel, obj = line.strip().split('\t')
            for tok in ('%s %s %s' % (subj, rel, obj)).split(' '):
                all_tok.add(tok)
    intersc = all_tok.intersection(concepts_tok_all)
    print('#all_tok=%d, #test_cep_tok=%d. #%d(%.4f test) in all_tok' % (len(all_tok), len(concepts_tok_all),
                                                                        len(intersc), len(intersc)/len(concepts_tok_all)))


def analysis_oie_token_existence(dataset_dir: str):
    print('analysis_oie_token_existence for %s' % (dataset_dir))
    # mention candidate pool include all mentions from train, dev, test sets
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    oie_dev_path = '%s/oie_triples.dev.txt' % (dataset_dir)
    oie_test_path = '%s/oie_triples.test.txt' % (dataset_dir)
    oie_triples_train = load_oie_triples(oie_train_path)
    oie_triples_dev = load_oie_triples(oie_dev_path)
    oie_triples_test = load_oie_triples(oie_test_path)
    mentions_train = set(m for (subj, rel, obj) in oie_triples_train for m in [subj, obj])
    mentions_dev = set(m for (subj, rel, obj) in oie_triples_dev for m in [subj, obj])
    mentions_test = set(m for (subj, rel, obj) in oie_triples_test for m in [subj, obj])
    mentions_all = mentions_train.union(mentions_dev).union(mentions_test)
    intersc = mentions_train.intersection(mentions_all)
    print('#train_ment=%d, #all_ment=%d. #%d(%.4f all) in train_ment' % (len(mentions_train), len(mentions_all), len(intersc), len(intersc)/len(mentions_all)))
    mention_tok_all = set(tok for m in mentions_all for tok in m.split(' '))
    train_tok_all = set(tok for (subj, rel, obj) in oie_triples_train for tok in (' '.join([subj, rel, obj])).split(' '))
    cg_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cg_pairs_train = load_cg_pairs(cg_train_path)
    for ent, ceps in cg_pairs_train.items():
        train_tok_all.update(ent.split(' '))
        for cep in ceps:
            train_tok_all.update(cep.split(' '))
    intersc = train_tok_all.intersection(mention_tok_all)
    print('#train_tok=%d, #all_ment_tok=%d. #%d(%.4f all) in train_tok' % (len(train_tok_all), len(mention_tok_all), len(intersc), len(intersc)/len(mention_tok_all)))
    # relation analyis: no relation candidate
    # so only examine test vs train
    relations_train = set(rel for (subj, rel, obj) in oie_triples_train)
    relations_test = set(rel for (subj, rel, obj) in oie_triples_test)
    intersc = relations_train.intersection(relations_test)
    print('#train_rel=%d, #test_rel=%d. #%d(%.4f test) in train_rel' % (len(relations_train), len(relations_test), len(intersc), len(intersc)/len(relations_test)))
    relation_tok_test = set(tok for rel in relations_test for tok in rel.split(' '))
    intersc = relation_tok_test.intersection(train_tok_all)
    print('#train_tok=%d, #all_rel_tok=%d. #%d(%.4f all) in train_tok' % (len(train_tok_all), len(relation_tok_test), len(intersc), len(intersc)/len(relation_tok_test)))


def load_oie_triples(fpath: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(fpath) as fopen:
        for line in fopen:
            subj, rel, obj = line.strip().split('\t')
            triples.append((subj, rel, obj))
    return triples


def get_concept_vocab(cg_pairs_train: dict, cg_pairs_dev: dict, cg_pairs_test: dict) -> Dict[str, int]:
    """
    Concepts from train, valid and test set
    """
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
    return concept_vocab


def get_tok_vocab(cg_triples_train: list, oie_triples_train: list) -> Dict[str, int]:
    """
    Tokens only from train set
    """
    tok_vocab = {'<PAD>': PAD_idx, '<UNK>': UNK_idx}
    for subj, rel, obj in oie_triples_train:
        for tok in (' '.join([subj, rel, obj])).split(' '):
            if tok not in tok_vocab:
                tok_vocab[tok] = len(tok_vocab)
    for ent, rel, cep in cg_triples_train:
        for tok in (' '.join([ent, rel, cep])).split(' '):
            if tok not in tok_vocab:
                tok_vocab[tok] = len(tok_vocab)
    # add special toks
    tok_vocab[SELF_LOOP] = len(tok_vocab)
    return tok_vocab


def get_mention_rel_vocabs(oie_triples_train: list, oie_triples_dev: list, oie_triples_test: list) -> Tuple[dict, dict]:
    mention_vocab = {}
    rel_vocab = {}
    for subj, rel, obj in (oie_triples_train + oie_triples_dev + oie_triples_test):
        if subj not in mention_vocab:
            mention_vocab[subj] = len(mention_vocab)
        if obj not in mention_vocab:
            mention_vocab[obj] = len(mention_vocab)
        if rel not in rel_vocab:
            rel_vocab[rel] = len(rel_vocab)
    return mention_vocab, rel_vocab


def prepare_ingredients_transE(dataset_dir: str) -> tuple:
    # Load Concept Graph
    cg_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cg_dev_path = '%s/cg_pairs.dev.txt' % (dataset_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    cg_pairs_train = load_cg_pairs(cg_train_path)
    cg_pairs_dev = load_cg_pairs(cg_dev_path)
    cg_pairs_test = load_cg_pairs(cg_test_path)
    concept_vocab = get_concept_vocab(cg_pairs_train, cg_pairs_dev, cg_pairs_test)
    cg_triples_train = cg_pairs_to_cg_triples(cg_pairs_train)
    # cg_triples_dev = cg_pairs_to_cg_triples(cg_pairs_dev)
    # cg_triples_test = cg_pairs_to_cg_triples(cg_pairs_test)
    # Load Open KG
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    oie_dev_path = '%s/oie_triples.dev.txt' % (dataset_dir)
    oie_test_path = '%s/oie_triples.test.txt' % (dataset_dir)
    oie_triples_train = load_oie_triples(oie_train_path)
    oie_triples_dev = load_oie_triples(oie_dev_path)
    oie_triples_test = load_oie_triples(oie_test_path)
    tok_vocab = get_tok_vocab(cg_triples_train, oie_triples_train)
    mention_vocab, rel_vocab = get_mention_rel_vocabs(oie_triples_train, oie_triples_dev, oie_triples_test)
    all_triple_ids_map = {'h': defaultdict(set),
                          't': defaultdict(set)}  # resources for OLP filtered eval setting
    for (h, r, t) in (oie_triples_train + oie_triples_dev + oie_triples_test):
        all_triple_ids_map['h'][(mention_vocab[h], rel_vocab[r])].add(mention_vocab[t])
        all_triple_ids_map['t'][(mention_vocab[t], rel_vocab[r])].add(mention_vocab[h])
    train_set = CGCOLPTriplesDst(cg_triples_train+oie_triples_train, tok_vocab)
    dev_cg_set = CGCPairsDst(cg_pairs_dev, tok_vocab, concept_vocab)
    dev_oie_set = OLPTriplesDst(oie_triples_dev, tok_vocab, mention_vocab, rel_vocab)
    test_cg_set = CGCPairsDst(cg_pairs_test, tok_vocab, concept_vocab)
    test_oie_set = OLPTriplesDst(oie_triples_test, tok_vocab, mention_vocab, rel_vocab)
    return (train_set, dev_cg_set, dev_oie_set, test_cg_set, test_oie_set,
            tok_vocab, mention_vocab, concept_vocab, rel_vocab, all_triple_ids_map)


def get_concept_tok_tensor(concept_vocab: dict, tok_vocab: dict) -> th.LongTensor:
    concepts = []
    cep_lens = []
    for cep in concept_vocab.keys():
        cep_num = [tok_vocab.get(_, UNK_idx) for _ in cep.split(' ')]
        concepts.append(cep_num)
        cep_lens.append(len(cep_num))
    max_len = max(cep_lens)
    concepts = [pad_sequence_to_length(_, max_len, lambda: PAD_idx) for _ in concepts]
    return th.LongTensor(concepts), th.LongTensor(cep_lens)


def prepare_ingredients_TaxoRelGraph(dataset_dir: str) -> tuple:
    # Load Concept Graph
    cg_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cg_dev_path = '%s/cg_pairs.dev.txt' % (dataset_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    cg_pairs_train = load_cg_pairs(cg_train_path)
    cg_pairs_dev = load_cg_pairs(cg_dev_path)
    cg_pairs_test = load_cg_pairs(cg_test_path)
    concept_vocab = get_concept_vocab(cg_pairs_train, cg_pairs_dev, cg_pairs_test)
    # Load Open KG
    oie_train_path = '%s/oie_triples.train.txt' % (dataset_dir)
    oie_dev_path = '%s/oie_triples.dev.txt' % (dataset_dir)
    oie_test_path = '%s/oie_triples.test.txt' % (dataset_dir)
    oie_triples_train = load_oie_triples(oie_train_path)
    oie_triples_dev = load_oie_triples(oie_dev_path)
    oie_triples_test = load_oie_triples(oie_test_path)
    tok_vocab = get_tok_vocab(cg_pairs_to_cg_triples(cg_pairs_train), oie_triples_train)
    mention_vocab, rel_vocab = get_mention_rel_vocabs(oie_triples_train, oie_triples_dev, oie_triples_test)
    all_triple_ids_map = {'h': defaultdict(set),
                          't': defaultdict(set)}  # resources for OLP filtered eval setting
    for (h, r, t) in (oie_triples_train + oie_triples_dev + oie_triples_test):
        all_triple_ids_map['h'][(mention_vocab[h], rel_vocab[r])].add(mention_vocab[t])
        all_triple_ids_map['t'][(mention_vocab[t], rel_vocab[r])].add(mention_vocab[h])
    # create dataset
    train_CGC_set = CGCEgoGraphDst(cg_pairs_train, oie_triples_train, tok_vocab)
    dev_CGC_set = CGCEgoGraphDst(cg_pairs_dev, oie_triples_dev, tok_vocab)
    test_CGC_set = CGCEgoGraphDst(cg_pairs_test, oie_triples_test, tok_vocab)
    return (train_CGC_set, dev_CGC_set, test_CGC_set,
            tok_vocab, mention_vocab, concept_vocab, rel_vocab, all_triple_ids_map)


if __name__ == '__main__':
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/MSCG-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-OPIEC'
    dataset_dir = 'data/CGC-OLP-BENCH/MSCG-OPIEC'
    # analysis_oie_token_existence(dataset_dir)
    # analysis_concept_token_existence(dataset_dir)

    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    # train_set, tok_vocab, mention_vocab, concept_vocab = prepare_ingredients_transE(dataset_dir)
    # train_iter = data.DataLoader(train_set, collate_fn=collate_fn_triples, batch_size=4, shuffle=True)

"""
Data Loaders for HAKEGCN
Author: Jiaying Lu
Create Date: Sep 8, 2021
"""

from collections import defaultdict
from typing import Tuple, Dict, List
from enum import Enum
import random

import tqdm
import numpy as np
import torch as th
from torch.utils import data
from allennlp.common.util import pad_sequence_to_length
import networkx as nx
import dgl

from .data_loader import BatchType, TAXO_EDGE, CompGCNOLPTripleDst, get_mention_rel_vocabs
from .data_loader import get_rv_for_u, load_cg_pairs, cg_pairs_to_cg_triples, get_concept_vocab
from .data_loader import load_oie_triples, get_tok_vocab, CompGCNCGCTripleDst


class HAKEGCNDst(data.Dataset):
    def __init__(self, triples: List[Tuple[str, str, str]], all_phrase2id: dict,
                 pid_graph_dict: dict, neg_method: str, neg_size: int, train_phraseids: np.array,
                 batch_type: BatchType, gsample_method: str, gsample_prob: float, ):
        """
        graph sample only for positive triple(subj,obj).
        """
        self.triples = []
        self.pid_graph_dict = pid_graph_dict
        self.neg_size = neg_size
        self.neg_method = neg_method
        self.train_phraseids = train_phraseids
        self.batch_type = batch_type
        self.gsample_method = gsample_method
        self.gsample_prob = gsample_prob
        for s, r, o in tqdm.tqdm(triples):
            sid = all_phrase2id[s]
            rid = all_phrase2id[r]
            oid = all_phrase2id[o]
            self.triples.append((sid, rid, oid))
        self.len = len(self.triples)
        # hr,tr_map for valid negative sampling
        # hr,tr_freq for calculating sampling weight
        self.hr_map, self.tr_map, self.hr_freq, self.tr_freq = self.two_tuple_count()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        """
        do graph sampling for subj, obj but not negative samples
        delete current triple from subj/obj ego graphs.
        """
        # get positive triple with ego graphs
        pos_triple = self.triples[idx]
        h_pid, r_pid, t_pid = pos_triple
        h_graph = self.pid_graph_dict[h_pid]
        h_graph = remove_edge_from_dglgraph(h_graph, pos_triple)
        t_graph = self.pid_graph_dict[t_pid]
        t_graph = remove_edge_from_dglgraph(t_graph, pos_triple)
        # TODO: graph sampling
        """
        # conduct graph sampling if needed
        if self.gsample_prob <= 0.0:
            return self.triples[idx]
        elif self.gsample_method == 'node_sampling':
            ((s, r, o), subj_g, subj_node_toks, subj_node_tlens, subj_edge_toks,
             subj_edge_tlens, r_toks, r_tlen, obj_g, obj_node_toks, obj_node_tlens,
             obj_edge_toks, obj_edge_tlens) = self.triples[idx]
            while True:
                subj_nmask = th.rand(subj_g.num_nodes()) >= self.gsample_prob
                subj_nmask[0] = True     # always keep ego node
                subj_g = dgl.node_subgraph(subj_g, subj_nmask)
                subj_node_toks = subj_node_toks[subj_nmask]
                subj_node_tlens = subj_node_tlens[subj_nmask]
                subj_edge_toks = subj_edge_toks[subj_g.edata[dgl.EID]]
                subj_edge_tlens = subj_edge_tlens[subj_g.edata[dgl.EID]]
                if subj_edge_toks.size(0) > 0:
                    break
            return ((s, r, o), subj_g, subj_node_toks, subj_node_tlens, subj_edge_toks,
                    subj_edge_tlens, r_toks, r_tlen, obj_g, obj_node_toks, obj_node_tlens,
                    obj_edge_toks, obj_edge_tlens)
        elif self.gsample_method == 'edeg_sampling':
            pass
        else:
            print('INvalid gsample method for HAKEGCNDst')
            exit(-1)
        """
        # negative sampling
        # subsampling_weight, inspired by word2vec
        subsampling_weight = self.hr_freq[(h_pid, r_pid)] + self.tr_freq[(t_pid, r_pid)]
        subsampling_weight = th.sqrt(1 / th.Tensor([subsampling_weight]))  # scalar
        neg_triples = []
        neg_size = 0
        while neg_size < self.neg_size:
            if self.neg_method == 'self_adversarial':
                neg_triples_tmp = np.random.choice(self.train_phraseids, size=self.neg_size*2, replace=False)
            # TODO: implement other neg_method
            if self.batch_type == BatchType.HEAD_BATCH:
                mask = np.in1d(
                    neg_triples_tmp,
                    self.tr_map[(t_pid, r_pid)],
                    assume_unique=True,
                    invert=True
                )
            elif self.batch_type == BatchType.TAIL_BATCH:
                mask = np.in1d(
                    neg_triples_tmp,
                    self.hr_map[(h_pid, r_pid)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Invalid BatchType: {}'.format(self.batch_type))
            neg_triples_tmp = neg_triples_tmp[mask]
            neg_triples.append(neg_triples_tmp)
            neg_size += neg_triples_tmp.size
        neg_triples = np.concatenate(neg_triples)[:self.neg_size]
        pos_triple = th.LongTensor(pos_triple)
        neg_triples = th.from_numpy(neg_triples)
        return pos_triple, neg_triples, subsampling_weight, self.batch_type, h_graph, t_graph

    @staticmethod
    def collate_fn(data: list) -> tuple:
        pos_samples, neg_samples, samp_weights, batch_types, h_graphs, t_graphs = zip(*data)
        pos_samples = th.stack(pos_samples, dim=0)   # (B, 3)
        neg_samples = th.stack(neg_samples, dim=0)   # (B, neg_s)
        samp_weights = th.cat(samp_weights, dim=0)
        batch_type = batch_types[0]      # single item
        h_graphs = dgl.batch(h_graphs)   # (B, )
        t_graphs = dgl.batch(t_graphs)   # (B, )
        return (pos_samples, neg_samples, samp_weights, batch_type, h_graphs, t_graphs)

    def two_tuple_count(self) -> tuple:
        """
        Return two dict:
        dict({(h, r): [t1, t2, ...]}),
        dict({(t, r): [h1, h2, ...]}),
        """
        hr_map = {}
        hr_freq = {}
        tr_map = {}
        tr_freq = {}

        init_cnt = 3
        for head, rel, tail in self.triples:
            if (head, rel) not in hr_map.keys():
                hr_map[(head, rel)] = set()

            if (tail, rel) not in tr_map.keys():
                tr_map[(tail, rel)] = set()

            if (head, rel) not in hr_freq.keys():
                hr_freq[(head, rel)] = init_cnt

            if (tail, rel) not in tr_freq.keys():
                tr_freq[(tail, rel)] = init_cnt

            hr_map[(head, rel)].add(tail)
            tr_map[(tail, rel)].add(head)
            hr_freq[(head, rel)] += 1
            tr_freq[(tail, rel)] += 1

        for key in tr_map.keys():
            tr_map[key] = np.array(list(tr_map[key]))

        for key in hr_map.keys():
            hr_map[key] = np.array(list(hr_map[key]))

        return hr_map, tr_map, hr_freq, tr_freq


def remove_edge_from_dglgraph(g: dgl.graph, triple: Tuple[int, int, int]) -> dgl.graph:
    srcs, dsts, eids = g.edges(form='all')
    for i in range(eids.size(0)):
        src_pid = g.ndata['phrid'][srcs[i]].item()
        edge_pid = g.edata['phrid'][eids[i]].item()
        dst_pid = g.ndata['phrid'][dsts[i]].item()
        if triple == (src_pid, edge_pid, dst_pid):
            g_new = dgl.remove_edges(g, eids[i])
            return g_new
    # assume triple must exist in g
    # TODO: check if there is a bug
    return g


def construct_ego_graphs_for_all_mentions(cg_pairs_train: dict, oie_triples_train: list,
                                          cg_pairs_dev: dict, cg_pairs_test: dict,
                                          oie_triples_dev: list, oie_triples_test: list) -> Tuple[dict, dict]:
    """
    Pre-compute graphs for entities/concepts, subjects, objects;
    including train, dev, test.
    Ego graph edges only from train set.
    """
    all_phrase2id = {TAXO_EDGE: 0}  # mentions, relations, concepts
    nonrel_pool = set()
    for ent, ceps in {**cg_pairs_train, **cg_pairs_dev, **cg_pairs_test}.items():
        nonrel_pool.add(ent)
        if ent not in all_phrase2id:
            all_phrase2id[ent] = len(all_phrase2id)
        for cep in ceps:
            nonrel_pool.add(cep)
            if cep not in all_phrase2id:
                all_phrase2id[cep] = len(all_phrase2id)
    for (s, r, o) in (oie_triples_train + oie_triples_dev + oie_triples_test):
        nonrel_pool.add(s)
        nonrel_pool.add(o)
        if s not in all_phrase2id:
            all_phrase2id[s] = len(all_phrase2id)
        if r not in all_phrase2id:
            all_phrase2id[r] = len(all_phrase2id)
        if o not in all_phrase2id:
            all_phrase2id[o] = len(all_phrase2id)
    # resources for building ego graph
    u_rv_dict, v_ru_dict = get_rv_for_u(cg_pairs_train, oie_triples_train)
    result = {}   # store pid - egograph
    for ent, pid in tqdm.tqdm(all_phrase2id.items()):
        # skip phrase that only serve as relations
        if ent not in nonrel_pool:
            continue
        nxg = nx.DiGraph()
        # one-hop neighbours
        edges = set()
        one_hop_neighs = set()
        for (r, v) in u_rv_dict[ent]:
            edges.add((ent, r, v))
            one_hop_neighs.add(v)
        for (r, u) in v_ru_dict[ent]:
            edges.add((u, r, ent))
            one_hop_neighs.add(u)
        # two-hop neighbours
        for n in one_hop_neighs:
            for (r, v) in u_rv_dict[n]:
                edges.add((n, r, v))
            for (r, u) in v_ru_dict[n]:
                edges.add((u, r, n))
        # build graph
        if len(edges) > 0:
            for (s, r, o) in edges:
                nxg.add_edge(s, o, rel=r)
        else:
            nxg.add_node(ent)
        node_id_map = {ent: 0}  # {phrase: nid}
        for n in nxg.nodes:
            if n not in node_id_map:
                node_id_map[n] = len(node_id_map)
        u_l = []
        v_l = []
        edge_pids = []
        for (u, v, rel) in nxg.edges.data('rel'):
            u_l.append(node_id_map[u])
            v_l.append(node_id_map[v])
            edge_pids.append(all_phrase2id[rel])
        G = dgl.graph((u_l, v_l))
        if len(u_l) <= 0:
            G = dgl.add_nodes(G, 1)
        node_pids = [[] for _ in range(len(node_id_map))]
        for node, nid in node_id_map.items():
            node_pids[nid] = all_phrase2id[node]
        G.ndata['phrid'] = th.LongTensor(node_pids)
        G.edata['phrid'] = th.LongTensor(edge_pids)
        result[pid] = G
    return result, all_phrase2id


def prepare_ingredients_HAKEGCN(dataset_dir: str, neg_method: str, neg_size: int,
                                gsample_method: str, gsample_prob: float) -> tuple:
    """
    Proposed Model that extend HAKE with GCN
    Args:
    negative sampling strategy
        neg_size: int
        neg_method: str, ['self_adversarial']
    graph sampling strategy
        gsample_method: str, ['node_sampling', 'edge_sampling']
        gsample_prob: 0 means no sampling; otherwise means prob to drop node in graph
    """
    # Load Concept Graph
    cg_train_path = '%s/cg_pairs.train.txt' % (dataset_dir)
    cg_dev_path = '%s/cg_pairs.dev.txt' % (dataset_dir)
    cg_test_path = '%s/cg_pairs.test.txt' % (dataset_dir)
    cg_pairs_train = load_cg_pairs(cg_train_path)
    cg_triples_train = cg_pairs_to_cg_triples(cg_pairs_train)
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
    tok_vocab = get_tok_vocab(cg_triples_train, oie_triples_train)
    # resources
    # str-egograph dict that pre-compute egographs for ent/concept/subj/obj in train,dev,test
    pid_graph_dict, all_phrase2id = construct_ego_graphs_for_all_mentions(cg_pairs_train, oie_triples_train,
                                                                          cg_pairs_dev, cg_pairs_test,
                                                                          oie_triples_dev, oie_triples_test)
    print('construct_ego_graphs_for_all_mentions Done. #ego_graphs=%d' % (len(pid_graph_dict)))
    # pool for training negative sampling, using pid from all_phrase2id
    train_phraseids = set()
    for ent, ceps in cg_pairs_train.items():
        ent_pid = all_phrase2id[ent]
        train_phraseids.add(ent_pid)
        for cep in ceps:
            cep_pid = all_phrase2id[cep]
            train_phraseids.add(cep_pid)
    for subj, rel, obj in oie_triples_train:
        subj_pid = all_phrase2id[subj]
        train_phraseids.add(subj_pid)
        obj_pid = all_phrase2id[obj]
        train_phraseids.add(obj_pid)
    train_phraseids = np.array(list(train_phraseids))
    train_set_head_batch = HAKEGCNDst(cg_triples_train+oie_triples_train, all_phrase2id, pid_graph_dict,
                                      neg_method, neg_size, train_phraseids, BatchType.HEAD_BATCH,
                                      gsample_method, gsample_prob)
    train_set_tail_batch = HAKEGCNDst(cg_triples_train+oie_triples_train, all_phrase2id, pid_graph_dict,
                                      neg_method, neg_size, train_phraseids, BatchType.TAIL_BATCH,
                                      gsample_method, gsample_prob)
    dev_cg_set = CompGCNCGCTripleDst(cg_pairs_dev, all_phrase2id, all_phrase2id, concept_vocab)
    test_cg_set = CompGCNCGCTripleDst(cg_pairs_test, all_phrase2id, all_phrase2id, concept_vocab)
    # resources for olp test
    olp_ment_vocab, olp_rel_vocab = get_mention_rel_vocabs(oie_triples_train, oie_triples_dev, oie_triples_test)
    dev_olp_set = CompGCNOLPTripleDst(oie_triples_dev, olp_ment_vocab, all_phrase2id)
    test_olp_set = CompGCNOLPTripleDst(oie_triples_test, olp_ment_vocab, all_phrase2id)
    all_triple_ids_map = {'h': defaultdict(set),
                          't': defaultdict(set)}  # resources for OLP filtered eval setting
    # subj/obj use ment_vocab, rel use all_phrase2id; align with olp_set
    for (h, r, t) in (oie_triples_train + oie_triples_dev + oie_triples_test):
        all_triple_ids_map['h'][(olp_ment_vocab[h], all_phrase2id[r])].add(olp_ment_vocab[t])
        all_triple_ids_map['t'][(olp_ment_vocab[t], all_phrase2id[r])].add(olp_ment_vocab[h])
    return (train_set_head_batch, train_set_tail_batch, dev_cg_set, test_cg_set,
            dev_olp_set, test_olp_set, all_triple_ids_map, olp_ment_vocab,
            tok_vocab, concept_vocab, all_phrase2id, pid_graph_dict)


if __name__ == '__main__':
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    train_set_head_batch = prepare_ingredients_HAKEGCN(dataset_dir, 'self_adversarial', 8, 'node_sampling', 0.2)
    _ = train_set_head_batch[0]

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
TAXO_EDGE = "IsA"


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
    """
    Ego graph: 2-hop ego net of central entity,
               surrounding by 2-hop relational triples
    """
    def __init__(self, cg_pairs: Dict[str, set], oie_triples: List[Tuple[str, str, str]],
                 tok_vocab: dict, cep_vocab: dict, max_len: int):
        self.graphs = []
        subj_oie_dict = defaultdict(set)
        obj_oie_dict = defaultdict(set)
        for subj, rel, obj in oie_triples:
            subj_oie_dict[subj].add((rel, obj))
            obj_oie_dict[obj].add((rel, subj))
        # build one graph for each ent
        eg_nodes = []
        eg_edges = []
        for ent, ceps in tqdm.tqdm(cg_pairs.items()):
            ego_graph = CGCEgoGraphDst.create_2hop_ego_graph(ent, subj_oie_dict, obj_oie_dict)
            eg_nodes.append(ego_graph.number_of_nodes())
            eg_edges.append(ego_graph.number_of_edges())
            # cep_tids = [[tok_vocab.get(t, UNK_idx) for t in c.split(' ')] for c in ceps]
            cep_vec = [0.0 for idx in range(len(cep_vocab))]
            for c in ceps:
                cep_vec[cep_vocab[c]] = 1.0
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
            node_toks, node_tlens = CGCEgoGraphDst.tids_list_to_tensor(node_tids, max_len)
            edge_toks, edge_tlens = CGCEgoGraphDst.tids_list_to_tensor(edge_tids, max_len)
            # self.graphs.append((g, node_tids, edge_tids, cep_vec))
            self.graphs.append((g, node_toks, node_tlens, edge_toks, edge_tlens, cep_vec))
        self.avg_node_cnt = sum(eg_nodes) / len(eg_nodes)
        self.avg_edge_cnt = sum(eg_edges) / len(eg_edges)

    @staticmethod
    def create_2hop_ego_graph(ent: str, subj_oie_dict: dict, obj_oie_dict: dict) -> nx.DiGraph():
        DG = nx.DiGraph()
        edges = set()
        # one-hop neighbours
        one_hop_neighs = set()
        if ent in subj_oie_dict:
            for (r, o) in subj_oie_dict[ent]:
                one_hop_neighs.add(o)
                edges.add((ent, r, o))
        if ent in obj_oie_dict:
            for (r, s) in obj_oie_dict[ent]:
                one_hop_neighs.add(s)
                edges.add((s, r, ent))
        # two-hop neighbours
        for neigh in one_hop_neighs:
            if neigh in subj_oie_dict:
                for (r, o) in subj_oie_dict[neigh]:
                    edges.add((neigh, r, o))
            if neigh in obj_oie_dict:
                for (r, s) in obj_oie_dict[neigh]:
                    edges.add((s, r, neigh))
        if len(edges) > 0:
            for (s, r, o) in edges:
                DG.add_edge(s, o, rel=r)
        else:
            DG.add_node(ent)
        # add self loop
        for n in DG:
            DG.add_edge(n, n, rel=SELF_LOOP)
        return DG

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        return self.graphs[idx]

    @staticmethod
    def tids_list_to_tensor(tids: list, max_len: int) -> Tuple[th.tensor, list]:
        # tids: 2D list
        tlens = [min(len(tid), max_len) for tid in tids]
        tids_tensor = [pad_sequence_to_length(tid, max_len, lambda: PAD_idx)
                       for tid in tids]
        return th.LongTensor(tids_tensor), tlens

    @staticmethod
    def collate_fn(data: list) -> tuple:
        g_l, node_toks_l, node_tlens_l, edge_toks_l, edge_tlens_l, cep_vec_l = zip(*data)
        bg = dgl.batch(g_l)
        # 1D list for all nodes, edges, concepts
        # node
        node_toks = th.cat(node_toks_l, 0)   # (n_cnt, max_l)
        node_tlens = th.LongTensor([tl for tlens in node_tlens_l for tl in tlens])  # (n_cnt, )
        # edge
        edge_toks = th.cat(edge_toks_l, 0)   # (n_cnt, max_l)
        edge_tlens = th.LongTensor([tl for tlens in edge_tlens_l for tl in tlens])  # (e_cnt, )
        # concept as target vector
        cep_vec_l = th.FloatTensor(cep_vec_l)   # (B, cep_cnt)
        return bg, node_toks, node_tlens, edge_toks, edge_tlens, cep_vec_l


class OLPEgoGraphDst(data.Dataset):
    """
    Ego graph: 2-hop ego net of central subj, obj,
               surrounding by 2-hop taxonomy entities/concepts
    """
    def __init__(self, cg_pairs: Dict[str, set], oie_triples: List[Tuple[str, str, str]],
                 tok_vocab: dict, mention_vocab: dict, rel_vocab: dict, max_len: int,
                 sample_2hop_eg: bool = True):
        self.graphs = []
        eg_nodes = []
        eg_edges = []
        reverse_cg_pairs = defaultdict(set)
        for c, ps in cg_pairs.items():
            for p in ps:
                reverse_cg_pairs[p].add(c)
        for subj, rel, obj in tqdm.tqdm(oie_triples):
            if sample_2hop_eg:
                subj_eg = OLPEgoGraphDst.create_2hop_ego_graph(subj, cg_pairs, reverse_cg_pairs)
            else:
                subj_eg = OLPEgoGraphDst.create_1hop_ego_graph(subj, cg_pairs, reverse_cg_pairs)
            eg_nodes.append(subj_eg.number_of_nodes())
            eg_edges.append(subj_eg.number_of_edges())
            subj_g, subj_node_tids = OLPEgoGraphDst.networkx_to_dgl_graph(subj_eg, subj, tok_vocab)
            subj_node_toks, subj_node_tlens = OLPEgoGraphDst.tids_list_to_tensor(subj_node_tids, max_len)
            if sample_2hop_eg:
                obj_eg = OLPEgoGraphDst.create_2hop_ego_graph(obj, cg_pairs, reverse_cg_pairs)
            else:
                obj_eg = OLPEgoGraphDst.create_1hop_ego_graph(obj, cg_pairs, reverse_cg_pairs)
            eg_nodes.append(obj_eg.number_of_nodes())
            eg_edges.append(obj_eg.number_of_edges())
            obj_g, obj_node_tids = OLPEgoGraphDst.networkx_to_dgl_graph(obj_eg, obj, tok_vocab)
            obj_node_toks, obj_node_tlens = OLPEgoGraphDst.tids_list_to_tensor(obj_node_tids, max_len)
            rel_tids = [tok_vocab.get(t, UNK_idx) for t in rel.split(' ')]
            rel_toks = th.LongTensor(pad_sequence_to_length(rel_tids, max_len, lambda: PAD_idx))
            rel_tlen = len(rel_tids)
            triple = (mention_vocab[subj], rel_vocab[rel], mention_vocab[obj])
            self.graphs.append((subj_g, subj_node_toks, subj_node_tlens, rel_toks, rel_tlen,
                                obj_g, obj_node_toks, obj_node_tlens, triple))
        # print('OLP EgoGraph avg #node=%.2f, #edge=%.2f' % (sum(eg_nodes)/len(eg_nodes), sum(eg_edges)/len(eg_edges)))
        self.avg_node_cnt = sum(eg_nodes) / len(eg_nodes)
        self.avg_edge_cnt = sum(eg_edges) / len(eg_edges)

    @staticmethod
    def create_1hop_ego_graph(ent: str, cg_pairs: dict, reverse_cg_pairs: dict) -> nx.DiGraph():
        DG = nx.DiGraph()
        edges = set()
        # one-hop neighbours
        one_hop_neighs = set()
        if ent in cg_pairs:
            for p in cg_pairs[ent]:
                one_hop_neighs.add(p)
                edges.add((ent, p))
        if ent in reverse_cg_pairs:
            for c in reverse_cg_pairs[ent]:
                one_hop_neighs.add(c)
                edges.add((c, ent))
        # build graph
        if len(edges) > 0:
            DG.add_edges_from(edges)
        else:
            DG.add_node(ent)
        # add self loops
        for n in DG:
            DG.add_edge(n, n)
        return DG

    @staticmethod
    def create_2hop_ego_graph(ent: str, cg_pairs: dict, reverse_cg_pairs: dict) -> nx.DiGraph():
        DG = nx.DiGraph()
        edges = set()
        # one-hop neighbours
        one_hop_neighs = set()
        if ent in cg_pairs:
            for p in cg_pairs[ent]:
                one_hop_neighs.add(p)
                edges.add((ent, p))
        if ent in reverse_cg_pairs:
            for c in reverse_cg_pairs[ent]:
                one_hop_neighs.add(c)
                edges.add((c, ent))
        # two-hop neighbours
        for neigh in one_hop_neighs:
            if neigh in cg_pairs:
                for p in cg_pairs[neigh]:
                    edges.add((neigh, p))
            if neigh in reverse_cg_pairs:
                for c in reverse_cg_pairs[neigh]:
                    edges.add((c, neigh))
        # build graph
        if len(edges) > 0:
            DG.add_edges_from(edges)
        else:
            DG.add_node(ent)
        # add self loops
        for n in DG:
            DG.add_edge(n, n)
        return DG

    @staticmethod
    def networkx_to_dgl_graph(eg: nx.DiGraph, ego_ent: str, tok_vocab: dict) -> Tuple[dgl.graph, List[List[str]]]:
        node_id_map = {ego_ent: 0}  # {mention: nid}
        for n in eg.nodes:
            if n not in node_id_map:
                node_id_map[n] = len(node_id_map)
        u_l = []
        v_l = []
        for (u, v) in eg.edges:
            u_l.append(node_id_map[u])
            v_l.append(node_id_map[v])
        u_l = th.tensor(u_l)
        v_l = th.tensor(v_l)
        g = dgl.graph((u_l, v_l))
        node_tids = [[] for _ in range(len(node_id_map))]
        for ent, nid in node_id_map.items():
            node_tids[nid] = [tok_vocab.get(t, UNK_idx) for t in ent.split(' ')]
        return g, node_tids

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        return self.graphs[idx]

    @staticmethod
    def tids_list_to_tensor(tids: list, max_len: int) -> Tuple[th.tensor, list]:
        # tids: 2D list
        tlens = [min(len(tid), max_len) for tid in tids]
        tids_tensor = [pad_sequence_to_length(tid, max_len, lambda: PAD_idx)
                       for tid in tids]
        return th.LongTensor(tids_tensor), tlens

    @staticmethod
    def collate_fn(data: list) -> tuple:
        subj_g_l, subj_node_toks_l, subj_node_tlens_l, rel_toks_l, rel_tlen_l,\
            obj_g_l, obj_node_toks_l, obj_node_tlens_l, triples = zip(*data)
        subj_bg = dgl.batch(subj_g_l)
        subj_node_toks = th.cat(subj_node_toks_l, 0)  # (n_cnt, max_l)
        subj_node_tlens = th.LongTensor([tl for tlens in subj_node_tlens_l for tl in tlens])  # (n_cnt, )
        rel_toks = th.stack(rel_toks_l, 0)     # (B, max_l)
        rel_tlens = th.LongTensor(rel_tlen_l)  # (B, )
        obj_bg = dgl.batch(obj_g_l)
        obj_node_toks = th.cat(obj_node_toks_l, 0)  # (n_cnt, max_l)
        obj_node_tlens = th.LongTensor([tl for tlens in obj_node_tlens_l for tl in tlens])  # (n_cnt, )
        return (subj_bg, subj_node_toks, subj_node_tlens, rel_toks, rel_tlens,
                obj_bg, obj_node_toks, obj_node_tlens, triples)


class CGCOLPGraphTrainDst(data.Dataset):
    """
    One big graph that contains both CGC and OLP info
    """
    def __init__(self, cg_pairs: Dict[str, set], oie_triples: List[Tuple[str, str, str]],
                 tok_vocab: dict, node_vocab: dict, edge_vocab: dict):
        g = dgl.graph((th.tensor([0]), th.tensor([0])))
        g = dgl.remove_edges(g, th.tensor([0, 0]))   # graph with one node
        g = dgl.add_nodes(g, len(node_vocab)-1)
        # add taxo edges
        reverse_cg_pairs = defaultdict(set)
        for c, ps in tqdm.tqdm(cg_pairs.items()):
            for p in ps:
                reverse_cg_pairs[p].add(c)
            c_nid = node_vocab[c]
            ps_nids = [node_vocab[p] for p in ps]
            g = dgl.add_edges(g, th.tensor([c_nid for _ in range(len(ps))]), th.tensor(ps_nids),
                              {'e_vid': th.tensor([[edge_vocab[TAXO_EDGE]] for _ in range(len(ps))])})
        # add non-taxo edges
        all_triples_h = defaultdict(set)
        all_triples_t = defaultdict(set)
        for subj, rel, obj in tqdm.tqdm(oie_triples):
            all_triples_h[(subj, rel)].add(obj)
            all_triples_t[(obj, rel)].add(subj)
            g = dgl.add_edges(g, th.tensor([node_vocab[subj]]), th.tensor([node_vocab[obj]]),
                              {'e_vid': th.tensor([[edge_vocab[rel]]])})
        g = dgl.remove_self_loop(g)   # ensure no self-loops
        self.graph = g
        # prepare triples (head/tail, BCE labels)
        self.triples = []
        for c, ps in tqdm.tqdm(cg_pairs.items()):
            for p in ps:
                tail_BCE_label = th.zeros(len(node_vocab))
                ps_nids = [node_vocab[_] for _ in ps]
                tail_BCE_label[ps_nids] = 1.0
                head_BCE_label = th.zeros(len(node_vocab))
                cs_nids = [node_vocab[_] for _ in reverse_cg_pairs[p]]
                head_BCE_label[cs_nids] = 1.0
                hid = node_vocab[c]
                rid = edge_vocab[TAXO_EDGE]
                tid = node_vocab[p]
                self.triples.append((hid, rid, tid, head_BCE_label, tail_BCE_label))
        for subj, rel, obj in tqdm.tqdm(oie_triples):
            tail_BCE_label = th.zeros(len(node_vocab))
            tail_nids = [node_vocab[_] for _ in all_triples_h[(subj, rel)]]
            tail_BCE_label[tail_nids] = 1.0
            head_BCE_label = th.zeros(len(node_vocab))
            head_nids = [node_vocab[_] for _ in all_triples_t[(obj, rel)]]
            head_BCE_label[head_nids] = 1.0
            hid = node_vocab[subj]
            rid = edge_vocab[rel]
            tid = node_vocab[obj]
            self.triples.append((hid, rid, tid, head_BCE_label, tail_BCE_label))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple:
        return self.triples[idx]

    @staticmethod
    def collate_fn(data: list) -> tuple:
        """
        head_BCE_labels: predction on head
        tail_BCE_labels: predction on tail
        """
        hids, rids, tids, head_BCE_labels, tail_BCE_labels = zip(*data)
        hids = th.LongTensor(hids)  # in node_vocab
        rids = th.LongTensor(rids)  # in edge_vocab
        tids = th.LongTensor(tids)  # in node_vocab
        head_BCE_labels = th.stack(head_BCE_labels, 0)  # (B, n_cnt)
        tail_BCE_labels = th.stack(tail_BCE_labels, 0)  # (B, n_cnt)
        return (hids, rids, tids, head_BCE_labels, tail_BCE_labels)


class CompGCNCGCTripleDst(data.Dataset):
    def __init__(self, cg_pairs: Dict[str, set],
                 node_vocab: dict, edge_vocab: dict, concept_vocab: dict):
        self.triples = []
        for c, ps in cg_pairs.items():
            hid = node_vocab[c]
            rid = edge_vocab[TAXO_EDGE]
            cep_ids = [concept_vocab[p] for p in ps]
            self.triples.append((hid, rid, cep_ids))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple:
        return self.triples[idx]

    @staticmethod
    def collate_fn(data: list) -> tuple:
        hids, rids, cep_ids_l = zip(*data)
        hids = th.LongTensor(hids)  # in node_vocab
        rids = th.LongTensor(rids)  # in edge_vocab
        # cep_ids_l varialble length 2d list
        return (hids, rids, cep_ids_l)


class CompGCNOLPTripleDst(data.Dataset):
    def __init__(self, oie_triples: List[Tuple[str, str, str]],
                 mention_vocab: dict, edge_vocab: dict):
        self.triples = []
        for subj, rel, obj in oie_triples:
            sid = mention_vocab[subj]
            rid = edge_vocab[rel]
            oid = mention_vocab[obj]
            self.triples.append((sid, rid, oid))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple:
        return self.triples[idx]

    @staticmethod
    def collate_fn(data: list) -> tuple:
        sids, rids, oids = zip(*data)
        sids = th.LongTensor(sids)  # in mention_vocab
        rids = th.LongTensor(rids)  # in edge_vocab
        oids = th.LongTensor(oids)  # in mention_vocab
        return (sids, rids, oids)


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
            triples.append((ent, TAXO_EDGE, cep))
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


def prepare_ingredients_TaxoRelGraph(dataset_dir: str, phrase_max_len: int, OLP_2hop_egograph: bool = True) -> tuple:
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
    train_CGC_set = CGCEgoGraphDst(cg_pairs_train, oie_triples_train, tok_vocab, concept_vocab, phrase_max_len)
    dev_CGC_set = CGCEgoGraphDst(cg_pairs_dev, oie_triples_train, tok_vocab, concept_vocab, phrase_max_len)
    test_CGC_set = CGCEgoGraphDst(cg_pairs_test, oie_triples_train, tok_vocab, concept_vocab, phrase_max_len)
    train_OLP_set = OLPEgoGraphDst(cg_pairs_train, oie_triples_train, tok_vocab,
                                   mention_vocab, rel_vocab, phrase_max_len, OLP_2hop_egograph)
    dev_OLP_set = OLPEgoGraphDst(cg_pairs_train, oie_triples_dev, tok_vocab,
                                 mention_vocab, rel_vocab, phrase_max_len, OLP_2hop_egograph)
    test_OLP_set = OLPEgoGraphDst(cg_pairs_train, oie_triples_test, tok_vocab,
                                  mention_vocab, rel_vocab, phrase_max_len, OLP_2hop_egograph)
    return (train_CGC_set, dev_CGC_set, test_CGC_set, train_OLP_set, dev_OLP_set, test_OLP_set,
            tok_vocab, mention_vocab, concept_vocab, rel_vocab, all_triple_ids_map)


def prepare_ingredients_CompGCN(dataset_dir: str) -> tuple:
    """
    one single big graph to get node, edge embeddings
    let all candidaites exist in graph. Candidate can be disconnected to other nodes.
    """
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
    mention_vocab, edge_vocab = get_mention_rel_vocabs(oie_triples_train, oie_triples_dev, oie_triples_test)
    edge_vocab[TAXO_EDGE] = len(edge_vocab)   # contains all possible semantic edges
    all_triple_ids_map = {'h': defaultdict(set),
                          't': defaultdict(set)}  # resources for OLP filtered eval setting
    for (h, r, t) in (oie_triples_train + oie_triples_dev + oie_triples_test):
        all_triple_ids_map['h'][(mention_vocab[h], edge_vocab[r])].add(mention_vocab[t])
        all_triple_ids_map['t'][(mention_vocab[t], edge_vocab[r])].add(mention_vocab[h])
    # build datasets
    node_vocab = {}  # contains train/dev/test mentions, concepts
    node_vocab.update(mention_vocab)
    for ent in set(cg_pairs_train.keys()).union(set(cg_pairs_dev.keys())).union(set(cg_pairs_test.keys())):
        if ent not in node_vocab:
            node_vocab[ent] = len(node_vocab)
    for cep in concept_vocab:
        if cep not in node_vocab:
            node_vocab[cep] = len(node_vocab)
    train_set = CGCOLPGraphTrainDst(cg_pairs_train, oie_triples_train, tok_vocab, node_vocab,
                                    edge_vocab)
    dev_CGC_set = CompGCNCGCTripleDst(cg_pairs_dev, node_vocab, edge_vocab, concept_vocab)
    test_CGC_set = CompGCNCGCTripleDst(cg_pairs_test, node_vocab, edge_vocab, concept_vocab)
    dev_OLP_set = CompGCNOLPTripleDst(oie_triples_dev, mention_vocab, edge_vocab)
    test_OLP_set = CompGCNOLPTripleDst(oie_triples_test, mention_vocab, edge_vocab)
    return (train_set, dev_CGC_set, test_CGC_set, dev_OLP_set, test_OLP_set,
            tok_vocab, node_vocab, edge_vocab, mention_vocab, concept_vocab, all_triple_ids_map)


if __name__ == '__main__':
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/MSCG-ReVerb'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    dataset_dir = 'data/CGC-OLP-BENCH/SEMusic-OPIEC'
    dataset_dir = 'data/CGC-OLP-BENCH/MSCG-OPIEC'
    # analysis_oie_token_existence(dataset_dir)
    # analysis_concept_token_existence(dataset_dir)

    dataset_dir = 'data/CGC-OLP-BENCH/SEMedical-OPIEC'
    # train_set, tok_vocab, mention_vocab, concept_vocab = prepare_ingredients_transE(dataset_dir)
    # train_iter = data.DataLoader(train_set, collate_fn=collate_fn_triples, batch_size=4, shuffle=True)
    # prepare_ingredients_TaxoRelGraph(dataset_dir)
    # prepare_ingredients_CompGCN(dataset_dir, 16)

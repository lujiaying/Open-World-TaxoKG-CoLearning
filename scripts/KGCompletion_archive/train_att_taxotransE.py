"""
Training and evaluate Att-Taxo-TransE
Can not apply big sparse adjacency matrix,
as many operations are not multiplications.
Author: Anonymous Siamese
Create Date: Jun 22, 2021
"""

import time
import os
import random
from typing import Tuple

import numpy as np
from scipy import sparse as spsp
import torch as th
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import prepare_ingredients, sample_negative_triples,\
        get_taxo_parents_children, get_normalized_adj_matrix
from model.TransE import cal_metrics
from model.AttTaxoTransE import AttTaxoTransE

# Sacred Setup to keep everything in record
ex = sacred.Experiment('Att-Taxo-TransE')
ex.observers.append(FileStorageObserver("logs/Att-Taxo-TransE"))
ex.observers.append(NeptuneObserver(project_name='jlu/Learn-To-Abstract', source_extensions=['.py']))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'corpus_type': '',     # WN18RR | CN100k
           'checkpoint_dir': '',  # to set
           'dataset_dir': {
               'WN18RR': 'data/WN18RR',
               'CN100k': 'data/CN-100K'
               },
           'epoch': 1000,
           'validate_freq': 10,
           'batch_size': 128,
           'dist_norm': 1,
           'emb_dim': 100,
           'attn_dim': 10,
           'loss_margin': 3.0,
           'optim_lr': 1e-2,
           'optim_wdecay': 1e-4,
           'use_scheduler': True,
           'scheduler_step': 100,
           'scheduler_gamma': 0.5,
            }


def test(model: th.nn.Module, data_loader: DataLoader, ent_count: int, rel_count: int,
         device: th.device, known_triples_map: dict, taxo_dict: dict) -> Tuple[float, float, float, float]:
    hits_1 = 0.0
    hits_3 = 0.0
    hits_10 = 0.0
    mrr = 0.0
    total_cnt = 0.0

    with th.no_grad():
        all_embs = []
        ent_start = 0
        bsize = 128
        while ent_start < ent_count:
            ent_ids = th.arange(ent_start, min(ent_count, ent_start+bsize), device=device)  # (bsize)
            ent_start += bsize
            embs = model._aggregate_over_taxo(ent_ids, taxo_dict)
            all_embs.append(embs)
        all_embs = th.cat(all_embs, dim=0)   # (ent_c, dim)

        ent_ids = th.arange(end=ent_count, device=device)  # ent_c
        for (batch_h, batch_r, batch_t) in data_loader:
            batch_size = batch_h.size(0)
            all_ents = ent_ids.unsqueeze(0).repeat(batch_size, 1)    # B*ent_c
            batch_h, batch_r, batch_t = batch_h.to(device), batch_r.to(device), batch_t.to(device)   # (B, )
            batch_h = batch_h.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            batch_r = batch_r.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            batch_t = batch_t.reshape(-1, 1).repeat(1, ent_count)  # B*ent_c
            # check all possible tails
            triples = th.stack((batch_h, batch_r, all_ents), dim=2).reshape(-1, 3)  # (B*ent_c)*3
            tail_preds = model.predict(triples, all_embs).reshape(batch_size, -1)   # B*ent_c
            # check all possible heads
            triples = th.stack((all_ents, batch_r, batch_t), dim=2).reshape(-1, 3)  # (B*ent_c)*3
            head_preds = model.predict(triples, all_embs).reshape(batch_size, -1)   # B*ent_c
            # get metrics
            batch_h = batch_h[:, 0].unsqueeze(1)   # B*1
            batch_r = batch_r[:, 0].unsqueeze(1)   # B*1
            batch_t = batch_t[:, 0].unsqueeze(1)   # B*1
            b_hits_1, b_hits_3, b_hits_10, b_mrr = cal_metrics(tail_preds, batch_h, batch_r, batch_t,
                                                               is_tail_preds=True, known_triples_map=known_triples_map)
            hits_1 += b_hits_1
            hits_3 += b_hits_3
            hits_10 += b_hits_10
            mrr += b_mrr
            b_hits_1, b_hits_3, b_hits_10, b_mrr = cal_metrics(head_preds, batch_h, batch_r, batch_t,
                                                               is_tail_preds=False, known_triples_map=known_triples_map)
            hits_1 += b_hits_1
            hits_3 += b_hits_3
            hits_10 += b_hits_10
            mrr += b_mrr
            total_cnt += (2 * batch_size)
    return hits_1/total_cnt, hits_3/total_cnt, hits_10/total_cnt, mrr/total_cnt


@ex.automain
def main(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info('[%s] Random seed set to %d' % (time.ctime(), opt['seed']))

    # Sanity check
    if opt['corpus_type'] not in ['WN18RR', 'CN100k']:
        _log.error('corpus_type=%s invalid' % (opt['corpus_type']))
        exit(-1)
    if opt['checkpoint_dir'] == '':
        _log.error('checkpoint_dir=%s invalid' % (opt['checkpoint_dir']))
        exit(-1)
    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])
    # Setup essential vars
    dataset_dir = opt['dataset_dir'][opt['corpus_type']]
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')

    # Load corpus
    train_set, dev_set, test_set, \
        ent_vocab, rel_vocab, train_triples, all_triples_map = prepare_ingredients(dataset_dir, opt['corpus_type'])
    train_iter = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True)
    dev_iter = DataLoader(dev_set, batch_size=opt['batch_size']//4, shuffle=False)
    test_iter = DataLoader(test_set, batch_size=opt['batch_size']//4, shuffle=False)
    _log.info('[%s] Load dataset Done, len=%d,%d,%d' % (time.ctime(),
              len(train_set), len(dev_set), len(test_set)))
    _log.info('corpus=%s, Entity cnt=%d, rel cnt=%d' % (opt['corpus_type'], len(ent_vocab), len(rel_vocab)))
    taxo_dict = get_taxo_parents_children(train_triples, rel_vocab, opt['corpus_type'])
    _log.info('[%s] avg parents each ent=%.2f, avg children=%.2f' % (time.ctime(), sum(len(_) for _ in taxo_dict['p'].values())/len(taxo_dict['p']),
                                                                     sum(len(_) for _ in taxo_dict['c'].values())/len(taxo_dict['c'])))

    # Build model
    model = AttTaxoTransE(len(ent_vocab), len(rel_vocab), norm=opt['dist_norm'],
                          dim=opt['emb_dim'], attn_dim=opt['attn_dim'])
    model = model.to(device)
    criterion = th.nn.MarginRankingLoss(margin=opt['loss_margin'])
    criterion = criterion.to(device)
    # weigh_decay only for non-bias parameters
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    _log.info('Model no_decay param cnt=%d, decay_param cnt=%d' % (len(no_decay), len(decay)))
    params = [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': opt['optim_wdecay']}]
    optimizer = th.optim.AdamW(params, opt['optim_lr'])
    if opt['use_scheduler']:
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=opt['scheduler_step'],
                                                 gamma=opt['scheduler_gamma'])
    _log.info('[%s] Model build Done. Use optim=%s, device=%s' % (time.ctime(), 'AdamW', device))

    best_mrr = 0.0
    for i_epoch in range(opt['epoch']):
        # do train
        model.train()
        train_loss = []
        for i_batch, (batch_h, batch_r, batch_t) in enumerate(train_iter):
            optimizer.zero_grad()
            # batch_h, r, t size=(batch,)
            pos_triples = th.stack((batch_h, batch_r, batch_t), dim=1)  # B*3
            neg_triples = sample_negative_triples(batch_h, batch_r, batch_t, ent_vocab, train_triples)   # B*3
            pos_triples = pos_triples.to(device)
            neg_triples = neg_triples.to(device)
            pos_scores = model(pos_triples, taxo_dict)
            neg_scores = model(neg_triples, taxo_dict)
            target = pos_triples.new_tensor([-1])
            loss = criterion(pos_scores, neg_scores, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        if opt['use_scheduler']:
            scheduler.step()
        avg_loss = sum(train_loss) / len(train_loss)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f' % (time.ctime(), i_epoch, avg_loss))
        # do eval
        if i_epoch % opt['validate_freq'] == 0:
            model.eval()
            hits_1, hits_3, hits_10, mrr = test(model, dev_iter, len(ent_vocab), len(rel_vocab),
                                                device, all_triples_map, taxo_dict)
            _run.log_scalar("dev.hits1", hits_1, i_epoch)
            _run.log_scalar("dev.hits3", hits_3, i_epoch)
            _run.log_scalar("dev.hits10", hits_10, i_epoch)
            _run.log_scalar("dev.mrr", mrr, i_epoch)
            _log.info('[%s] epoch#%d evaluate, hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), i_epoch,
                                                                                        hits_1, hits_3, hits_10, mrr))
            if mrr >= best_mrr:
                best_mrr = mrr
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['corpus_type'])
                th.save(model.state_dict(), save_path)

    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    hits_1, hits_3, hits_10, mrr = test(model, test_iter, len(ent_vocab), len(rel_vocab),
                                        device, all_triples_map, taxo_dict)
    _run.log_scalar("test.hits1", hits_1)
    _run.log_scalar("test.hits3", hits_3)
    _run.log_scalar("test.hits10", hits_10)
    _run.log_scalar("test.mrr", mrr)
    _log.info('[%s] TEST on best model, hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), hits_1, hits_3, hits_10, mrr))

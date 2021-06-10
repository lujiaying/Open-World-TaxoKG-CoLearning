"""
Training and evaluate Taxo-TransE
Author: Jiaying Lu
Create Date: Jun 9, 2021
"""

import time
import os
import random
from typing import Tuple

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver

from model.data_loader import prepare_ingredients, sample_negative_triples, get_taxo_parents_children
from model.TransE import cal_metrics
from model.TaxoTransE import TaxoTransE


# Sacred Setup to keep everything in record
ex = sacred.Experiment('Taxo-TransE')
ex.observers.append(FileStorageObserver("logs/Taxo-TransE"))
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
           'batch_size': 128,
           'dist_norm': 1,
           'emb_dim': 100,
           'epoch': 1000,
           'validate_freq': 10,
           'optim_type': 'Adam',   # Adam | SGD
           'optim_lr': 3e-4,
           'optim_momentum': 0.0,    # for SGD
           'optim_wdecay': 0.5e-4,
           'loss_margin': 1.0,
           'use_scheduler': True,
           'scheduler_step': 100,
           'scheduler_gamma': 0.5,
            }


def test(model: th.nn.Module, data_loader: DataLoader, ent_count: int, device: th.device, known_triples_map: dict) -> Tuple[float, float, float, float]:
    hits_1 = 0.0
    hits_3 = 0.0
    hits_10 = 0.0
    mrr = 0.0
    total_cnt = 0.0

    ent_ids = th.arange(end=ent_count, device=device).unsqueeze(0)
    for (batch_h, batch_r, batch_t) in data_loader:
        batch_size = batch_h.size(0)
        all_ents = ent_ids.repeat(batch_size, 1)    # B*ent_c
        batch_h, batch_r, batch_t = batch_h.to(device), batch_r.to(device), batch_t.to(device)
        batch_h = batch_h.reshape(-1, 1).repeat(1, all_ents.size(1))  # B*ent_c
        batch_r = batch_r.reshape(-1, 1).repeat(1, all_ents.size(1))  # B*ent_c
        batch_t = batch_t.reshape(-1, 1).repeat(1, all_ents.size(1))  # B*ent_c

        # check all possible tails
        triples = th.stack((batch_h, batch_r, all_ents), dim=2).reshape(-1, 3)  # (B*ent_c)*3
        tail_preds = model(triples).reshape(batch_size, -1)   # B*ent_c
        # check all possible heads
        triples = th.stack((all_ents, batch_r, batch_t), dim=2).reshape(-1, 3)  # (B*ent_c)*3
        head_preds = model(triples).reshape(batch_size, -1)   # B*ent_c
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
    _log.info('taxo triples=%d (%.2f of total)' % (len(taxo_dict['p']), len(taxo_dict['p'])/len(train_triples)))

    # Build model
    device = th.device('cuda') if opt['gpu'] else th.device('cpu')
    model = TaxoTransE(len(ent_vocab), len(rel_vocab), taxo_dict, opt['dist_norm'], opt['emb_dim'])
    model = model.to(device)
    criterion = th.nn.MarginRankingLoss(margin=opt['loss_margin'])
    criterion = criterion.to(device)
    if opt['optim_type'] == 'Adam':
        optimizer = th.optim.Adam(model.parameters(), opt['optim_lr'], weight_decay=opt['optim_wdecay'])
    elif opt['optim_type'] == 'SGD':
        optimizer = th.optim.SGD(model.parameters(), opt['optim_lr'], momentum=opt['optim_momentum'], weight_decay=opt['optim_wdecay'])
    else:
        _log.error('opt["optim_type"] =%s not in [Adam, SGD]' % (opt['optim_type']))
        exit(-1)
    if opt['use_scheduler']:
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=opt['scheduler_step'], gamma=opt['scheduler_gamma'])
    _log.info('[%s] Model build Done. Use optim=%s, device=%s' % (time.ctime(), opt['optim_type'], device))

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
            pos_scores = model(pos_triples)
            neg_scores = model(neg_triples)
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
            hits_1, hits_3, hits_10, mrr = test(model, dev_iter, len(ent_vocab), device, all_triples_map)
            _run.log_scalar("dev.hits1", hits_1, i_epoch)
            _run.log_scalar("dev.hits3", hits_3, i_epoch)
            _run.log_scalar("dev.hits10", hits_10, i_epoch)
            _run.log_scalar("dev.mrr", mrr, i_epoch)
            _log.info('[%s] epoch#%d evaluate, hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), i_epoch, hits_1, hits_3, hits_10, mrr))
            if mrr >= best_mrr:
                best_mrr = mrr
                save_path = '%s/exp_%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['corpus_type'])
                th.save(model.state_dict(), save_path)

    # after all epochs, test based on best checkpoint
    checkpoint = th.load(save_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    hits_1, hits_3, hits_10, mrr = test(model, test_iter, len(ent_vocab), device, all_triples_map)
    _run.log_scalar("test.hits1", hits_1)
    _run.log_scalar("test.hits3", hits_3)
    _run.log_scalar("test.hits10", hits_10)
    _run.log_scalar("test.mrr", mrr)
    _log.info('[%s] TEST on best model, hits@1,3,10=%.3f,%.3f,%.3f, mrr=%.3f' % (time.ctime(), hits_1, hits_3, hits_10, mrr))

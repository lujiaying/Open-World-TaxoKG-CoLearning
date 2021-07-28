#!/bin/bash
#SBATCH --job-name=TaxoKG-TransE
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_OpenTransE

# Jun 9
# python -m scripts.train_transE with 'motivation="Param from Nguyen 2018"' \
#     'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/TransE"'\
#     'opt.corpus_type="WN18RR"' 'opt.batch_size=128' \
#     'opt.optim_lr=0.01' 'opt.optim_wdecay=0.0'\
#     'opt.loss_margin=5.0' 'opt.emb_dim=50' \
#     'opt.scheduler_step=50' 'opt.scheduler_gamma=0.65'

# python -m scripts.train_transE with 'motivation="TransE SGD instead of Adam"' \
#     'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/TransE"'\
#     'opt.corpus_type="WN18RR"' 'opt.batch_size=128' \
#     'opt.optim_type=SGD' 'opt.optim_lr=0.01' 'opt.optim_wdecay=0.0'\
#     'opt.loss_margin=2.0' 'opt.emb_dim=20'

# python -m scripts.train_transE with 'motivation="SGD Param from Nguyen 2018"' \
#     'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/TransE"'\
#     'opt.corpus_type="WN18RR"' 'opt.batch_size=128' \
#     'opt.optim_type=SGD' 'opt.optim_lr=1e-2' 'opt.optim_wdecay=0.0' \
#     'opt.loss_margin=5.0' 'opt.emb_dim=50' 'opt.optim_momentum=0.9' \
#     'opt.scheduler_step=100' 'opt.scheduler_gamma=0.65' 'opt.epoch=2000'

# python -m scripts.train_taxotransE with 'motivation="TaxoTransE Nguyen2018"' \
#      'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/TaxoTransE"'\
#      'opt.corpus_type="WN18RR"' 'opt.batch_size=128' \
#      'opt.optim_lr=0.01' 'opt.optim_wdecay=0.0'\
#      'opt.loss_margin=5.0' 'opt.emb_dim=50' \
#      'opt.scheduler_step=50' 'opt.scheduler_gamma=0.65'


# Jun 15
# python -m scripts.train_transE with 'motivation="Param from Nguyen 2018"' \
#     'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/TransE"'\
#     'opt.corpus_type="CN100k"' 'opt.batch_size=128' \
#     'opt.optim_lr=0.01' 'opt.optim_wdecay=0.0'\
#     'opt.loss_margin=5.0' 'opt.emb_dim=100' \
#     'opt.scheduler_step=50' 'opt.scheduler_gamma=0.65'

# Jun 16
# margin search [1.0, 3.0, 5.0]
# python -m scripts.train_taxotransE with 'motivation="GIN epsilon +sage mean-concat"' \
#      'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/TaxoTransE"'\
#      'opt.corpus_type="CN100k"' 'opt.batch_size=256' \
#      'opt.optim_lr=1e-2' 'opt.optim_wdecay=0.0'\
#      'opt.loss_margin=5.0' 'opt.emb_dim=100' \
#      'opt.scheduler_step=100' 'opt.scheduler_gamma=0.5'


# Jun 22
# python -m scripts.train_att_taxotransE with 'motivation="remove r in attn score"' \
#      'opt.gpu=True' 'opt.checkpoint_dir="checkpoints/AttTaxoTransE"'\
#      'opt.corpus_type="WN18RR"' 'opt.optim_lr=1e-2' \
#      'opt.attn_dim=16' 'opt.loss_margin=3.0'

# Jul 21
# python -m scripts.train_opentransE with 'motivation="add grad_norm, batch=512"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"'\
#     'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'
#     'opt.optim_lr=3e-4' 'opt.batch_size=512'
# python -m scripts.train_opentransE with 'motivation="add grad_norm, batch=512"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"'\
#     'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#     'opt.optim_lr=3e-4' 'opt.batch_size=512'
# python -m scripts.train_opentransE with 'motivation="add grad_norm, batch=512"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#     'opt.optim_lr=3e-4' 'opt.batch_size=512'
# python -m scripts.train_opentransE with 'motivation="add grad_norm, batch=512"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#     'opt.optim_lr=3e-4' 'opt.batch_size=512'

# Jul 24
# CUDA_VISIBLE_DEVICES=2 python -m scripts.train_opentransE with 'motivation="GloVe init; weighted sum"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#     'opt.optim_lr=3e-4' 'opt.batch_size=512'\
#     'opt.emb_dim=300' 'opt.pretrain_tok_emb="GloVe"'
# CUDA_VISIBLE_DEVICES=0 python -m scripts.train_opentransE with 'motivation="GloVe init; weighted sum eval"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512'\
#    'opt.emb_dim=300' 'opt.pretrain_tok_emb="GloVe"'
python -m scripts.train_opentransE with 'motivation="ReVerb datasets"'\
   'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"'\
   'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
   'opt.optim_lr=3e-4' 'opt.batch_size=512'

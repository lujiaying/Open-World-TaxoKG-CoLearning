#!/bin/bash
#SBATCH --job-name=TaxoKG
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_OpenHAKE

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
# python -m scripts.train_opentransE with 'motivation="default param"'\
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
# python -m scripts.train_opentransE with 'motivation="ReVerb datasets"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512'
# python -m scripts.train_opentransE with 'motivation="ReVerb datasets"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512'
# python -m scripts.train_opentransE with 'motivation="ReVerb datasets"'\
#    'opt.gpu=True' 'opt.dataset_type="MSCG-ReVerb"'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512'
# python -m scripts.train_opentransE with 'motivation="ReVerb datasets"'\
#    'opt.gpu=True' 'opt.dataset_type="MSCG-OPIEC"'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512'

# Jul 28
# python -m scripts.train_TaxoRelGraph with 'motivation="default with u_sub_e instead of v_sub_e"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/TaxoRelGraph"'
# python -m scripts.train_TaxoRelGraph with 'motivation="MTL, small batch size"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/TaxoRelGraph"' \
#     'opt.CGC_batch_size=16' 'opt.OLP_batch_size=128' 'opt.epoch=1000'
# python -m scripts.train_TaxoRelGraph with 'motivation="MTL, large batch, MLP for cep, rel emb"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/TaxoRelGraph"' 'opt.epoch=1500'

# Jul 31
# python -m scripts.train_TaxoRelGraph with 'motivation="MTL; large batch, MLP, dropout"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"'\
#     'opt.checkpoint_dir="checkpoints/TaxoRelGraph"' 'opt.epoch=1500'
# python -m scripts.train_TaxoRelGraph with 'motivation="MTL; large batch, MLP, dropout"'\
#     'opt.gpu=True' 'opt.dataset_type="MSCG-OPIEC"'\
#     'opt.checkpoint_dir="checkpoints/TaxoRelGraph"' 'opt.epoch=1500'

# Aug 5
#  python -m scripts.train_TaxoRelGraph with 'motivation="OLP 1hop eg"'\
#      'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#      'opt.checkpoint_dir="checkpoints/TaxoRelGraph"' 'opt.epoch=1500'\
#      'opt.OLP_2hop_egograph=False'

# Aug 8
# python -m scripts.train_CompGCN with 'motivation="DistMult; comp opt, data ReRun"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150'
# python -m scripts.train_CompGCN with 'motivation="TransE default param"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'
# python -m scripts.train_CompGCN with 'motivation="DistMult"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150'
python -m scripts.train_CompGCN with 'motivation="DistMult; comp opt, data ReRun"'\
    'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"'\
    'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150'
# python -m scripts.train_CompGCN with 'motivation="DistMult"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150'
# python -m scripts.train_CompGCN with 'motivation="DistMult for big graph"'\
#     'opt.gpu=True' 'opt.dataset_type="MSCG-OPIEC"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150'\
#     'opt.epoch=300' 'opt.batch_size=2048' 'opt.optim_lr=3e-4'
# python -m scripts.train_CompGCN with 'motivation="DistMult for big graph"'\
#     'opt.gpu=True' 'opt.dataset_type="MSCG-ReVerb"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150'\
#     'opt.epoch=300' 'opt.batch_size=2048' 'opt.optim_lr=3e-4'

# Agu 17
# python -m scripts.train_opentransE with 'motivation="DistMult"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#    'opt.model_type="DistMult"' 'opt.emb_dim=100'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=128' 'opt.epoch=300'
# python -m scripts.train_opentransE with 'motivation="DistMult, opt ReRun"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"'\
#    'opt.model_type="DistMult"' 'opt.emb_dim=100'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512' 'opt.epoch=500'
# python -m scripts.train_opentransE with 'motivation="DistMult; comp_opt,dataset ReRun"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"'\
#    'opt.model_type="DistMult"' 'opt.emb_dim=100'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512' 'opt.epoch=500'
# python -m scripts.train_opentransE with 'motivation="DistMult, comp_opt ReRun"'\
#    'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"'\
#    'opt.model_type="DistMult"' 'opt.emb_dim=100'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512' 'opt.epoch=500'
# python -m scripts.train_opentransE with 'motivation="DistMult, comp_opt ReRun"'\
#    'opt.gpu=True' 'opt.dataset_type="MSCG-OPIEC"'\
#    'opt.model_type="DistMult"' 'opt.emb_dim=200'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512' 'opt.epoch=500'
# python -m scripts.train_opentransE with 'motivation="DistMult, comp_opt ReRun"'\
#    'opt.gpu=True' 'opt.dataset_type="MSCG-ReVerb"'\
#    'opt.model_type="DistMult"' 'opt.emb_dim=200'\
#    'opt.checkpoint_dir="checkpoints/OpenTransE"' 'opt.loss_margin=1.0'\
#    'opt.optim_lr=3e-4' 'opt.batch_size=512' 'opt.epoch=500'

# Aug 25
# python -m scripts.train_openHAKE with 'motivation="OPIEC Re-Run"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"' 'opt.epoch=500'
# python -m scripts.train_openHAKE with 'motivation="OPIEC Re-Run"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"' 'opt.epoch=500'
# python -m scripts.train_openHAKE with 'motivation="trial"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"' 'opt.epoch=550'
# python -m scripts.train_openHAKE with 'motivation="trial"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"' 'opt.epoch=550'
python -m scripts.train_openHAKE with 'motivation="OPIEC Re-Run"'\
    'opt.gpu=True' 'opt.dataset_type="MSCG-OPIEC"' 'opt.epoch=200'
# python -m scripts.train_openHAKE with 'motivation="trial"'\
#     'opt.gpu=True' 'opt.dataset_type="MSCG-ReVerb"' 'opt.epoch=200'

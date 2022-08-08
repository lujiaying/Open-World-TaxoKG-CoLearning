#!/bin/bash
#SBATCH --job-name=TaxoKG
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_akbc22_rebuttal_HAEK_CompGCN_SEMedical-ReVerb_all

# Aug 7, 2022
# python -m scripts.train_openHAKE_OKG_alone with 'motivation="OKG_only SEMusic-OPIEC"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"' 'opt.epoch=500' \
#     'opt.dataset_mix_mode="OKG_only"'
# python -m scripts.train_openHAKE_OKG_alone with 'motivation="OKG_only SEMusic-ReVerb"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"' 'opt.epoch=300' \
#     'opt.dataset_mix_mode="OKG_only"'
# python -m scripts.train_openHAKE_OKG_alone with 'motivation="TAXO_only SEMusic-OPIEC"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"' 'opt.epoch=300' \
#     'opt.dataset_mix_mode="TAXO_only"'
# python -m scripts.train_openHAKE_OKG_alone with 'motivation="TAXO_only SEMusic-ReVerb"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-ReVerb"' 'opt.epoch=300' \
#     'opt.dataset_mix_mode="TAXO_only"'

# python -m scripts.train_openHAKE_OKG_alone with 'motivation="OKG_only"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"' 'opt.epoch=300' \
#     'opt.dataset_mix_mode="OKG_only"'
# python -m scripts.train_openHAKE_OKG_alone with 'motivation="TAXO_only"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"' 'opt.epoch=300' \
#     'opt.dataset_mix_mode="TAXO_only"'

# CompGCN 
# python -m scripts.train_CompGCN with 'motivation="OKG_only; CompGCN-DistMult"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150' \
#     'opt.dataset_mix_mode="OKG_only"'
# python -m scripts.train_CompGCN with 'motivation="TAXO_only; CompGCN-DistMult"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150' \
#     'opt.dataset_mix_mode="TAXO_only"'
# python -m scripts.train_CompGCN with 'motivation="OKG_only; CompGCN-DistMult"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150' \
#     'opt.dataset_mix_mode="OKG_only"'
# python -m scripts.train_CompGCN with 'motivation="TAXO_only; CompGCN-DistMult"'\
#     'opt.gpu=True' 'opt.dataset_type="SEMusic-OPIEC"'\
#     'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150' \
#     'opt.dataset_mix_mode="TAXO_only"'

# Aug 8
python -m scripts.train_openHAKE_OKG_alone with 'motivation="OKG_only SEMedical-ReVerb"'\
    'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"' 'opt.epoch=300' \
    'opt.dataset_mix_mode="OKG_only"'
python -m scripts.train_openHAKE_OKG_alone with 'motivation="TAXO_only SEMedical-ReVerb"'\
    'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"' 'opt.epoch=300' \
    'opt.dataset_mix_mode="TAXO_only"'
python -m scripts.train_CompGCN with 'motivation="OKG_only; CompGCN-DistMult"'\
    'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"'\
    'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150' \
    'opt.dataset_mix_mode="OKG_only"'
python -m scripts.train_CompGCN with 'motivation="TAXO_only; CompGCN-DistMult"'\
    'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"'\
    'opt.score_func="DistMult"' 'opt.gcn_layer=2' 'opt.gcn_emb_dim=150' \
    'opt.dataset_mix_mode="TAXO_only"'

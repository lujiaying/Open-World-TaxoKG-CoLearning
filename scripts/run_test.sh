#!/bin/bash
#SBATCH --job-name=Test-TaxoKG-TransE
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_test_HAKEGCN

# Jun 20
# python -m scripts.test_taxotransE with 'config_path="logs/Taxo-TransE/11/config.json"' \
#     'checkpoint_path="checkpoints/TaxoTransE/exp_11_WN18RR.best.ckpt"'
# python -m scripts.test_transE with 'config_path="logs/base-TransE/7/config.json"' \
#     'checkpoint_path="checkpoints/TransE/exp_7_WN18RR.best.ckpt"'
# python -m scripts.test_taxotransE with 'config_path="logs/Taxo-TransE/13/config.json"' \
#     'checkpoint_path="checkpoints/TaxoTransE/exp_13_CN100k.best.ckpt"'
# python -m scripts.test_transE with 'config_path="logs/base-TransE/12/config.json"' \
#     'checkpoint_path="checkpoints/TransE/exp_12_CN100k.best.ckpt"'

# Aug 24
# python -m scripts.eval_CountInfer with 'motivation="trial"'\
#     'opt.model_type="NaiveCountInfer"'\
#     'opt.dataset_type="MSCG-OPIEC"'
    #'opt.dataset_type="MSCG-OPIEC"'
    #'opt.dataset_type="MSCG-ReVerb"'
    #'opt.dataset_type="SEMusic-ReVerb"'
    #'opt.dataset_type="SEMedical-ReVerb"'
    #'opt.dataset_type="SEMedical-OPIEC"'
    #'opt.dataset_type="SEMusic-OPIEC"'

# Aug 26
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/1/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_1_SEMedical-OPIEC.best.ckpt"'
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/2/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_2_SEMusic-OPIEC.best.ckpt"'
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/3/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_3_SEMedical-ReVerb.best.ckpt"'
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/4/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_4_SEMusic-ReVerb.best.ckpt"'
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/5/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_5_MSCG-OPIEC.best.ckpt"'
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/8/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_8_SEMedical-OPIEC.best.ckpt"'
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/7/config.json"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_7_SEMusic-OPIEC.best.ckpt"'

# Oct 5
# Human Eval
# python -m scripts.test_RGCN with 'config_path="logs/base-RGCN/13/config.json"'\
#     'checkpoint_path="checkpoints/RGCN/exp_13_MSCG-ReVerb.best.ckpt"'\
#     'human_eval_path="human_eval/models/RGCN/MSCG-ReVerb"'
# python -m scripts.test_CompGCN with 'config_path="logs/CompGCN/23/config.json"'\
#     'checkpoint_path="checkpoints/CompGCN/exp_23_MSCG-ReVerb.best.ckpt"'\
#      'human_eval_path="human_eval/models/CompGCN/MSCG-ReVerb"'
#    'checkpoint_path="checkpoints/CompGCN/exp_15_SEMusic-ReVerb.best.ckpt"'\
#    'checkpoint_path="checkpoints/CompGCN/exp_23_MSCG-ReVerb.best.ckpt"'\
#    'checkpoint_path="checkpoints/CompGCN/exp_11_SEMedical-OPIEC.best.ckpt"'\
# python -m scripts.test_openHAKE with 'config_path="logs/open-HAKE/11/config.json"'\
#      'checkpoint_path="checkpoints/OpenHAKE/exp_11_MSCG-ReVerb.best.ckpt"'\
#      'do_human_eval="human_eval/models/HAKE/MSCG-ReVerb"'
#     'checkpoint_path="checkpoints/OpenHAKE/exp_13_MSCG-OPIEC.best.ckpt"' # have not generate
#     'checkpoint_path="checkpoints/OpenHAKE/exp_11_MSCG-ReVerb.best.ckpt"'\
#    'checkpoint_path="checkpoints/OpenHAKE/exp_8_SEMedical-OPIEC.best.ckpt"'\
#     'checkpoint_path="checkpoints/OpenHAKE/exp_4_SEMusic-ReVerb.best.ckpt"'\

# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/85/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_85_MSCG-ReVerb.best.ckpt"'\
#       'human_eval_path="human_eval/models/HAKEGCN/MSCG-ReVerb/"'
#      'human_eval_path="human_eval/neigh_impact/taxonomic/SEMusic-OPIEC/"'
#      'checkpoint_path="checkpoints/HAKEGCN/exp_3_SEMusic-ReVerb.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_60_SEMedical-OPIEC.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_ocg/exp_69_SEMusic-ReVerb.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_64_SEMedical-ReVerb.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_62_SEMusic-OPIEC.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_73_SEMusic-OPIEC.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_76_SEMedical-ReVerb.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_78_SEMedical-ReVerb.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_72_SEMusic-OPIEC.best.ckpt"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_85_MSCG-ReVerb.best.ckpt"'\

# Oct 10
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/78/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_78_SEMedical-ReVerb.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/taxonomic/SEMedical-ReVerb/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/64/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_64_SEMedical-ReVerb.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/relational/SEMedical-ReVerb/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/76/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_76_SEMedical-ReVerb.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/both/SEMedical-ReVerb/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/72/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_72_SEMusic-OPIEC.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/taxonomic/SEMusic-OPIEC/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/62/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_62_SEMusic-OPIEC.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/relational/SEMusic-OPIEC/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/73/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_73_SEMusic-OPIEC.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/both/SEMusic-OPIEC/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN/17/config.json"'\
#      'checkpoint_path="checkpoints/HAKEGCN/exp_17_MSCG-OPIEC.best.ckpt"'\
#      'human_eval_path="human_eval/neigh_impact/taxonomic/MSCG-OPIEC/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN_by_Oct/84/config.json"'\
#      'checkpoint_path="checkpoints/HAKEGCN_by_Oct/exp_84_MSCG-OPIEC.best.ckpt"'\
#      'human_eval_path="human_eval/neigh_impact/relational/MSCG-OPIEC/"'
# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN/18/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN/exp_18_MSCG-OPIEC.best.ckpt"'\
#       'human_eval_path="human_eval/neigh_impact/both/MSCG-OPIEC/"'


# python -m scripts.test_HAKEGCN with 'config_path="logs/HAKEGCN/19/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN/exp_19_MSCG-ReVerb.best.ckpt"'\
#       'human_eval_path="human_eval/models/HAKEGCN/MSCG-ReVerb"'

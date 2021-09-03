#!/bin/bash
#SBATCH --job-name=Test-TaxoKG-TransE
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_test_openHAKE

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
    #'opt.dataset_type="MSCG-OPIEC"'
    #'opt.dataset_type="MSCG-OPIEC"'
    #'opt.dataset_type="MSCG-ReVerb"'
    #'opt.dataset_type="MSCG-OPIEC"'
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

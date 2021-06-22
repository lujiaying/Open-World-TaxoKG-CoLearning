#!/bin/bash
#SBATCH --job-name=Test-TaxoKG-TransE
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_test_TaxoTransE

# Jun 20
python -m scripts.test_taxotransE with 'config_path="logs/Taxo-TransE/11/config.json"' \
    'checkpoint_path="checkpoints/TaxoTransE/exp_11_WN18RR.best.ckpt"'
# python -m scripts.test_transE with 'config_path="logs/base-TransE/7/config.json"' \
#     'checkpoint_path="checkpoints/TransE/exp_7_WN18RR.best.ckpt"'
# python -m scripts.test_taxotransE with 'config_path="logs/Taxo-TransE/13/config.json"' \
#     'checkpoint_path="checkpoints/TaxoTransE/exp_13_CN100k.best.ckpt"'
# python -m scripts.test_transE with 'config_path="logs/base-TransE/12/config.json"' \
#     'checkpoint_path="checkpoints/TransE/exp_12_CN100k.best.ckpt"'

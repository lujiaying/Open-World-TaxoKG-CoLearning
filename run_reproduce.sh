#!/bin/bash
#SBATCH --job-name=Reproduce-TaxoKG
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_reproduce_HAKEGCN

# SEMusic
python -m scripts.test_HAKEGCN with 'config_path="checkpoints/HAKEGCN_reproduce/SEMusic-OPIEC/config.json"'\
      'checkpoint_path="checkpoints/HAKEGCN_reproduce/SEMusic-OPIEC/exp_62_SEMusic-OPIEC.best.ckpt"'

# SEMedical
# python -m scripts.test_HAKEGCN with 'config_path="checkpoints/HAKEGCN_reproduce/SEMedical-ReVerb/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_reproduce/SEMedical-ReVerb/exp_64_SEMedical-ReVerb.best.ckpt"'

# MSCG
# python -m scripts.test_HAKEGCN with 'config_path="checkpoints/HAKEGCN_reproduce/MSCG-ReVerb/config.json"'\
#       'checkpoint_path="checkpoints/HAKEGCN_reproduce/MSCG-ReVerb/exp_19_MSCG-ReVerb.best.ckpt"'

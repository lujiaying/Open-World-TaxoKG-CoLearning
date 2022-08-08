#!/bin/bash
#SBATCH --job-name=TaxoKG
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_akbc22_rebuttal_HAKEGCN_SEMedical-ReVerb_Taxo_only

# Aug 7
##SBATCH --output=logs/slurm_akbc22_rebuttal_HAKEGCN_SEMedical-OPIEC_all
# python -m scripts.train_HAKEGCN with 'motivation="OKG_only "'\
#      'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"' 'opt.optim_type="RAdam"'\
#      'opt.tok_emb_dim=500' 'opt.emb_dim=1000' 'opt.epoch=1000' 'opt.batch_size=128' 'opt.neg_size=256'\
#      'opt.keep_edges="both"' 'opt.dataset_mix_mode="OKG_only"'
# python -m scripts.train_HAKEGCN with 'motivation="TAXO_only "'\
#      'opt.gpu=True' 'opt.dataset_type="SEMedical-OPIEC"' 'opt.optim_type="RAdam"'\
#      'opt.tok_emb_dim=500' 'opt.emb_dim=1000' 'opt.epoch=1000' 'opt.batch_size=128' 'opt.neg_size=256'\
#      'opt.keep_edges="both"' 'opt.dataset_mix_mode="TAXO_only"'

# Aug 8
# python -m scripts.train_HAKEGCN with 'motivation="OKG_only"'\
#      'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"' 'opt.optim_type="RAdam"'\
#      'opt.tok_emb_dim=500' 'opt.emb_dim=1000' 'opt.epoch=1200' 'opt.batch_size=256' 'opt.neg_size=256'\
#      'opt.keep_edges="both"' 'opt.dataset_mix_mode="OKG_only"'
python -m scripts.train_HAKEGCN with 'motivation="TAXO_only"'\
     'opt.gpu=True' 'opt.dataset_type="SEMedical-ReVerb"' 'opt.optim_type="RAdam"'\
     'opt.tok_emb_dim=500' 'opt.emb_dim=1000' 'opt.epoch=1200' 'opt.batch_size=256' 'opt.neg_size=256'\
     'opt.keep_edges="both"' 'opt.dataset_mix_mode="TAXO_only"'

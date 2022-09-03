# Open-World Taxonomy and Knowledge Graph Co-Learning (AKBC'22)

Data and code of AKBC'22 paper "Open-World Taxonomy and Knowledge Graph Co-Learning". 
For any suggestion/question, please feel free to create an issue or drop an email @ ([jiaying.lu@emory.edu and j.carlyang@emory.edu](mailto:jiaying.lu@emory.edu,j.carlyang@emory.edu)).

**Table of Contents**
- [Datasets](#datasets)
- [Prerequisites](#prerequisites)
- [Reproduce Results](#reproduce-results)

## Datasets
The six datasets used in paper can be downloaded from https://figshare.com/articles/dataset/Taxo-KG-Bench/16415727.  
After downloading and decompressing them under `./data/` directory, the directory looks like:
```
📁 ./data/CGC-OLP-BENCH
|-- 📁 MSCG-OPIEC
|-- 📁 MSCG-ReVerb
|-- 📁 SEMedical-OPIEC
|-- 📁 SEMedical-ReVerb
|-- 📁 SEmusic-OPIEC
|-- 📁 SEMusic-ReVerb
|-- CHANGELOG
```

And each subdirectory represents one of the six datasets, for instance:
```
📁 ./data/CGC-OLP-BENCH/MSCG-OPIEC
|-- oie_triples.train.txt
|-- oie_triples.dev.txt
|-- oie_triples.test.txt
|-- cg_pairs.train.txt
|-- cg_pairs.dev.txt
|-- cg_pairs.test.txt
```
where `oie_triples.*.txt` indicate the OpenKG triples after alginment, and `cg_pairs.*.txt` indicate the AutoTAXOnomy pairs after aligment.

## Prerequisites

```
python>=3.7.10
conda>=4.10.3
```

The packages used in this project is listed in `./taxoKG_conda_env.yml`.
You can re-create and/or reactivate it by the following scripts:

```Shell
# Re-create the environment. This will install the env *taxoKG* under your default conda path
conda env create --file taxoKG_conda_env.yml
# Or if you want to specify the env path
conda env create --file taxoKG_conda_env.yml -p /home/user/anaconda3/envs/env_name

# Reactivate the environment
conda activate taxoKG
```

## Reproduce Results

The porposed HAKEGCN model checkpoints can be downloaded from https://figshare.com/articles/software/HakeGCN_Checkpoints/17108342.  
Please decompress them under `./checkpoints` folder.

Examples of getting the reported numbers for the main experiments (Table 1, 2, 3): 
```
sh run_reproduce.sh
```

Notes for using `run_reproduce.sh`:  
L2-L4 just contains the set-ups for SLURM. I add it because I am working on a SLURM controled GPU server.
You don't need to change anything to run it.
```
#SBATCH --job-name=Reproduce-TaxoKG 
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/slurm_reproduce_HAKEGCN 
```

## Citing Our Work
If you use our data or code in a scientific publication, please cite the following paper:
```bibtex
@inproceedings{lu22:hakeGCN,
  author     = {Jiaying Lu and
                Carl Yang},
  title      = {Open-World Taxonomy and Knowledge Graph Co-Learning},
  year       = {2022},
  Booktitle  = {4th Conference on Automated Knowledge Base Construction},
  Series = {AKBC 2022},
}
```

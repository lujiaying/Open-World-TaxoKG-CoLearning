# Open-World Taxonomy and Knowledge Graph Co-Learning

Code for the manuscript submission "Open-World Taxonomy and Knowledge Graph Co-Learning".

- [Datasets](#datasets)
- [Prerequisites](#prerequisites)
- [Reproduce Results](#reproduce-results)

## Datasets
The six datasets used in paper can be downloaded from https://figshare.com/s/ca54dd1ca5f08a203017.  
After downloading it and decompressing it under `./data/` directory, it looks like:
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

The porposed HAKEGCN model checkpoints can be downloaded from https://figshare.com/s/96647c8dc7f9e73988cc.  
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

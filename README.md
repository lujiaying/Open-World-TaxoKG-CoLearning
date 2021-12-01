# Learn-to-Abstract-Taxonomy

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


# Variant Effect Prediction (VEP) across Individual Human Genomes with GPN

## Overview
This repository contains the code, configuration files, and analysis workflows for evaluating the impact of individual genetic backgrounds (local sequence context and epistasis) on **Genomic Pre-trained Network (GPN)** variant effect predictions. We compare Log-Likelihood Ratio (LLR) scores computed on personalized haplotype genomes from the 1000 Genomes Project (starting with YRI individual `NA19238`) against the standard `GRCh38` reference assembly.

---

## Directory Architecture
Heavy genomic datasets, model weights, and inference outputs are stored outside the repository root (`../`) to keep Git history lightweight:

```imanolhpc
├── gpn/                        # [External] GPN architecture, training runs, and model checkpoints (e.g., checkpoint-120000)
├── data/                       # [External] Reference genomes (GRCh38 softmasked), 1000G VCFs, and personalized FASTAs
└── imanol-pib/                 # [This Repo] Scripts, workflow definitions, and reporting notebooks
    ├── scripts/                # Standalone Python and Slurm scripts for training, consensus generation, and VEP scoring
    ├── notebooks/              # Exploratory data analysis and visualization
    ├── global_params.yml       # Centralized paths and experiment parameters
    ├── workflow.py             # gwf workflow definitions (planned automation)
    └── pixi.toml               # Reproducible environment specification (planned)```

#Current Execution Workflow (Conda + Manual Slurm)

Reproducibility Note: All experiments and results generated to date were executed using a dedicated Conda environment (gpn-gpu) and standalone Slurm batch jobs (sbatch) rather than the automated gwf/pixi pipeline.

##Environment Setup

Activate the Conda environment containing PyTorch, Hugging Face transformers, gpn, pyfaidx, and bcftools:

```(Bash)
conda activate gpn-gpu```

##Model Pre-Training

The GPN model was trained from scratch using Distributed Data Parallel (DDP) across 4 GPUs for 120,000 optimization steps with a 512 bp sequence context window.

Checkpoints: Saved periodically under ../gpn/outputs/checkpoints/ (e.g., checkpoint-95000).








# Template repository for a library project

## Initial set up

```bash
pixi run init
```

## Incomporate updates changes to upstream fork

If you forked the `imanol-pib` rather than using it as template, you can incorporate changes/fixes made to `imanol-pib`.

Add upstream if not already added

```bash
git remote add upstream https://github.com/munch-group/imanol-pib.git
```

Fetch upstream changes

```bash
git fetch upstream
```

Either rebase your changes on top of upstream (cleaner history)

```bash
git rebase upstream/main
```

Or, merge upstream into your fork (preserves history)

```bash
git merge upstream/main
```

If you want to see what's changed upstream before applying:

```bash
git log HEAD..upstream/main
```

See the actual diff

```bash
git diff HEAD...upstream/main
```

Then push your updated fork:

```bash
git push origin main
```

If you rebased and need to force push
    
```bash
git push origin main --force-with-lease
```

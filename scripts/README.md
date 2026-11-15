# Training & Utility Scripts

This directory contains the Slurm batch scripts used to train and resume **Genomic Pre-trained Network (GPN)** models on the HPC cluster, alongside repository maintenance utilities inherited from the project template.

---

## 1. Model Training Scripts (`*.slurm`)

All training runs execute Distributed Data Parallel (DDP) training via `torchrun` across **4× NVIDIA L40S GPUs** (`--partition=gpu-l40s`, `--gres=gpu:4`, 64 GB RAM, 16 CPUs) inside the `gpn-gpu` Conda environment.

| Script | Target Model / Dataset | YAML Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| `human_train.slurm` | Human (`data/human_intervals`) | `recipes/gpn_training/human.yaml` | Trains the human reference GPN model from scratch (`outputs/gpn_human_4gpu`). |
| `brass_train.slurm` | Brassicales (`genomes-brassicales-balanced-v1`) | `recipes/gpn_training/4gpu.yaml` | Launches multi-GPU training from scratch in offline Hugging Face mode (`outputs/gpn_4gpu_scratch`). |
| `brass_resume_train.slurm` | Brassicales (`genomes-brassicales-balanced-v1`) | `recipes/gpn_training/4gpu.yaml` | Resumes an interrupted training run from an existing checkpoint specified in the `.yaml` config. |

### Cluster & Hardware Stability Flags
Because multi-GPU training runs on `gpu-l40s` nodes can encounter PCIe peer-to-peer deadlocks, all training scripts enforce specific NCCL and runtime environment variables:
* `NCCL_P2P_DISABLE=1` (and `NCCL_P2P_LEVEL=PHB` on human runs): Disables direct GPU-to-GPU P2P transfers over PCIe to prevent hangs during gradient synchronization.
* `MASTER_PORT=$(shuf -i ...)`: Assigns a randomized rendezvous port per Slurm job to avoid port collisions when multiple jobs share a node.
* `TRANSFORMERS_OFFLINE=1` & `HF_DATASETS_OFFLINE=1`: Forces Hugging Face libraries to load local datasets and tokenizers without attempting external network requests from compute nodes (this helps against job terminations due to disconnections).

### Submitting a Training Job
```bash
sbatch scripts/human_train.slurm

## 2. YAML Configuration Breakdown (recipes/gpn_training/*.yaml)

Training hyperparameters, model architecture, and checkpointing behavior are controlled via YAML files passed to gpn.cli ss train. Both recipes share the same nucleotide tokenizer (tokenizer-gpn-brassicales, covering standard DNA tokens A, C, G, T, N) and optimization schedule, with minor architectural and evaluation differences.

### Data and output paths

* `dataset_name`: Path to the pre-processed genomic intervals dataset (e.g., `data/human_intervals` or `data/genomes-brassicales-balanced-v1`).

* `tokenizer_name`: Path to the single-nucleotide DNA tokenizer (tokenizer-gpn-brassicales, mapping A, C, G, T, N).

* `output_dir`: Target directory where model checkpoints and training logs are saved.

### Model Architecture

* `encoder=convnet`: Uses a 25-layer dilated convolutional neural network (`num_hidden_layers=25`, `hidden_size=512`, `intermediate_size=2048`).

* `first_kernel_size` / `rest_kernel_size` / `dilation_cycle`: Define the convolutional filter widths and exponential dilation schedule, expanding the receptive field across the 512 bp sequence context window.

### Optimization and Batch Size

* Effective Global Batch Size (2,048): Controlled by `per_device_train_batch_size: 64` × `gradient_accumulation_steps: 8` across 4 GPUs.

* `max_steps: 120000`: Total number of optimization steps (this number is chosen empirically and it is not claimed that it is optimal).

* `learning_rate: 0.001`: Peak learning rate managed by a `constant_with_warmup` scheduler (`warmup_steps: 1000`) using the `adamw_torch optimizer` (`weight_decay: 0.01`).

* `fp16: true`: Enables half-precision floating-point training to accelerate throughput and reduce GPU memory usage.


### Masked Language Modelling and Soft-Masking

* `mlm_probability: 0.15`: Randomly masks 15% of nucleotides in each input window for self-supervised prediction.

* `soft_masked_loss_weight_train: 0.1`: Down-weights the cross-entropy loss on lowercase (repetitive/soft-masked) bases to 10% during training so model capacity focuses on non-repetitive functional sequence.

* `soft_masked_loss_weight_evaluation: 0.0`: Excludes soft-masked repeat regions when computing validation loss.


### Checkpoints and Resuming Interrupted Runs

Because Slurm jobs have a strict 48-hour wall-time limit (--time=48:00:00), long 120k-step runs can be split across sequential submissions using Hugging Face checkpoints:

Periodic Saving (`save_steps: 5000`): Every 5,000 steps, the trainer writes a snapshot directory inside `output_dir` (e.g., checkpoint-70000/, checkpoint-120000/).

Disk Management (`save_total_limit`): Retains only the most recent checkpoints on disk and automatically deletes older snapshots to conserve cluster storage.

Checkpoint Contents: Each checkpoint-<step>/ directory stores:

model.safetensors / pytorch_model.bin: Model weights.

optimizer.pt & scheduler.pt: Exact AdamW momentum/variance states and learning rate schedule position.

`trainer_state.json` & `rng_state_*.pth`: Step counter, loss history, and per-GPU random number generator states for deterministic continuation.

How to Resume: Set `resume_from_checkpoint` in the .yaml file to the latest saved folder before submitting the resume script:

`resume_from_checkpoint: /faststorage/project/imanol-pib/people/imanolhpc/gpn/outputs/gpn_4gpu_scratch/checkpoint-70000`


## 3. Data Processing Scripts

The dataset used for Brassicales training exists in HuggingFace ready for use, but training on other species requires preparing 
the dataset for the conditions required by GPN. As stated by the authors: "Any replacement dataset_name must identify a local or
Hugging Face dataset with train and validation splits and a fixed-length seq string in every row."

* `prepare_human_intervals.py`: Takes as input the reference human genome and creates training and validation splits in Parquet
   format, with `seq_length = 512`.


 

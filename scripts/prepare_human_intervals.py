import os
import pandas as pd
from pyfaidx import Fasta

# Paths
FASTA_PATH = "/faststorage/project/imanol-pib/people/imanolhpc/data/human_ref/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"
OUTPUT_DIR = "/faststorage/project/imanol-pib/people/imanolhpc/data/human_intervals_sm"
WINDOW_SIZE = 512
STEP_SIZE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chromosomes: 1 to 20 for training; 21 and 22 for validation
TRAIN_CHROMS = [str(i) for i in range(1, 21)]
VAL_CHROMS = ["21", "22"]

print(f"Indexing and loading FASTA: {FASTA_PATH}...")
genome = Fasta(FASTA_PATH, as_raw=True, sequence_always_upper=True)

train_records = []
val_records = []

def extract_intervals(chrom_list, target_list):
    for chrom in chrom_list:
        # Checking chr prefix 'chr'
        chrom_key = chrom if chrom in genome.keys() else f"chr{chrom}"
        if chrom_key not in genome.keys():
            print(f"Warning: Chromosome {chrom} not found in FASTA. Skipping...")
            continue
        
        seq_len = len(genome[chrom_key])
        print(f"Extracting chromosome windows {chrom} ({seq_len:,} pb)...")
        seq = genome[chrom_key][:]
        
        for start in range(0, seq_len - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            subseq = seq[start:end]
            if "N" not in subseq:
                target_list.append({"chrom": chrom, "start": start, "end": end})

print("\n--- Processing TRAIN (chr1 - chr20) ---")
extract_intervals(TRAIN_CHROMS, train_records)

print("\n--- Processing VALIDATION (chr21 - chr22) ---")
extract_intervals(VAL_CHROMS, val_records)

print("\nSaving datasets in Parquet format...")
df_train = pd.DataFrame(train_records)
df_val = pd.DataFrame(val_records)

train_out = os.path.join(OUTPUT_DIR, "train.parquet")
val_out = os.path.join(OUTPUT_DIR, "validation.parquet")

df_train.to_parquet(train_out, index=False)
df_val.to_parquet(val_out, index=False)

print(f"Process Finished:")
print(f" -> Train: {len(df_train):,} windows saved in {train_out}")
print(f" -> Validation: {len(df_val):,} windows saved in {val_out}")

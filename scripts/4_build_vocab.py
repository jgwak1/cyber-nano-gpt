import os
import glob
import json
import csv
from collections import defaultdict

# =============================================================================
# 1. PATH CONFIGURATION & ARCHITECTURAL CONTEXT
# =============================================================================
# ARCHITECTURAL DECISION: Why is PySpark output read from local disk?
# In a production ML architecture, the ETL phase (Spark) is horizontally scaled 
# and writes to a distributed data lake (e.g., AWS S3). However, the Training phase 
# (PyTorch) requires data to be staged onto local NVMe SSDs attached to the GPU nodes. 
# Streaming data directly from S3 during training causes severe network I/O bottlenecks, 
# starving the GPUs. This local directory simulates that mandatory staging phase.
INPUT_DIR = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"

# Output path for the mapping dictionary. The size of this JSON determines 
# the dimension of the LLM's Embedding Layer (vocab_size).
OUTPUT_VOCAB_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\vocab.json"

# =============================================================================
# 2. FEATURE CATEGORIZATION
# =============================================================================
# BINNED_PREFIXES: Continuous variables (Log-binned). 
# Characteristic: Must populate a contiguous integer space from 0 to MAX without gaps.
BINNED_PREFIXES = {
    "DUR", "FWD_PKT", "BWD_PKT", "FWD_BYT", "BWD_BYT", 
    "BYT_SEC", "PKT_SEC", "MAX_LEN", "MEAN_LEN", "IAT_MEAN", 
    "IAT_MAX", "WIN"
}

# RAW_PREFIXES: Categorical variables (Ports, Protocols, TCP Flags).
# Characteristic: Only explicitly observed values are mapped. Unobserved ports 
# during inference will fallback to a prefix-specific UNK token.
RAW_PREFIXES = {
    "PORT", "PROTO", "D_U_RATIO", "SYN", "ACK", "FIN", "RST"
}

def main():
    print(">>> 1. Scanning Spark output partitions...")
    # Scan for Spark partition files (excluding Hadoop CRC checksum files).
    csv_files = glob.glob(os.path.join(INPUT_DIR, "part-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Data not found in {INPUT_DIR}")

    # State trackers
    binned_max_tracker = defaultdict(int) # Tracks only the 'maximum' observed value for continuous features.
    raw_observed_sets = defaultdict(set)  # Collects unique identifiers for categorical features.

    # =============================================================================
    # 3. EXHAUSTIVE SCAN (O(N) String Parsing)
    # =============================================================================
    print(">>> 2. Performing exhaustive scan to determine feature boundaries...")
    # Why use the built-in csv module instead of Pandas?
    # Loading 1M+ rows into RAM simultaneously causes unnecessary memory overhead. 
    # A generator-based, line-by-line read guarantees memory stability regardless of file size.
    
    total_seqs = 0
    benign_seqs = 0  # Tracker for strictly normal traffic

    for file_path in csv_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            
            for row in reader:
                if not row: continue
                sequence = row[0]
                label = row[1]                
                
                total_seqs += 1
                
                # Strict Isolation of Normal Behavior.
                # Vocabulary MUST NOT contain artifacts unique to malicious traffic.
                if label != "Benign":
                    continue
                    
                benign_seqs += 1



                # Tokenize by space delimiter
                tokens = sequence.split(" ")
                for token in tokens:
                    if token == "[SEP]": continue
                    
                    # Strip brackets ("[PORT_443]" -> "PORT_443")
                    clean_token = token.strip("[]")
                    if "_" not in clean_token: continue
                    
                    # Split strictly at the LAST underscore to handle prefixes with underscores.
                    # e.g., "FWD_PKT_3" -> ["FWD_PKT", "3"]
                    prefix, value_str = clean_token.rsplit("_", 1)
                    value = int(value_str)
                    
                    if prefix in BINNED_PREFIXES:
                        # Update maximum bound (used later to force-fill 0 to MAX)
                        if value > binned_max_tracker[prefix]:
                            binned_max_tracker[prefix] = value
                    elif prefix in RAW_PREFIXES:
                        # Register observed categorical identifier
                        raw_observed_sets[prefix].add(value)

    print(f"    - Scanned {total_seqs} sequences across {len(csv_files)} partitions.")
    print(f"    - Used for Vocab (Benign Only): {benign_seqs} sequences.")
    # =============================================================================
    # 4. VOCABULARY COMPILATION (Pre-emptive Bin Population)
    # =============================================================================
    print(">>> 3. Compiling Custom Vocabulary Matrix...")
    vocab = {}
    current_id = 0

    def add_token(token_str):
        nonlocal current_id
        if token_str not in vocab:
            vocab[token_str] = current_id
            current_id += 1

    # [1] Boundary Token (Assigned the lowest ID)
    add_token("[SEP]")
    
    # [2] Pre-allocate Prefix-Specific UNK Tokens
    # Provides a safe fallback for unobserved categorical values during inference.
    for prefix in RAW_PREFIXES:
        add_token(f"[{prefix}_UNK]")
    for prefix in BINNED_PREFIXES:
        add_token(f"[{prefix}_UNK]") 

    # [3] Pre-emptive Continuous Bins (Solution 2 Implementation)
    print("\n>>> [DEBUG] Absolute Max Values found in dataset:")
    for prefix, max_val in binned_max_tracker.items():
        print(f"    - {prefix}: {max_val}")

    # Core Logic: Even if "FWD_PKT_2" is missing from the training data, 
    # we forcefully allocate it if it falls between 0 and the observed MAX. 
    for prefix in BINNED_PREFIXES:
        raw_max = binned_max_tracker[prefix]
        
        # Safety Cap: Force truncate to 30 if physically impossible garbage values exist.
        # This fundamentally prevents MemoryError caused by data corruption exhausting RAM.
        safe_max = min(raw_max, 30)
        
        for v in range(0, safe_max + 1):
            add_token(f"[{prefix}_{v}]")

    # [4] Categorical Tokens (Exact mapping)
    # Only explicitly observed ports/protocols are registered.
    for prefix in RAW_PREFIXES:
        for v in sorted(list(raw_observed_sets[prefix])):
            add_token(f"[{prefix}_{v}]")

    # =============================================================================
    # 5. PHYSICAL DISK WRITE (JSON Serialization)
    # =============================================================================
    with open(OUTPUT_VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)
        
    print(f"\n>>> 4. Vocab Compilation Complete.")
    print(f"    - Total Vocabulary Size: {len(vocab)} unique tokens.")
    print(f"    - Saved to: {OUTPUT_VOCAB_PATH}")

if __name__ == "__main__":
    main()
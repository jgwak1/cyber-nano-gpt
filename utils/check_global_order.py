import os
import glob
import pandas as pd
from datetime import datetime

def verify_dataset_integrity(data_dir):
    """
    Validates both Intra-shard and Inter-shard chronological order.
    Ensures that the distributed range partitioning is intact.
    """
    # 1. Load and sort Spark shards using the same logic as the DataLoader.
    # Sorting is mandatory to reconstruct the global chronological sequence.
    files = sorted(glob.glob(os.path.join(data_dir, "part-*.csv")))
    
    if not files:
        print("[ERROR] No shard files found. Check the data directory.")
        return

    global_last_timestamp = None
    total_violations = 0

    print(f">>> Scanning {len(files)} shards for temporal integrity...")

    for i, file_path in enumerate(files):
        # Shards are relatively small (~35MB), so Pandas is efficient for rapid validation.
        df = pd.read_csv(file_path)
        
        # Convert 'Timestamp' column to datetime objects for accurate comparison.
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # A. Intra-shard Check: Verify the sequence is monotonically increasing within a single file.
        is_internally_sorted = df['Timestamp'].is_monotonic_increasing
        
        # B. Inter-shard Check: Verify temporal continuity with the previous shard.
        shard_first_ts = df['Timestamp'].iloc[0]
        shard_last_ts = df['Timestamp'].iloc[-1]
        
        inter_shard_violation = False
        if global_last_timestamp is not None:
            if shard_first_ts < global_last_timestamp:
                inter_shard_violation = True
                total_violations += 1

        # Output validation results for the current shard
        status = "PASS" if is_internally_sorted and not inter_shard_violation else "FAIL"
        print(f"[{status}] Shard {i:02d} | Start: {shard_first_ts} | End: {shard_last_ts}")
        
        if inter_shard_violation:
            print(f"    ! Critical: Temporal inversion detected at Shard {i}.")
            print(f"      (Current Start: {shard_first_ts} < Previous End: {global_last_timestamp})")

        if not is_internally_sorted:
            print(f"    ! Warning: Internal sorting failed within Shard {i}.")

        global_last_timestamp = shard_last_ts

    print("-" * 50)
    if total_violations == 0:
        print(">>> SUCCESS: Global Chronological Order is Verified.")
    else:
        print(f">>> FAILED: {total_violations} global order violations found.")

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"
    verify_dataset_integrity(DATA_PATH)
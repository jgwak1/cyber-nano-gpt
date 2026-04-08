import os
import glob
import pandas as pd

def verify_production_shards(data_dir, min_required_rows=13):
    """
    Integration test to verify the correctness of the ETL pipeline's Hard Boundary Reset.
    Ensures absolute sequence integrity before feeding data to PyTorch DataLoader.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "part-*.csv")))
    if not files:
        raise FileNotFoundError(f"[FAIL] No shard files found in {data_dir}")

    print(f">>> Loading {len(files)} production shards for Integrity Check...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    print("\n" + "="*50)
    print(">>> [VERIFICATION] Hard Boundary Reset Correctness")
    print("="*50)

    # 1. Contamination Check
    bot_count = len(df[df['Label'] != 'Benign'])
    if bot_count > 0:
        print(f"[FAIL] Contamination detected: {bot_count} non-Benign rows found.")
    else:
        print("[PASS] Zero Bot contamination. Dataset is 100% pure Benign.")

    # 2. Block Integrity Check (No Sub-batch Fragments)
    block_sizes = df.groupby('block_id').size()
    invalid_blocks = block_sizes[block_sizes < min_required_rows]
    
    if len(invalid_blocks) > 0:
        print(f"[FAIL] Found {len(invalid_blocks)} blocks with fewer than {min_required_rows} rows.")
    else:
        print(f"[PASS] All {len(block_sizes):,} blocks meet the minimum size requirement (>= {min_required_rows} rows).")

    # 3. Final Output Metrics for Engineer Log
    print("-" * 50)
    print(f"Total Usable Training Rows:  {len(df):,}")
    print(f"Total Valid Blocks:          {len(block_sizes):,}")
    print(f"Minimum Block Size Verified: {block_sizes.min()} rows")
    print(f"Maximum Block Size Verified: {block_sizes.max():,} rows")
    print("="*50 + "\n")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"
    verify_production_shards(PROCESSED_DATA_PATH)
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, window

# Infrastructure paths
JAVA_HOME_PATH = r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
HADOOP_HOME_PATH = r"C:\hadoop"
os.environ["JAVA_HOME"] = JAVA_HOME_PATH
os.environ["HADOOP_HOME"] = HADOOP_HOME_PATH

def run_full_analysis(data_dir, block_size_tokens=256, tokens_per_row=20):
    spark = SparkSession.builder \
        .appName("CyberNanoGPT-Dataset-Analysis") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # 1. Load the processed shards from local disk
    print(f">>> Loading dataset from: {data_dir}")
    df = spark.read.csv(data_dir, header=True, inferSchema=True)

    # -------------------------------------------------------------------------
    # AUDIT 1: Global Label Distribution
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(">>> [AUDIT 1] Global Label Distribution (Post-Cleaning):")
    df.groupBy("Label").count().orderBy(col("count").desc()).show()
    
    # -------------------------------------------------------------------------
    # AUDIT 2: Temporal Label Distribution (10-min Windows)
    # -------------------------------------------------------------------------
    df_ts = df.withColumn("ts", to_timestamp(col("Timestamp"), "MM/dd/yyyy HH:mm:ss"))
    print("\n" + "="*50)
    print(">>> [AUDIT 2] Analyzing Temporal Label Distribution (10-min Windows):")
    df_ts.groupBy(window(col("ts"), "10 minutes"), "Label") \
        .count() \
        .orderBy("window.start") \
        .show(100, truncate=False)

    # -------------------------------------------------------------------------
    # AUDIT 3: Contiguous Benign Blocks (Micro-Continuity)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(">>> [AUDIT 3] Contiguous Benign Blocks (Micro-Continuity):")
    
    # 병목 방지를 위해 Timestamp와 Label만 추출하여 Pandas로 전환
    pdf = df.select("Timestamp", "Label").toPandas()
    pdf['Timestamp'] = pd.to_datetime(pdf['Timestamp'])
    pdf = pdf.sort_values('Timestamp').reset_index(drop=True)

    # Gaps and Islands
    pdf['block_id'] = (pdf['Label'] != pdf['Label'].shift(1)).cumsum()
    
    benign_blocks = pdf[pdf['Label'] == 'Benign']
    block_lengths_rows = benign_blocks.groupby('block_id').size()
    block_lengths_tokens = block_lengths_rows * tokens_per_row
    
    min_rows_required = (block_size_tokens // tokens_per_row) + 1
    valid_blocks = block_lengths_rows[block_lengths_rows >= min_rows_required]
    
    total_benign_rows = len(benign_blocks)
    preserved_rows = valid_blocks.sum()
    data_preservation_ratio = (preserved_rows / total_benign_rows) * 100 if total_benign_rows > 0 else 0
    valid_tokens = valid_blocks * tokens_per_row

    # =========================================================================
    # FIRST PRINCIPLES DEFINITION OF "TOO SMALL"
    # 1. The Transformer model requires a fixed input sequence length: 'block_size_tokens' (e.g., 256).
    # 2. Each row (network session) yields approximately 'tokens_per_row' (e.g., 20).
    # 3. To construct at least one valid training sequence without breaking temporal causality, 
    #    a contiguous block must contain at least: ceil(block_size_tokens / tokens_per_row) rows.
    #    Example: 256 / 20 = 12.8 -> requires 13 contiguous rows.
    # 4. Therefore, any block with strictly fewer than 13 rows (< min_rows_required) is "too small" 
    #    because it cannot generate even a single complete context window for the model.
    # =========================================================================

    # Global Block Distribution (Including Blocks that are too small for a batch)
    print(f"Total Isolated Benign Blocks: {len(block_lengths_rows):,}")
    print(f"Global Average Length: {block_lengths_rows.mean():.1f} rows ({block_lengths_tokens.mean():.1f} tokens)")
    print(f"Global Median Length: {block_lengths_rows.median():.1f} rows ({block_lengths_tokens.median():.1f} tokens)")
    print(f"Global Max Length: {block_lengths_rows.max():.1f} rows ({block_lengths_tokens.max():.1f} tokens)")
    print(f"Global Min Length: {block_lengths_rows.min():.1f} rows ({block_lengths_tokens.min():.1f} tokens)")

    # -------------------------------------------------------------------------
    # AUDIT 3.1: Isolated Distribution of Valid Blocks (Excluding Blocks that are too small for a batch)
    # -------------------------------------------------------------------------
    print("-" * 50)
    print(f">>> [AUDIT 3.1] Distribution of VALID Blocks (>= {min_rows_required} rows):")
    print(f"Total Valid Blocks: {len(valid_blocks):,}")
    print(f"Data Preservation Ratio: {data_preservation_ratio:.2f}% ({preserved_rows:,} rows)")
    print(f"Valid Average Length: {valid_blocks.mean():.1f} rows ({valid_tokens.mean():.1f} tokens)")
    print(f"Valid Median Length:  {valid_blocks.median():.1f} rows ({valid_tokens.median():.1f} tokens)")
    print(f"Valid Max Length:     {valid_blocks.max():,} rows ({valid_tokens.max():,} tokens)")
    print(f"Valid Min Length:     {valid_blocks.min():,} rows ({valid_tokens.min():,} tokens)")

    print("="*50 + "\n")

    spark.stop()

if __name__ == "__main__":
    PROCESSED_DATA_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"
    run_full_analysis(PROCESSED_DATA_PATH)
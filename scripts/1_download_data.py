import os
import subprocess
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# Define local directory for raw data ingestion
RAW_DATA_DIR = "data/raw"

# Public S3 bucket for CIC-IDS-2018 (Managed by AWS)
S3_BUCKET_PATH = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"

# TARGET_FILE: 'Friday-02-03-2018' contains Botnet (Ares/Zeus) and Benign traffic.
# This is optimal for validating the "Probability Engine" (Anomaly Detection).
TARGET_FILE = "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
OUTPUT_PATH = os.path.join(RAW_DATA_DIR, TARGET_FILE)

def ensure_directory_exists(path):
    """Create directory if it does not exist to prevent FileNotFoundError."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f">>> Created directory: {path}")

def download_data_via_s3_cli(s3_path, local_path):
    """
    Executes AWS CLI for high-performance data transfer.
    Uses --no-sign-request for public access without AWS credentials.
    """
    print(f">>> Starting download: {TARGET_FILE}")
    try:
        # Utilizing subprocess to call AWS CLI for multi-part download support
        subprocess.run([
            "aws", "s3", "cp", 
            os.path.join(s3_path, TARGET_FILE), 
            local_path, 
            "--no-sign-request"
        ], check=True)
        print(">>> Download successful.")
    except subprocess.CalledProcessError as e:
        print(f">>> AWS CLI error: {e}")
    except Exception as e:
        print(f">>> Unexpected error: {e}")

def validate_schema(file_path):
    """
    Perform a quick sanity check on the CSV structure.
    Verify essential columns for the Transformer serialization.
    """
    print(">>> Validating CSV Schema...")
    try:
        # Load only the first 5 rows to minimize memory footprint
        df_sample = pd.read_csv(file_path, nrows=5)
        
        # Check for mandatory feature: 'Dst Port'
        if "Dst Port" in df_sample.columns:
            print(">>> Validation Passed: 'Dst Port' column found.")
            print(f">>> Feature Count: {len(df_sample.columns)}")
        else:
            print(">>> Validation Failed: Missing 'Dst Port'. Check file integrity.")
            
    except Exception as e:
        print(f">>> Data validation error: {e}")

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    ensure_directory_exists(RAW_DATA_DIR)
    
    # Download logic: Skips if the file is already present locally
    if not os.path.exists(OUTPUT_PATH):
        download_data_via_s3_cli(S3_BUCKET_PATH, OUTPUT_PATH)
    else:
        print(f">>> {TARGET_FILE} already exists. Skipping download.")

    # Integrity check before proceeding to DVC ingestion
    validate_schema(OUTPUT_PATH)
    
    print("\n>>> Ingestion Pipeline Step 1: Complete.")
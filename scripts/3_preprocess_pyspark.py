import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log10, floor, concat_ws, lit, concat

# =============================================================================
# 1. INFRASTRUCTURE & ENVIRONMENT HARDENING
# =============================================================================
# Hard-injecting paths. PySpark uses Py4J to spawn JVMs, making JAVA_HOME mandatory.
# On Windows, Hadoop's winutils.exe acts as a required POSIX bridge to interact with S3.
JAVA_HOME_PATH = r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
HADOOP_HOME_PATH = r"C:\hadoop"
SPARK_TEMP_DIR = r"C:\spark_temp"

os.environ["JAVA_HOME"] = JAVA_HOME_PATH
os.environ["HADOOP_HOME"] = HADOOP_HOME_PATH

# Physical disk target for the final serialized tokens.
OUTPUT_DIR = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"

def main():
    # -------------------------------------------------------------------------
    # ARCHITECTURAL DECISION: Why Spark for a 350MB CSV?
    # Context: Pandas can easily process 350MB in-memory on a single node.
    # Justification: This is a design-for-scalability choice. In an actual SOC environment, 
    # network flows (NetFlow/PCAP) generate Terabytes of logs daily. 
    # Building this on Spark ensures the ETL pipeline is horizontally scalable across 
    # distributed clusters (e.g., YARN, EMR, Kubernetes) with zero code refactoring.
    # -------------------------------------------------------------------------
    spark = SparkSession.builder \
        .appName("CyberNanoGPT-Preprocessing") \
        .config("spark.driver.memory", "6g") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.local.dir", SPARK_TEMP_DIR) \
        .getOrCreate()

    # Override S3A defaults. Bypasses Hadoop string parsing bugs ("24h" -> 86400) 
    # and forces anonymous access for public buckets.
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    
    # Monkey-Patches for now
    hadoop_conf.set("fs.s3a.multipart.purge.age", "86400") 
    hadoop_conf.set("fs.s3a.threads.keepalivetime", "60")
    hadoop_conf.set("fs.s3a.connection.establish.timeout", "5000")
    hadoop_conf.set("fs.s3a.connection.timeout", "200000")
    hadoop_conf.set("fs.s3a.connection.request.timeout", "60000")

    hadoop_conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")

    # =============================================================================
    # 2. DATA INGESTION (LAZY EVALUATION)
    # =============================================================================
    S3A_PATH = "s3a://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    print(">>> 1. Building DAG for S3 ingestion...")
    
    # Spark Lazy Evaluation: This does NOT load data into RAM. 
    # It only fetches the schema to construct the Directed Acyclic Graph (DAG) for future execution.
    df = spark.read.csv(S3A_PATH, header=True, inferSchema=True)

    # =============================================================================
    # 3. FEATURE DIMENSIONALITY REDUCTION (19 kept, 61 dropped)
    # =============================================================================
    # DROP RATIONALE:
    # 1. Temporal/Spatial Bias: 'Timestamp', 'Source IP'. Model will memorize time/location instead of behavior.
    # 2. Multicollinearity (100% Correlation): 'Subflow *' exactly duplicates 'Tot *'. 'Fwd Seg Size Avg' == 'Fwd Pkt Len Mean'.
    # 3. Structural Redundancy: 'Fwd/Bwd IAT *' metrics are mathematical derivatives of 'Flow Duration' and 'Flow IAT'.
    # 4. Zero Variance / Extreme Sparsity: 'URG/CWE/ECE' flags. 'Active/Idle' metrics (only exist in long-polling sessions, blank otherwise). 
    #                                      'Fwd Pkt Len Min' (fixed TCP header size).
    
    # KEEP RATIONALE: Core markers of volumetric attacks, beaconing, and protocol violations.
    target_cols = [
        "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", 
        "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Byts/s", "Flow Pkts/s", 
        "Fwd Pkt Len Max", "Fwd Pkt Len Mean", "Flow IAT Mean", "Flow IAT Max", 
        "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt", 
        "Init Fwd Win Byts", "Down/Up Ratio", 
        "Label" # ground-truth
    ]
    
    # Cast variables to double for log10 operations. Drops corrupted rows.
    df_clean = df.select([col(c).cast("double").alias(c) if c != "Label" else col(c) for c in target_cols]).dropna()

    # =============================================================================
    # 4. SERIALIZATION & DISCRETIZATION
    # =============================================================================
    print(">>> 2. Constructing Serialization Map...")
    
    def bin_token(prefix, column_name):
        # Log-binning. Compresses continuous scale into discrete order-of-magnitude tokens 
        # to restrict vocabulary explosion in the LLM embedding layer.
        # e.g., 15300 bytes -> log10(15301) -> 4.18 -> floor() -> [FWD_BYT_4]
        return concat(lit(f"[{prefix}_"), floor(log10(col(column_name) + 1)), lit("]"))
        
    def raw_token(prefix, column_name):
        # Direct string casting for discrete identifiers/flags (e.g., [PORT_443], [SYN_1]).
        return concat(lit(f"[{prefix}_"), col(column_name).cast("int"), lit("]"))

    # Sequence Ordering: Context flows from Macro (Identifiers & Volume) -> Micro (Timing) -> State (Flags)
    tokens = [
        raw_token("PORT", "Dst Port"),
        raw_token("PROTO", "Protocol"),
        bin_token("DUR", "Flow Duration"),
        bin_token("FWD_PKT", "Tot Fwd Pkts"),
        bin_token("BWD_PKT", "Tot Bwd Pkts"),
        bin_token("FWD_BYT", "TotLen Fwd Pkts"),
        bin_token("BWD_BYT", "TotLen Bwd Pkts"),
        bin_token("BYT_SEC", "Flow Byts/s"),
        bin_token("PKT_SEC", "Flow Pkts/s"),
        bin_token("MAX_LEN", "Fwd Pkt Len Max"),
        bin_token("MEAN_LEN", "Fwd Pkt Len Mean"),
        bin_token("IAT_MEAN", "Flow IAT Mean"),
        bin_token("IAT_MAX", "Flow IAT Max"),
        bin_token("WIN", "Init Fwd Win Byts"),
        raw_token("D_U_RATIO", "Down/Up Ratio"),
        raw_token("SYN", "SYN Flag Cnt"),
        raw_token("ACK", "ACK Flag Cnt"),
        raw_token("FIN", "FIN Flag Cnt"),
        raw_token("RST", "RST Flag Cnt"),
        # Boundary token. Critical to prevent cross-attention bleeding between independent sessions.
        lit("[SEP]") 
    ]

    # Materialize the 1D sequence using space delimiter.
    df_final = df_clean.withColumn("Sequence", concat_ws(" ", *tokens)).select("Sequence", "Label")

    # =============================================================================
    # 5. I/O EXECUTION (ACTION TRIGGERS)
    # =============================================================================
    # df_final currently exists ONLY as a logical DAG. It is NOT in RAM or on Disk.
    
    # Action 1: Triggers the DAG. Loads subset into RAM, processes it, and prints to console.
    print("\n>>> 3. Console Output (Volatile Memory):")
    df_final.show(10, truncate=False)

    # I/O cleanup to prevent PySpark write collisions.
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # Action 2: Triggers full DAG. Pulls from S3 -> RAM -> Transforms -> Writes to local SSD.
    # You can view the physical output files (CSV/TXT) in the OUTPUT_DIR path.
    # These files are the actual dataset that PyTorch will ingest.
    print(f"\n>>> 4. Serializing to Physical Disk: {OUTPUT_DIR}")
    df_final.write.mode("overwrite").option("header", "true").csv(OUTPUT_DIR)
    print(">>> I/O Write Complete.")

    spark.stop()

if __name__ == "__main__":
    main()
import os
from pyspark.sql import SparkSession

# Define the S3A Data Lake path for the CIC-IDS2018 dataset
S3A_PATH = "s3a://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"

# Runtime Environment Injection: Bypassing Windows registry latency/permission issues
# Ensure these paths match your local installation of OpenJDK 17 and Winutils
JAVA_HOME_PATH = r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
HADOOP_HOME_PATH = r"C:\hadoop"

os.environ["JAVA_HOME"] = JAVA_HOME_PATH
os.environ["HADOOP_HOME"] = HADOOP_HOME_PATH

# Manually update PATH to include Java and Hadoop binaries for the child JVM process
current_paths = os.environ.get("PATH", "").split(os.pathsep)
extra_paths = [os.path.join(JAVA_HOME_PATH, "bin"), os.path.join(HADOOP_HOME_PATH, "bin")]
for p in extra_paths:
    if p not in current_paths:
        current_paths.insert(0, p)
os.environ["PATH"] = os.pathsep.join(current_paths)

def main():
    # External Maven packages required for S3A filesystem support in Spark 3.x
    packages = [
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262"
    ]

    print(">>> Initializing SparkSession with S3A Data Lake connectors...")
    spark = SparkSession.builder \
        .appName("CyberNanoGPT-S3-DataLake-Inspection") \
        .config("spark.driver.memory", "4g") \
        .config("spark.jars.packages", ",".join(packages)) \
        .getOrCreate()

    # Hadoop Configuration: Tuning S3A parameters for stability and performance
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    
    # CRITICAL: Resolve 'NumberFormatException' caused by string-based time units (e.g., "24h", "60s")
    # Explicitly overriding these with integer values to bypass library parsing bugs
    hadoop_conf.set("fs.s3a.multipart.purge.age", "86400") 
    hadoop_conf.set("fs.s3a.threads.keepalivetime", "60")
    hadoop_conf.set("fs.s3a.connection.timeout", "60000")
    hadoop_conf.set("fs.s3a.connection.establish.timeout", "5000")
    
    # Security/Performance: Enable anonymous access for public S3 buckets and fast upload mode
    hadoop_conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.fast.upload", "true")

    print(f">>> Fetching schema from S3: {S3A_PATH}")
    try:
        # Load dataset with schema inference to identify all 80 features
        df = spark.read.csv(S3A_PATH, header=True, inferSchema=True)
        
        columns = df.columns
        print("\n" + "="*60)
        print(f"Feature Discovery Success: {len(columns)} columns identified")
        print("="*60)
        # Display features in blocks of 10 for readability
        for i in range(0, len(columns), 10):
            print(columns[i:i+10])
        print("="*60 + "\n")
        
    except Exception as e:
        print(f">>> DATA LAKE ACCESS FAILED: {e}")
    finally:
        # Graceful shutdown of the Spark session
        spark.stop()

if __name__ == "__main__":
    main()
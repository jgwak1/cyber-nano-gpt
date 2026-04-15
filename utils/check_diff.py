import json

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BOTH_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\vocab_both_benign_malware.json"
BENIGN_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\vocab_only_benign.json"

def main():
    try:
        with open(BOTH_PATH, 'r', encoding='utf-8') as f:
            both_vocab = set(json.load(f).keys())
            
        with open(BENIGN_PATH, 'r', encoding='utf-8') as f:
            benign_vocab = set(json.load(f).keys())

        # 집합 연산 (Set Difference): 전체 - 정상 = 악성 고유 토큰
        malware_only_tokens = both_vocab - benign_vocab
        
        print(f">>> Identified {len(malware_only_tokens)} Malware-Only Tokens:")
        for token in sorted(list(malware_only_tokens)):
            print(f"    - {token}")
            
    except FileNotFoundError as e:
        print(f"File Error: {e}")

if __name__ == "__main__":
    main()



# =============================================================================
# DEPRECATED OUTPUT LOG
# =============================================================================
# The following output is outdated and no longer applicable.
# Reason: In the current architecture, 3_preprocess_pyspark.py enforces a strict 
# physical isolation of normal behavior (df_clean.filter(col("Label") == "Benign")). 
# Because the exported CSV partitions now contain exactly 0% malicious traffic, 
# building a vocabulary on these files inherently yields 0 malware-only tokens. 
# The 24 anomalous [D_U_RATIO_*] tokens logged below are artifacts from an earlier 
# pipeline version where malicious sequences were not physically excluded from the disk.

#(X) python3 C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\dev\check_diff.py       
#(X) >>> Identified 24 Malware-Only Tokens:
#     - [D_U_RATIO_22]
#     - [D_U_RATIO_23]
#     - [D_U_RATIO_24]
#     - [D_U_RATIO_25]
#     - [D_U_RATIO_26]
#     - [D_U_RATIO_27]
#     - [D_U_RATIO_28]
#     - [D_U_RATIO_29]
#     - [D_U_RATIO_30]
#     - [D_U_RATIO_31]
#     - [D_U_RATIO_32]
#     - [D_U_RATIO_34]
#     - [D_U_RATIO_35]
#     - [D_U_RATIO_36]
#     - [D_U_RATIO_37]
#     - [D_U_RATIO_38]
#     - [D_U_RATIO_39]
#     - [D_U_RATIO_40]
#     - [D_U_RATIO_41]
#     - [D_U_RATIO_42]
#     - [D_U_RATIO_45]
#     - [D_U_RATIO_51]
#     - [D_U_RATIO_53]
#     - [D_U_RATIO_55]
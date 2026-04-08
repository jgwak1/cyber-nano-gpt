# Engineering Log: Cyber-Nano-GPT

## 1. Architectural Decisions

### 1.1 Dataset Choice: CIC-IDS-2018
* **Decision**: Selected **CIC-IDS-2018** over UNSW-NB15 or other legacy datasets.
* **Rationale**: 
    * **Infrastructure Alignment**: It was generated within a real AWS EC2 environment, matching our S3 Data Lake and cloud-native MLOps architecture.
    * **Scalability**: The multi-GB volume necessitates a PySpark distributed ETL pipeline, ensuring horizontal scalability by decoupling processing from memory limits to support Terabyte-scale logs without refactoring.

### 1.2 Model Choice: Masked Self-Attention (GPT)
* **Decision**: Adopted a **Unidirectional Masked Self-Attention** (GPT-style) architecture.
* **Rationale**:
    * **Efficiency**: Parallel matrix operations enable $O(1)$ sequential complexity, outperforming LSTM's $O(N)$ bottlenecks.
    * **Long-term Dependency**: Self-attention provides a direct $O(1)$ access path to any historical token, preventing the information degradation common in recurrent models.

---

## 2. Data Engineering & Preprocessing

### 2.1 Feature Selection & Dimensionality Reduction
* **Decision**: Retained **19 core features** out of 80, dropping 61 to eliminate noise and bias.
* **Hierarchy**: Tokens follow a logical sequence: **Macro (Identifiers) → Volume → Velocity → State (Flags)** to guide the attention mechanism's focus.
* **Dropped Features**: Eliminated redundant "Subflow" metrics and biased identifiers like "Source IP" or "Timestamp" to prevent model over-fitting to specific environments.

### 2.2 Discretization & Overflow Protection
* **Log-Binning**: Continuous variables undergo $\lfloor \log_{10}(x+1) \rfloor$ transformation. This compresses the infinite numerical range into discrete tokens while preserving the magnitude of the signal.
* **The Hard Cap**: Applied `min(val, 30)` to all binned features.
* **Finding**: Identified $2^{63}-1$ (`Long.MAX_VALUE`) artifacts in `BYT_SEC` and `PKT_SEC` due to division-by-zero errors in the raw extractor (e.g., when Flow Duration is zero, the extractor assigns Long.MAX_VALUE to bypass the division-by-zero error). The hard cap prevents these artifacts from exhausting memory during embedding matrix allocation (i.e., it prevents a system crash by blocking an attempted impossible 9-quintillion-row matrix allocation).

### 2.3 Temporal Sequence Integrity (Hard Boundary Reset)
* **Context**: Preliminary audit revealed that while the dataset contains massive "Golden Windows" (up to 280k contiguous rows), a significant portion of the remaining data is interleaved with attack traffic.
* **Decision**: Implemented a **Hard Boundary Reset** to preserve the integrity of these large windows while strictly isolating them from fragmented sessions.
* **Mechanism**: Utilized state-based sequence segmentation to assign a unique `block_id` to contiguous 'Benign' states.
* **Caveats & Trade-offs**: 
    * **Data Sacrifice**: To guarantee 100% causal integrity, sub-batch fragments (< 13 rows) were intentionally dropped. 
    * **Result**: This trade-off prioritized sequence quality over raw volume, ensuring the Transformer never trains on "stitched" or broken temporal contexts.
* **Validation Metrics**: Verified 100% pure benign dataset containing 703,533 rows. Despite the reset, **81.54% of total benign data was preserved**, proving that the majority of the signal resides within large, high-integrity contiguous blocks.

---

## 3. Experimental Findings & Anomaly Logic

### 3.1 Point Anomaly (Vocabulary Isolation)
* **Finding**: Identified **24 Malware-exclusive tokens** in the `D_U_RATIO` feature, ranging from values **22 to 55**.
* **Significance**: These represent asymmetric C2 communication patterns where download volume significantly outweighs upload volume.
* **Strategy**: Compiled a **Strict Benign Vocabulary** (`vocab_only_benign.json`). These 24 tokens are mapped to `[UNK]` during inference, ensuring an immediate **Perplexity Spike**.

### 3.2 Contextual Anomaly (The "Grammar" of Attacks)
* **Logic**: While point anomalies catch obvious traces, most attacks use benign tokens in abnormal sequences or frequencies.
* **Mechanism**: The GPT model is trained to learn the conditional probability $P(t \mid \text{history})$ of normal traffic. 
* **Detection**: Any irregular pattern—even if composed of known tokens—results in a high **Negative Log-Likelihood (Loss)**. This mathematical "surprise" allows the detection of Zero-day attacks based on context rather than signatures.

---
*Documented on 2026-03-24*
# Cyber-Nano-GPT (MLOps Edition)

An End-to-End MLOps pipeline for a custom Cyber-Security Foundation Model.

## Project Goal
To engineer a production-grade **Anomaly Detection Pipeline** capable of scaling from local training to cloud deployment. This project demonstrates the full **Machine Learning Lifecycle (MLOps)**: Data Versioning, CI/CD, Model Registry, and Serverless Serving.

### The Objective: "The Probability Engine"
The goal is NOT to build a chatbot. The goal is to build a **Conditional Probability Estimator** that calculates the likelihood of every token in a log stream.
* **Low Loss:** Normal traffic (e.g., standard admin logins).
* **High Loss:** Anomalous patterns (e.g., obfuscated PowerShell, SQL injection) that deviate from learned syntax.

---

## Architecture: The "Guard & Detective" Pattern


### 1. The Guard (Nano-GPT)
* **Role:** Real-time Anomaly Detection (Sidecar).
* **Function:** Assigns a "Perplexity Score" to live logs. High perplexity triggers The Detective.

### 2. The Detective (LangChain + RAG)
* **Role:** Incident Investigation.
* **Function:** Retrieves Threat Intel (MITRE ATT&CK) via RAG to explain *why* the Nano-GPT flagged the log.

---

## Core Logic: How It Detects Attacks
Unlike classifiers that look for specific keywords, this model uses **Next-Token Prediction** to detect anomalies based on *mathematical surprise*.

**Scenario:** A Hacker attempts SQL Injection: `GET /login?user=' OR 1=1`

| Step | Context | Model Expectation | Actual Token | Loss (Surprise) | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `GET /login?` | `id`, `user` | `user` | **0.01** (Low) | Normal |
| 2 | `?user=` | `[string]` | `'` | **6.90** (High) | Quotes rare here |
| 3 | `?user='` | ` ` (Space) | `OR` | **9.21** (Massive) | Unexpected SQL |

---

## Technical Architecture

### 1. Core Implementation & Config
* **First Principles:** Self-Attention implemented manually in **NumPy** (Linear Algebra verification) before PyTorch.
* **Model:** 60M Parameters ("Nano"), 10 Layers, 8 Heads, 512 Embed Dim.
* **Hardware:** Optimized for Consumer GPUs (RTX 4000 - 8GB VRAM) using FP16 Mixed Precision.

### 2. Scalability & Distributed Training
* **Cloud-Agnostic:** Containerized (Docker) for AWS/GCP execution.
* **Distributed:** Training loop utilizes **Hugging Face Accelerate** and **DDP** (Distributed Data Parallel) to scale across multiple GPUs.

### 3. MLOps Infrastructure
* **Experiment Tracking:** MLflow (Loss curves).
* **Data Versioning:** DVC (S3 Backend).
* **CI/CD:** GitHub Actions (Unit Tests & Container Builds).
* **Serving:** FastAPI wrapped in Docker (AWS Lambda compatible).

---

## Directory Structure

The repository organizes the progression from **First Principles** to **Production Cloud Infrastructure**:

```text
cyber-nano-gpt/
├── .github/workflows/           # CI/CD Pipelines (Test & Build)
├── data/                        # Real-world datasets (LogHub, EVTX, Atomic Red Team)
├── deploy/                      # Dockerfiles & Terraform (Infrastructure as Code)
├── scripts/                     # Data pipelines, Tokenizer training, Utilities
├── src/
│   ├── api/                     # FastAPI Inference Server (The "Guard")
│   ├── model/                   # Production PyTorch Transformer architecture
│   ├── rag/                     # LangChain/VectorDB Agent (The "Detective")
│   ├── scratch_implementations/ # Educational: Manual NumPy & Raw PyTorch, Tensorflow attention
│   ├── training/                # Training loops (Pre-training & SFT)
│   └── rl/                      # Alignment: DPO implementation
└── tests/                       # Unit tests for shape verification
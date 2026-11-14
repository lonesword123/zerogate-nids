# ZEROGATE — Uncertainty-Aware Network Intrusion Detection System (NIDS)

**ZeroGate** is an open-set, uncertainty-aware Network Intrusion Detection System built on **PyTorch**, **FastAPI**, and **Split Conformal Prediction**. Unlike standard closed-world NIDS classifiers that silently misclassify unseen Zero-Day attacks as BENIGN with 99%+ Softmax confidence, ZeroGate quantifies epistemic doubt and local manifold density to **abstain and quarantine** novel attack vectors for 1-click active learning triage.

---

## 1. Core Architecture (ZeroGate v4)

1. **55-Feature Protocol & Flow Representation (scripts/preprocess_cicids.py)**:
   - Extracts 54 bidirectional flow features from **CIC-IDS2017** plus **Empirical Destination Port Self-Information** (-log10 P_train(Port)) and signed log(1+|x|) compression.
2. **Spectral-Normalized Bi-Lipschitz Encoder (backend/model.py)**:
   - Bounds the upper and lower Lipschitz constants via Spectral Normalization, residual skip connections, and a 12-D bottleneck autoencoder to prevent unseen Zero-Day features from collapsing onto the BENIGN manifold.
3. **GPU Local-Density Relative Manifold Distance (D_rel)**:
   - Normalizes each test flow's k-NN distance against the local neighborhood spacing of 20,000 reference In-Distribution prototypes so both dense HTTP and sparse UDP protocols calibrate to D_rel ~ 0.63 (95th percentile threshold D_rel = 1.42).
4. **Split Conformal Credibility (p-value)**:
   - Provides distribution-free finite-sample false-alarm guarantees (alpha = 0.05, empirical ID coverage 94.68%).

---

## 2. CIC-IDS2017 Held-Out Zero-Day Benchmark (logs/benchmark_results.json)

Unseen attack families (Bot, Heartbleed, Infiltration, Web Attack - Brute Force, SQL Injection, XSS) were strictly held out from training (n = 4,183 OOD flows vs. 12,000 ID test flows):

| Metric | Standard Closed-World NIDS | ZeroGate v4 (Ours) |
| :--- | :---: | :---: |
| **In-Distribution Accuracy (Benign vs. Known)** | 99.67% | **99.67%** |
| **Conformal ID Coverage (Target >= 95.0%)** | N/A | **94.68%** |
| **Zero-Day OOD AUROC (Higher=Better)** | 73.64% | **90.86%** |
| **FPR @ 95% TPR (Lower=Better)** | 57.16% | **34.61%** |
| **0-Shot Zero-Day Quarantine Rate (p < 0.05)** | 3.08% | **66.51%** |

### Per-Zero-Day Family Breakdown (0-Shot Unseen Detection)
* **Heartbleed (n=11)**: **100.00%** caught by ZeroGate (100.00% fooled as Benign in Closed-World)
* **Web Attack — XSS (n=652)**: **97.70%** caught by ZeroGate (96.01% fooled as Benign in Closed-World)
* **Infiltration (n=36)**: **97.22%** caught by ZeroGate (100.00% fooled as Benign in Closed-World)
* **Web Attack — Brute Force (n=1507)**: **90.98%** caught by ZeroGate (96.95% fooled as Benign in Closed-World)
* **Web Attack — SQL Injection (n=21)**: **71.43%** caught by ZeroGate (71.43% fooled as Benign in Closed-World)
* **Bot (n=1956)**: **36.45%** caught 0-shot (100.00% fooled as Benign in Closed-World)

---

## 3. Quick Start Commands

    # 1. Activate virtual environment
    source venv/bin/activate

    # 2. Preprocess CIC-IDS2017 CSVs & train ZeroGate v4
    python3 scripts/preprocess_cicids.py
    python3 -m scripts.train_zerogate

    # 3. Run the evaluation & active-learning benchmarks
    python3 -m experiments.evaluate
    python3 -m experiments.active_learning_eval

    # 4. Launch the unified FastAPI + SOC Telemetry Dashboard on http://localhost:8000
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

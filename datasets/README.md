# CIC-IDS2017 Dataset Structure

Raw CSV files from the **Canadian Institute for Cybersecurity (CIC-IDS2017)** benchmark are excluded from Git tracking due to GitHub's 100 MB file-size limit.

## 1. Raw Traffic Captures Used (`datasets/raw/`)
- `Monday-WorkingHours.pcap_ISCX.csv` (Benign Baseline)
- `Tuesday-WorkingHours.pcap_ISCX.csv` (Brute Force)
- `Wednesday-workingHours.pcap_ISCX.csv` (DoS / Heartbleed)
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Zero-Day Holdout: Web Attack Brute Force, XSS, SQLi)
- `Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv` (Zero-Day Holdout: Infiltration)
- `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Zero-Day Holdout: Botnet ARES)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (Known Attack: PortScan)
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (Known Attack: DDoS LOIC)

## 2. Processed Artifacts Included (`datasets/processed/`)
- `sample_streams.json`: Real held-out test flows (Benign, Known Attack, and Unseen Zero-Day) served to the live FastAPI telemetry dashboard.
- `zerogate_splits.npz`: Generated locally via `python3 scripts/preprocess_cicids.py` (56,000 Train ID | 12,000 Calibration ID | 12,000 Test ID | 4,183 Held-Out Zero-Day OOD flows).

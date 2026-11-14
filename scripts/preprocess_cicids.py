import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RAW_DIR = "datasets/raw"
PROCESSED_DIR = "datasets/processed"
MODELS_DIR = "models"

SELECTED_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Idle Mean",
]

ZERODAY_KEYWORDS = ["Bot", "Infiltration", "Heartbleed", "Web Attack", "Sql Injection", "XSS"]


def is_zeroday_label(label_str: str) -> bool:
    l_lower = str(label_str).strip().lower()
    return any(kw.lower() in l_lower for kw in ZERODAY_KEYWORDS)


def signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


def compute_port_surprise(ports: np.ndarray, port_prob_map: dict, default_prob: float) -> np.ndarray:
    probs = np.array([port_prob_map.get(int(round(p)), default_prob) for p in ports], dtype=np.float32)
    return (-np.log10(probs)).reshape(-1, 1)


def load_real_cicids():
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {RAW_DIR}/")

    dfs = []
    for path in csv_files:
        fname = os.path.basename(path)
        print(f"[*] Reading {fname}...")
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            continue

        available = [c for c in SELECTED_FEATURES if c in df.columns]
        df = df[available + ["Label"]]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=available + ["Label"])

        labels_clean = df["Label"].astype(str).str.strip()
        is_benign = labels_clean.str.upper() == "BENIGN"
        is_zd = labels_clean.apply(is_zeroday_label)
        is_known_atk = (~is_benign) & (~is_zd)

        df_zd = df[is_zd]
        df_known = df[is_known_atk]
        df_benign = df[is_benign]

        if len(df_known) > 15000:
            df_known = df_known.sample(n=15000, random_state=42)
        if len(df_benign) > 15000:
            df_benign = df_benign.sample(n=15000, random_state=42)

        dfs.append(pd.concat([df_benign, df_known, df_zd], ignore_index=True))

    return pd.concat(dfs, ignore_index=True)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_real_cicids()
    feature_cols = [c for c in SELECTED_FEATURES if c in df.columns]

    def map_category(label_str):
        l_str = str(label_str).strip()
        if l_str.upper() == "BENIGN":
            return 0
        if is_zeroday_label(l_str):
            return 2
        return 1

    df["Category"] = df["Label"].apply(map_category)

    id_df = df[df["Category"].isin([0, 1])].copy()
    ood_df = df[df["Category"] == 2].copy()

    if len(id_df) > 80000:
        id_df = id_df.sample(n=80000, random_state=42)
    if len(ood_df) > 10000:
        ood_df = ood_df.sample(n=10000, random_state=42)

    X_id = id_df[feature_cols].values.astype(np.float32)
    y_id = id_df["Category"].values.astype(np.int64)
    X_ood = ood_df[feature_cols].values.astype(np.float32)
    raw_ood_labels = ood_df["Label"].astype(str).str.strip().values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_id, y_id, test_size=0.30, random_state=42, stratify=y_id
    )
    X_cal, X_test_id, y_cal, y_test_id = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Fit Empirical Destination Port Self-Information strictly on X_train
    train_ports = np.round(X_train[:, 0]).astype(np.int64)
    unique_ports, counts = np.unique(train_ports, return_counts=True)
    total_train = float(len(train_ports))
    port_prob_map = {int(p): float((c + 1.0) / (total_train + 65536.0)) for p, c in zip(unique_ports, counts)}
    default_prob = float(1.0 / (total_train + 65536.0))

    def augment_and_log(X_raw: np.ndarray) -> np.ndarray:
        port_s = compute_port_surprise(X_raw[:, 0], port_prob_map, default_prob)
        X_aug = np.hstack([X_raw, port_s])
        return signed_log1p(X_aug)

    X_train_log = augment_and_log(X_train)
    X_cal_log = augment_and_log(X_cal)
    X_test_id_log = augment_and_log(X_test_id)
    X_ood_log = augment_and_log(X_ood)

    scaler = RobustScaler()
    X_train_scaled = np.clip(scaler.fit_transform(X_train_log), -15.0, 15.0).astype(np.float32)
    X_cal_scaled = np.clip(scaler.transform(X_cal_log), -15.0, 15.0).astype(np.float32)
    X_test_id_scaled = np.clip(scaler.transform(X_test_id_log), -15.0, 15.0).astype(np.float32)
    X_ood_scaled = np.clip(scaler.transform(X_ood_log), -15.0, 15.0).astype(np.float32)

    final_feature_names = feature_cols + ["Protocol Port Surprise"]

    joblib.dump(
        {
            "scaler": scaler,
            "port_prob_map": port_prob_map,
            "default_prob": default_prob,
            "raw_feature_names": feature_cols,
            "feature_names": final_feature_names,
        },
        os.path.join(MODELS_DIR, "scaler.joblib"),
    )
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(final_feature_names, f, indent=2)

    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "zerogate_splits.npz"),
        X_train=X_train_scaled,
        y_train=y_train,
        X_cal=X_cal_scaled,
        y_cal=y_cal,
        X_test_id=X_test_id_scaled,
        y_test_id=y_test_id,
        X_ood=X_ood_scaled,
        ood_labels=raw_ood_labels,
    )

    stream_payload = {
        "feature_names": feature_cols,
        "normal": X_test_id[y_test_id == 0][:50].tolist(),
        "known_attack": X_test_id[y_test_id == 1][:50].tolist(),
        "zeroday": X_ood[:50].tolist(),
        "zeroday_labels": raw_ood_labels[:50].tolist(),
    }
    with open(os.path.join(PROCESSED_DIR, "sample_streams.json"), "w") as f:
        json.dump(stream_payload, f)

    print(f"\n[+] Preprocessing complete with {len(final_feature_names)} features (including Training Port Surprise)!")
    print(f"    Train ID: {len(X_train_scaled)} | Calib ID: {len(X_cal_scaled)} | Zero-Day OOD: {len(X_ood_scaled)}")


if __name__ == "__main__":
    main()

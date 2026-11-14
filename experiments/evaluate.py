import os
import json
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from backend.model import ZeroGateNet, compute_relative_manifold_distance

PROCESSED_PATH = "datasets/processed/zerogate_splits.npz"
MODELS_DIR = "models"
LOGS_DIR = "logs"


def batched_inference(model, X: np.ndarray, device: torch.device, batch_size: int = 512, mc_passes: int = 20):
    cw_probs_list, ep_list, al_list, recon_list, z_list = [], [], [], [], []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32).to(device)
        out = model.forward_uncertainty(batch, mc_passes=mc_passes)
        cw_probs_list.append(out["closed_world_probs"])
        ep_list.append(out["epistemic_raw"])
        al_list.append(out["aleatoric_raw"])
        recon_list.append(out["recon_error"])
        z_list.append(out["latent_z"])

    return {
        "closed_world_probs": np.concatenate(cw_probs_list, axis=0),
        "epistemic_raw": np.concatenate(ep_list, axis=0),
        "aleatoric_raw": np.concatenate(al_list, axis=0),
        "recon_error": np.concatenate(recon_list, axis=0),
        "latent_z": np.concatenate(z_list, axis=0),
    }


def compute_fpr_at_95_tpr(y_true_ood: np.ndarray, anomaly_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true_ood, anomaly_scores)
    idx = np.searchsorted(tpr, 0.95, side="left")
    return float(fpr[min(idx, len(fpr) - 1)])


def compute_conformal_pvalues(cal_scores_sorted: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    idxs = np.searchsorted(cal_scores_sorted, test_scores, side="left")
    n_greater_equal = len(cal_scores_sorted) - idxs
    return (n_greater_equal + 1.0) / (len(cal_scores_sorted) + 1.0)


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(PROCESSED_PATH, allow_pickle=True)
    X_test_id, y_test_id = data["X_test_id"], data["y_test_id"]
    X_ood, ood_labels = data["X_ood"], data["ood_labels"]

    bundle = joblib.load(os.path.join(MODELS_DIR, "calibration_bundle.joblib"))
    model = ZeroGateNet(
        in_features=bundle["in_features"],
        latent_dim=bundle.get("latent_dim", 24),
        num_classes=2,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "zerogate_net.pt"), map_location=device))
    model.eval()

    print(f"[*] Running batched MC-Dropout + GPU Relative Manifold evaluation ({device})...")
    print(f"    ID Test Flows : {len(X_test_id)} | Unseen Zero-Day Flows : {len(X_ood)}")

    id_out = batched_inference(model, X_test_id, device, mc_passes=20)
    ood_out = batched_inference(model, X_ood, device, mc_passes=20)

    id_preds = np.argmax(id_out["closed_world_probs"], axis=1)
    id_acc = accuracy_score(y_test_id, id_preds) * 100.0

    cw_id_anomaly = 1.0 - np.max(id_out["closed_world_probs"], axis=1)
    cw_ood_anomaly = 1.0 - np.max(ood_out["closed_world_probs"], axis=1)

    silent_failures = np.mean(
        (np.argmax(ood_out["closed_world_probs"], axis=1) == 0)
        & (np.max(ood_out["closed_world_probs"], axis=1) >= 0.90)
    ) * 100.0

    id_maha = compute_relative_manifold_distance(
        X_test_id,
        id_out["latent_z"],
        bundle["X_ref"],
        bundle["Z_ref"],
        bundle["ref_radius_x"],
        bundle["ref_radius_z"],
        device=device,
        k=5,
    )
    ood_maha = compute_relative_manifold_distance(
        X_ood,
        ood_out["latent_z"],
        bundle["X_ref"],
        bundle["Z_ref"],
        bundle["ref_radius_x"],
        bundle["ref_radius_z"],
        device=device,
        k=5,
    )

    zg_id_score = (
        (id_maha / bundle["maha_scale"])
        + 0.35 * (id_out["recon_error"] / bundle["recon_scale"])
        + 0.20 * (id_out["epistemic_raw"] / bundle["ep_scale"])
    )
    zg_ood_score = (
        (ood_maha / bundle["maha_scale"])
        + 0.35 * (ood_out["recon_error"] / bundle["recon_scale"])
        + 0.20 * (ood_out["epistemic_raw"] / bundle["ep_scale"])
    )

    y_binary = np.concatenate([np.zeros(len(X_test_id)), np.ones(len(X_ood))])
    cw_scores_all = np.concatenate([cw_id_anomaly, cw_ood_anomaly])
    zg_scores_all = np.concatenate([zg_id_score, zg_ood_score])

    cw_auroc = roc_auc_score(y_binary, cw_scores_all) * 100.0
    zg_auroc = roc_auc_score(y_binary, zg_scores_all) * 100.0

    cw_fpr95 = compute_fpr_at_95_tpr(y_binary, cw_scores_all) * 100.0
    zg_fpr95 = compute_fpr_at_95_tpr(y_binary, zg_scores_all) * 100.0

    alpha = 0.05
    id_pvals = compute_conformal_pvalues(bundle["cal_nonconformity"], zg_id_score)
    ood_pvals = compute_conformal_pvalues(bundle["cal_nonconformity"], zg_ood_score)

    id_coverage = np.mean(id_pvals >= alpha) * 100.0
    ood_catch_rate = np.mean(ood_pvals < alpha) * 100.0

    per_class_report = {}
    for cls_name in sorted(np.unique(ood_labels)):
        mask = ood_labels == cls_name
        cls_caught = float(np.mean(ood_pvals[mask] < alpha) * 100.0)
        cls_maha = float(np.median(ood_maha[mask]))
        cls_cw_fooled = float(np.mean(np.argmax(ood_out["closed_world_probs"][mask], axis=1) == 0) * 100.0)
        per_class_report[str(cls_name)] = {
            "count": int(mask.sum()),
            "zerogate_catch_rate_pct": round(cls_caught, 2),
            "median_mahalanobis": round(cls_maha, 2),
            "closed_world_fooled_as_benign_pct": round(cls_cw_fooled, 2),
        }

    print("\n" + "=" * 68)
    print(" ZEROGATE v4 BENCHMARK REPORT (RELATIVE MANIFOLD + PORT SURPRISE)")
    print("=" * 68)
    print(f" In-Distribution Accuracy (Benign vs Known) : {id_acc:.2f}%")
    print(f" Conformal Coverage on ID (Target >= 95.0%) : {id_coverage:.2f}%")
    print("-" * 68)
    print(f" {'Metric':<36} | {'Closed-World':<12} | {'ZeroGate v4':<12}")
    print("-" * 68)
    print(f" {'Zero-Day OOD AUROC (Higher=Better)':<36} | {cw_auroc:>10.2f}% | {zg_auroc:>10.2f}%")
    print(f" {'FPR @ 95% TPR (Lower=Better)':<36} | {cw_fpr95:>10.2f}% | {zg_fpr95:>10.2f}%")
    print(f" {'Zero-Day Quarantine Rate (p < 0.05)':<36} | {100.0 - silent_failures:>10.2f}% | {ood_catch_rate:>10.2f}%")
    print("=" * 68)

    print("\n[*] Per-Zero-Day Family Breakdown (Unseen During Training):")
    for cls_name, stats in per_class_report.items():
        clean_name = "".join(ch if ord(ch) < 128 else "-" for ch in str(cls_name))
        print(
            f"  - {clean_name:<28} (n={stats['count']:<4}) | "
            f"ZeroGate Catch: {stats['zerogate_catch_rate_pct']:>6.2f}% | "
            f"Median Dm: {stats['median_mahalanobis']:>5.2f} | "
            f"CW Fooled as Benign: {stats['closed_world_fooled_as_benign_pct']:>6.2f}%"
        )

    out_file = os.path.join(LOGS_DIR, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "in_distribution_accuracy_pct": round(id_acc, 2),
                "conformal_id_coverage_pct": round(id_coverage, 2),
                "closed_world_auroc_pct": round(cw_auroc, 2),
                "zerogate_auroc_pct": round(zg_auroc, 2),
                "zeroday_quarantine_rate_pct": round(ood_catch_rate, 2),
                "per_zeroday_class": per_class_report,
            },
            f,
            indent=2,
        )
    print(f"\n[+] Full benchmark metrics saved to {out_file}")


if __name__ == "__main__":
    main()

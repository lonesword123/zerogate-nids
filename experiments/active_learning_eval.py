import os
import json
import joblib
import numpy as np
import torch
from backend.model import ZeroGateNet, compute_relative_manifold_distance

PROCESSED_PATH = "datasets/processed/zerogate_splits.npz"
MODELS_DIR = "models"
LOGS_DIR = "logs"


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(PROCESSED_PATH, allow_pickle=True)
    X_ood, ood_labels = data["X_ood"], data["ood_labels"]

    bundle = joblib.load(os.path.join(MODELS_DIR, "calibration_bundle.joblib"))
    model = ZeroGateNet(
        in_features=bundle["in_features"],
        latent_dim=bundle.get("latent_dim", 24),
        num_classes=2,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "zerogate_net.pt"), map_location=device))
    model.eval()

    # 1. Baseline 0-Shot Evaluation on all 4,183 Zero-Day Flows
    z_list, recon_list, ep_list = [], [], []
    for start in range(0, len(X_ood), 512):
        batch = torch.tensor(X_ood[start : start + 512], dtype=torch.float32).to(device)
        out = model.forward_uncertainty(batch, mc_passes=20)
        z_list.append(out["latent_z"])
        recon_list.append(out["recon_error"])
        ep_list.append(out["epistemic_raw"])

    Z_ood = np.concatenate(z_list, axis=0)
    recon_ood = np.concatenate(recon_list, axis=0)
    ep_ood = np.concatenate(ep_list, axis=0)

    ood_maha = compute_relative_manifold_distance(
        X_ood,
        Z_ood,
        bundle["X_ref"],
        bundle["Z_ref"],
        bundle["ref_radius_x"],
        bundle["ref_radius_z"],
        device=device,
        k=5,
    )

    zg_ood_score = (
        (ood_maha / bundle["maha_scale"])
        + 0.35 * (recon_ood / bundle["recon_scale"])
        + 0.20 * (ep_ood / bundle["ep_scale"])
    )

    cal_scores = bundle["cal_nonconformity"]
    idxs = np.searchsorted(cal_scores, zg_ood_score, side="left")
    ood_pvals = (len(cal_scores) - idxs + 1.0) / (len(cal_scores) + 1.0)
    zero_shot_caught = ood_pvals < 0.05

    print("\n" + "=" * 72)
    print(" ZEROGATE ACTIVE LEARNING CLOSED-LOOP EVALUATION (POST-TRIAGE PATCHING)")
    print("=" * 72)
    print(f" 0-Shot Quarantine Rate (Before Analyst Patching) : {np.mean(zero_shot_caught)*100:.2f}%")
    print("-" * 72)
    print(f" {'Analyst Patches per Family':<28} | {'Bot Catch %':<14} | {'Total Zero-Day Catch %'}")
    print("-" * 72)

    X_ood_t = torch.tensor(X_ood, dtype=torch.float32, device=device)
    Z_ood_t = torch.tensor(Z_ood, dtype=torch.float32, device=device)

    rng = np.random.default_rng(42)
    unique_families = np.unique(ood_labels)
    results_curve = {}

    for k_patches in [0, 1, 3, 5, 10]:
        if k_patches == 0:
            bot_mask = np.array(["Bot" in str(lbl) for lbl in ood_labels])
            bot_rate = float(np.mean(zero_shot_caught[bot_mask]) * 100.0)
            total_rate = float(np.mean(zero_shot_caught) * 100.0)
        else:
            patched_x, patched_z = [], []
            for fam in unique_families:
                fam_quarantined_idx = np.where((ood_labels == fam) & zero_shot_caught)[0]
                if len(fam_quarantined_idx) == 0:
                    fam_quarantined_idx = np.where(ood_labels == fam)[0]
                chosen = rng.choice(
                    fam_quarantined_idx,
                    size=min(k_patches, len(fam_quarantined_idx)),
                    replace=False,
                )
                patched_x.append(X_ood[chosen])
                patched_z.append(Z_ood[chosen])

            sig_x = torch.tensor(np.vstack(patched_x), dtype=torch.float32, device=device)
            sig_z = torch.tensor(np.vstack(patched_z), dtype=torch.float32, device=device)

            # Compute distance from all OOD flows to newly patched Zero-Day signatures
            dist_sig_x = torch.min(torch.cdist(X_ood_t, sig_x, p=2.0), dim=1).values.cpu().numpy()
            dist_sig_z = torch.min(torch.cdist(Z_ood_t, sig_z, p=2.0), dim=1).values.cpu().numpy()
            matched_new_signature = (0.5 * dist_sig_x + 0.5 * dist_sig_z) < 2.2

            combined_caught = zero_shot_caught | matched_new_signature
            bot_mask = np.array(["Bot" in str(lbl) for lbl in ood_labels])
            bot_rate = float(np.mean(combined_caught[bot_mask]) * 100.0)
            total_rate = float(np.mean(combined_caught) * 100.0)

        results_curve[f"{k_patches}_patches"] = {
            "bot_catch_pct": round(bot_rate, 2),
            "total_zeroday_catch_pct": round(total_rate, 2),
        }
        print(f" {f'{k_patches} Patched Flows':<28} | {bot_rate:>12.2f}% | {total_rate:>20.2f}%")

    print("=" * 72)

    out_path = os.path.join(LOGS_DIR, "active_learning_results.json")
    with open(out_path, "w") as f:
        json.dump(results_curve, f, indent=2)
    print(f"[+] Saved active learning progression to {out_path}\n")


if __name__ == "__main__":
    main()

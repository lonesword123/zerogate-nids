import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from backend.model import ZeroGateNet, compute_relative_manifold_distance

PROCESSED_PATH = "datasets/processed/zerogate_splits.npz"
MODELS_DIR = "models"


@torch.no_grad()
def build_reference_manifold_bank(X_train: np.ndarray, Z_train: np.ndarray, device: torch.device, max_ref: int = 20000, k: int = 5):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(max_ref, len(X_train)), replace=False)
    X_ref = X_train[idx].astype(np.float32)
    Z_ref = Z_train[idx].astype(np.float32)

    X_ref_t = torch.tensor(X_ref, dtype=torch.float32, device=device)
    Z_ref_t = torch.tensor(Z_ref, dtype=torch.float32, device=device)

    rad_x_list, rad_z_list = [], []
    batch_size = 512
    for start in range(0, len(X_ref), batch_size):
        xb = X_ref_t[start : start + batch_size]
        zb = Z_ref_t[start : start + batch_size]

        dx = torch.cdist(xb, X_ref_t, p=2.0)
        # k+1 because the 1st nearest neighbor in the reference bank is the point itself (distance 0)
        knn_x, _ = torch.topk(dx, k=k + 1, largest=False, dim=1)
        rad_x_list.append(torch.mean(knn_x[:, 1:], dim=1).cpu().numpy())

        dz = torch.cdist(zb, Z_ref_t, p=2.0)
        knn_z, _ = torch.topk(dz, k=k + 1, largest=False, dim=1)
        rad_z_list.append(torch.mean(knn_z[:, 1:], dim=1).cpu().numpy())

    ref_radius_x = np.concatenate(rad_x_list, axis=0)
    ref_radius_z = np.concatenate(rad_z_list, axis=0)
    return X_ref, Z_ref, ref_radius_x, ref_radius_z


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    data = np.load(PROCESSED_PATH, allow_pickle=True)

    X_train, y_train = data["X_train"], data["y_train"]
    X_cal, y_cal = data["X_cal"], data["y_cal"]
    X_ood = data["X_ood"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = X_train.shape[1]

    model = ZeroGateNet(in_features=in_features, latent_dim=24, num_classes=2, dropout_p=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=256,
        shuffle=True,
    )

    print(f"[*] Training ZeroGate v4 (Local-Density Manifold + Port Surprise) on {len(X_train)} flows ({device})...")
    model.train()
    for epoch in range(1, 16):
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _, recon = model(bx)
            # Reconstruct ALL In-Distribution flows (both Benign and Known Attacks)
            loss = ce_loss(logits, by) + 0.8 * mse_loss(recon, bx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            correct += (logits.argmax(dim=1) == by).sum().item()
            total += bx.size(0)

        if epoch in (1, 5, 10, 15):
            print(f"    Epoch {epoch:02d}/15 | Loss: {total_loss/total:.4f} | ID Acc: {100.0*correct/total:.2f}%")

    model.eval()
    z_train_list = []
    with torch.no_grad():
        for start in range(0, len(X_train), 1024):
            _, zb, _ = model(torch.tensor(X_train[start : start + 1024]).to(device))
            z_train_list.append(zb.cpu().numpy())
    Z_train = np.concatenate(z_train_list, axis=0)

    print("[*] Building GPU Reference Manifold Bank (20,000 ID Prototypes with Local Radii)...")
    X_ref, Z_ref, ref_radius_x, ref_radius_z = build_reference_manifold_bank(
        X_train, Z_train, device=device, max_ref=20000, k=5
    )

    cal_out = model.forward_uncertainty(torch.tensor(X_cal).to(device), mc_passes=20)
    cal_maha = compute_relative_manifold_distance(
        X_cal, cal_out["latent_z"], X_ref, Z_ref, ref_radius_x, ref_radius_z, device=device, k=5
    )

    ood_out = model.forward_uncertainty(torch.tensor(X_ood).to(device), mc_passes=20)
    ood_maha = compute_relative_manifold_distance(
        X_ood, ood_out["latent_z"], X_ref, Z_ref, ref_radius_x, ref_radius_z, device=device, k=5
    )

    ep_scale = float(np.percentile(cal_out["epistemic_raw"], 95) + 1e-5)
    maha_scale = float(np.percentile(cal_maha, 95) + 1e-5)
    recon_scale = float(np.percentile(cal_out["recon_error"], 95) + 1e-5)

    cal_nonconformity = (
        (cal_maha / maha_scale)
        + 0.35 * (cal_out["recon_error"] / recon_scale)
        + 0.20 * (cal_out["epistemic_raw"] / ep_scale)
    )

    print(f"    [Calibration ID] Median Relative Dist: {np.median(cal_maha):.2f} | 95th Pct: {maha_scale:.2f}")
    print(f"    [Unseen Zero-Day] Median Relative Dist: {np.median(ood_maha):.2f} | Max: {np.max(ood_maha):.2f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "zerogate_net.pt"))
    calibration_bundle = {
        "in_features": in_features,
        "latent_dim": 24,
        "X_ref": X_ref,
        "Z_ref": Z_ref,
        "ref_radius_x": ref_radius_x,
        "ref_radius_z": ref_radius_z,
        "cal_nonconformity": np.sort(cal_nonconformity),
        "ep_scale": ep_scale,
        "maha_scale": maha_scale,
        "recon_scale": recon_scale,
    }
    joblib.dump(calibration_bundle, os.path.join(MODELS_DIR, "calibration_bundle.joblib"))
    print(f"[+] Saved ZeroGate v4 Reference Manifold bundle to {MODELS_DIR}/")


if __name__ == "__main__":
    main()

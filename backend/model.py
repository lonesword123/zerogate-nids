import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np


class ZeroGateNet(nn.Module):
    """
    ZeroGate v4:
    - Spectral-Normalized Encoder extracts 24-D latent vector z
    - In-Distribution Autoencoder trained on all ID classes (Benign + Known Attacks)
    - GPU-accelerated Local-Density Relative Manifold Distance (k-NN LOF ratio)
    """
    def __init__(self, in_features: int, latent_dim: int = 24, num_classes: int = 2, dropout_p: float = 0.2):
        super().__init__()
        self.input_proj = spectral_norm(nn.Linear(in_features, 128))
        self.block1 = nn.Sequential(
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            spectral_norm(nn.Linear(128, 128)),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            spectral_norm(nn.Linear(128, latent_dim)),
        )
        self.skip_proj = spectral_norm(nn.Linear(in_features, latent_dim))

        self.id_ae = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.SiLU(),
            nn.Linear(32, 12),
            nn.SiLU(),
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Linear(32, in_features),
        )

        self.dropout_head = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.block1(h)
        z = self.block2(h) + 0.2 * self.skip_proj(x)
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        logits = self.classifier(self.dropout_head(z))
        recon = self.id_ae(x)
        return logits, z, recon

    @torch.no_grad()
    def forward_uncertainty(self, x: torch.Tensor, mc_passes: int = 20):
        self.eval()
        std_logits, z_det, recon_det = self.forward(x)
        closed_world_probs = F.softmax(std_logits, dim=-1)

        per_feat_sq = (x - recon_det) ** 2
        top_k_sq, _ = torch.topk(per_feat_sq, k=min(8, x.shape[1]), dim=-1)
        recon_error = torch.mean(top_k_sq, dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        mc_probs = []
        for _ in range(mc_passes):
            logits, _, _ = self.forward(x)
            mc_probs.append(F.softmax(logits, dim=-1))

        mc_probs = torch.stack(mc_probs, dim=0)
        mean_probs = mc_probs.mean(dim=0)

        eps = 1e-8
        total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
        aleatoric_entropy = -torch.mean(torch.sum(mc_probs * torch.log(mc_probs + eps), dim=-1), dim=0)
        epistemic_mi = torch.clamp(total_entropy - aleatoric_entropy, min=0.0)

        self.eval()
        return {
            "closed_world_probs": closed_world_probs.cpu().numpy(),
            "mean_probs": mean_probs.cpu().numpy(),
            "epistemic_raw": epistemic_mi.cpu().numpy(),
            "aleatoric_raw": aleatoric_entropy.cpu().numpy(),
            "recon_error": recon_error.cpu().numpy(),
            "latent_z": z_det.cpu().numpy(),
        }


@torch.no_grad()
def compute_relative_manifold_distance(
    X_query: np.ndarray,
    Z_query: np.ndarray,
    X_ref: np.ndarray,
    Z_ref: np.ndarray,
    ref_radius_x: np.ndarray,
    ref_radius_z: np.ndarray,
    device: torch.device,
    k: int = 5,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Computes Local-Density Normalized Manifold Distance on GPU:
    Divides each query flow's k-NN distance to X_ref/Z_ref by the local neighborhood
    spacing of those matched reference points so dense HTTP and sparse UDP normalize to ~1.0.
    """
    X_ref_t = torch.tensor(X_ref, dtype=torch.float32, device=device)
    Z_ref_t = torch.tensor(Z_ref, dtype=torch.float32, device=device)
    rad_x_t = torch.tensor(ref_radius_x, dtype=torch.float32, device=device)
    rad_z_t = torch.tensor(ref_radius_z, dtype=torch.float32, device=device)

    rel_dists = []
    for start in range(0, len(X_query), batch_size):
        xb = torch.tensor(X_query[start : start + batch_size], dtype=torch.float32, device=device)
        zb = torch.tensor(Z_query[start : start + batch_size], dtype=torch.float32, device=device)

        # Input space relative k-NN distance
        dist_x = torch.cdist(xb, X_ref_t, p=2.0)
        knn_d_x, knn_idx_x = torch.topk(dist_x, k=k, largest=False, dim=1)
        local_rad_x = torch.mean(rad_x_t[knn_idx_x], dim=1)
        rel_x = torch.mean(knn_d_x, dim=1) / (local_rad_x + 0.08)

        # Latent space relative k-NN distance
        dist_z = torch.cdist(zb, Z_ref_t, p=2.0)
        knn_d_z, knn_idx_z = torch.topk(dist_z, k=k, largest=False, dim=1)
        local_rad_z = torch.mean(rad_z_t[knn_idx_z], dim=1)
        rel_z = torch.mean(knn_d_z, dim=1) / (local_rad_z + 0.08)

        # Port surprise is the last column of xb
        port_surprise = torch.clamp(xb[:, -1], min=0.0)

        combined = 0.55 * rel_x + 0.30 * rel_z + 0.45 * port_surprise
        rel_dists.append(combined.cpu().numpy())

    return np.concatenate(rel_dists, axis=0)

import os
import json
import numpy as np
import matplotlib.pyplot as plt

LOGS_DIR = "logs"


def main():
    bench_path = os.path.join(LOGS_DIR, "benchmark_results.json")
    al_path = os.path.join(LOGS_DIR, "active_learning_results.json")

    if not os.path.exists(bench_path):
        print(f"[!] Run 'python3 -m experiments.evaluate' first.")
        return

    with open(bench_path, "r") as f:
        bench = json.load(f)

    plt.style.use("dark_background")

    # 1. Per-Zero-Day Family Catch Rate vs. Closed-World Detection Rate
    per_cls = bench["per_zeroday_class"]
    families = []
    zg_catch = []
    cw_catch = []

    for k, v in per_cls.items():
        clean_name = "".join(ch if ord(ch) < 128 else "-" for ch in str(k)).replace("---", "-")
        families.append(clean_name)
        zg_catch.append(v["zerogate_catch_rate_pct"])
        cw_catch.append(100.0 - v["closed_world_fooled_as_benign_pct"])

    x = np.arange(len(families))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor("#08080a")
    ax.set_facecolor("#0c0e14")

    ax.bar(x - width / 2, cw_catch, width, label="Standard Closed-World NIDS", color="#475569", edgecolor="#64748b")
    ax.bar(x + width / 2, zg_catch, width, label="ZeroGate v4 (0-Shot Conformal)", color="#f43f5e", edgecolor="#fda4af")

    ax.set_ylabel("Zero-Day Detection / Quarantine Rate (%)", fontsize=10, color="#e2e8f0")
    ax.set_title("CIC-IDS2017 Unseen Zero-Day Detection: Closed-World vs. ZeroGate v4", fontsize=12, fontweight="bold", color="#ffffff", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=9, color="#cbd5e1")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", color="#181c26", linestyle="--", linewidth=0.8)
    ax.legend(facecolor="#0c0e14", edgecolor="#222838", fontsize=9)

    plt.tight_layout()
    out1 = os.path.join(LOGS_DIR, "zeroday_family_comparison.png")
    plt.savefig(out1, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved {out1}")

    # 2. Active Learning Progression Curve
    if os.path.exists(al_path):
        with open(al_path, "r") as f:
            al_data = json.load(f)

        patches = [0, 1, 3, 5, 10]
        bot_curve = [al_data[f"{p}_patches"]["bot_catch_pct"] for p in patches]
        total_curve = [al_data[f"{p}_patches"]["total_zeroday_catch_pct"] for p in patches]

        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
        fig.patch.set_facecolor("#08080a")
        ax.set_facecolor("#0c0e14")

        ax.plot(patches, total_curve, marker="o", linewidth=2.4, color="#10b981", label="Total Zero-Day Catch Rate (All 4,183 Flows)")
        ax.plot(patches, bot_curve, marker="s", linewidth=2.2, linestyle="--", color="#38bdf8", label="Botnet Family Catch Rate (Port 8080)")

        ax.set_xlabel("Number of Analyst-Patched Flows per Family (/recalibrate)", fontsize=10, color="#e2e8f0")
        ax.set_ylabel("Detection Rate (%)", fontsize=10, color="#e2e8f0")
        ax.set_title("ZeroGate 1-Click Active Learning Progression", fontsize=12, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xticks(patches)
        ax.set_ylim(30, 103)
        ax.grid(True, color="#181c26", linestyle="--", linewidth=0.8)
        ax.legend(facecolor="#0c0e14", edgecolor="#222838", fontsize=9)

        plt.tight_layout()
        out2 = os.path.join(LOGS_DIR, "active_learning_progression.png")
        plt.savefig(out2, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[+] Saved {out2}")


if __name__ == "__main__":
    main()

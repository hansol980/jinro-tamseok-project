# -*- coding: utf-8 -*-
"""
집계 결과(results/results.json)로부터 논문용 그림을 생성한다.
모든 막대에 표본표준편차 에러바를 표시(멀티시드)하여 단일 실행 보고의 한계를 극복.

생성물(→ paper/figures/):
  fig1_image_mse.png       이미지 방어별 복원 MSE (mean±std)
  fig2_gnn_defense.png     GNN 공방 시나리오별 복원 코사인 (mean±std)
  fig3_gnn_trend.png       GNN 학습 라운드별 유출(코사인) 및 정확도 추세
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False
rcParams["savefig.dpi"] = 200
rcParams["figure.dpi"] = 200

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(HERE, "results", "results.json")))


def m(s):
    return s["mean"] if s else float("nan")


def sd(s):
    return s["std"] if s else 0.0


# ---------------------------------------------------------------------------
# Fig 1: 이미지 방어별 복원 MSE (mean±std)
# ---------------------------------------------------------------------------
def fig_image():
    order = ["none", "sparse_0.2", "sparse_0.5", "sparse_0.8", "quant", "pruning_0.2", "soteria_0.2"]
    labels = ["None", "Sparse\n0.2", "Sparse\n0.5", "Sparse\n0.8", "Quant\nFP16", "Pruning\n0.2", "Soteria\n0.2"]
    img = R["image"]
    means = [m(img[k]["mse"]) for k in order if k in img]
    stds = [sd(img[k]["mse"]) for k in order if k in img]
    labs = [labels[i] for i, k in enumerate(order) if k in img]
    base = means[0] if means else 1.0
    colors = ["#9e9e9e"] + ["#1565c0" if v > base * 3 else "#90a4ae" for v in means[1:]]

    fig, ax = plt.subplots(figsize=(8, 4.3))
    bars = ax.bar(labs, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="#37474f", linewidth=0.6, error_kw={"elinewidth": 1, "ecolor": "#444"})
    ax.set_ylabel("Reconstruction MSE (mean +/- std, n=%d)" % img[order[0]]["mse"]["n"], fontsize=11)
    ax.set_title("Fig. 1. Reconstruction error per defense on images (CIFAR-100) - multi-seed", fontsize=12, pad=10)
    ax.axhline(base, color="#e53935", linestyle="--", linewidth=1.0, alpha=0.7)
    for b, v, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, v + s + base * 0.05, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_image_mse.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_image_mse.png")


# ---------------------------------------------------------------------------
# Fig 2: GNN 공방 시나리오별 복원 코사인 (mean±std)
# ---------------------------------------------------------------------------
def fig_gnn_defense():
    order = ["baseline", "fedlog", "adaptive", "dp", "sparse"]
    labels = ["(1)\nBaseline\n(no defense)", "(2)\nFedLoG\n(scaling)", "(3)\nAdaptive\n(joint opt.)",
              "(4)\nDP\n(clip+noise)", "(5)\nSparse\n(95%)"]
    dfn = R["gnn"]["defense"]
    keys = [k for k in order if k in dfn]
    means = [m(dfn[k]["cosine"]) for k in keys]
    stds = [sd(dfn[k]["cosine"]) for k in keys]
    labs = [labels[order.index(k)] for k in keys]
    colors = ["#c62828" if v >= 0.5 else "#2e7d32" for v in means]
    n = dfn[keys[0]]["cosine"]["n"]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(labs, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="#263238", linewidth=0.6, error_kw={"elinewidth": 1, "ecolor": "#333"})
    ax.set_ylabel("Reconstruction cosine similarity (mean +/- std, n=%d)" % n, fontsize=10.5)
    ax.set_ylim(min(0, min(m_ - s_ for m_, s_ in zip(means, stds)) - 0.05), 1.05)
    ax.set_title("Fig. 2. Node-feature reconstruction per arms-race scenario on graph (WikiCS) - multi-seed", fontsize=11.5, pad=10)
    ax.axhline(0.5, color="#c62828", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(0.0, color="#888", linewidth=0.8)
    for b, v, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#c62828", label="Leakage risk (cos>=0.5)"),
                       Patch(facecolor="#2e7d32", label="Defense success (cos<0.5)")],
              loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_gnn_defense.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_gnn_defense.png")


# ---------------------------------------------------------------------------
# Fig 3: GNN 쌍체(시드 내) 공방 효과 — 견고한 결과
# ---------------------------------------------------------------------------
def fig_gnn_paired():
    pr = R["gnn"]["paired"]
    order = ["scaling_effect", "adaptive_break", "dp_defense", "sparse_defense"]
    labels = ["Feature scaling\n(no def.->def.)", "Adaptive attack\n(naive->joint opt.)",
              "DP defense\n(vs. adaptive)", "Sparsify defense\n(vs. adaptive)"]
    keys = [k for k in order if k in pr]
    means = [pr[k]["mean"] for k in keys]
    stds = [pr[k]["std"] for k in keys]
    labs = [labels[order.index(k)] for k in keys]
    # 양수(유출 증가)=빨강, 음수(유출 감소=방어)=초록, ~0=회색
    colors = ["#c62828" if v > 0.1 else ("#2e7d32" if v < -0.1 else "#9e9e9e") for v in means]
    n = pr[keys[0]]["n"]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labs, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="#263238", linewidth=0.6, error_kw={"elinewidth": 1, "ecolor": "#333"})
    ax.axhline(0, color="#444", linewidth=1.0)
    ax.set_ylabel("Change in reconstruction cosine, Delta (within-seed paired, n=%d)" % n, fontsize=10)
    ax.set_title("Fig. 3. Change in reconstruction per arms-race stage for the same target node (paired analysis)", fontsize=11, pad=10)
    for b, v, s in zip(bars, means, stds):
        off = s + 0.03 if v >= 0 else -(s + 0.03)
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax.text(0.015, 0.97, "Delta>0: more leakage (attack wins)\nDelta<0: less leakage (defense wins)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_gnn_paired.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_gnn_paired.png")


if __name__ == "__main__":
    fig_image()
    fig_gnn_defense()
    fig_gnn_paired()
    # 오래된 단일실행 그림 제거(있다면)
    old = os.path.join(OUT, "fig2_gnn_cosine.png")
    if os.path.exists(old):
        os.remove(old)
        print("removed stale fig2_gnn_cosine.png")
    print("All figures regenerated ->", OUT)

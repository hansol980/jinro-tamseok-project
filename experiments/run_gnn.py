# -*- coding: utf-8 -*-
"""
GNN(WikiCS + FedLoGModel) 그래디언트 유출 공방 — 멀티시드 엄밀 실험.

기존 main_gnn.py 대비 보완점
  1) 시드 고정으로 재현성 확보(N_SEEDS회 반복)
  2) "장식뿐인 3 클라이언트" -> 실제 다중 클라이언트 FedAvg(FedSGD) 학습
  3) 공격 전 모델을 충분히 학습(랜덤 초기화 모델 공격은 비현실적)
  4) DP를 클리핑+노이즈로 정정(common.dp_defense)
  5) 모델 분류 정확도 + 그래디언트 충실도(유용성)까지 측정
  6) 시나리오별 결과를 JSONL로 체크포인트 저장(중단되어도 부분결과 유지)

시나리오: baseline / fedlog / adaptive / dp / sparse
"""
import os, sys, json, time, argparse, random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WikiCS
from torch_geometric.utils import degree, subgraph
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnn_model import FedLoGModel
from gnn_dlg import dlg_attack_gnn
import common

DEVICE = torch.device("cpu")
N_CLIENTS = 3
ROUNDS = 40          # FedAvg 라운드 수(공격 전 충분히 학습)
LOCAL_LR = 0.01
DP_CLIP = 1.0
DP_NOISE_MULT = 1.0  # sigma = noise_mult * clip
SPARSITY = 0.95
ATTACK_ITERS = 200
ATTACK_RESTARTS = 3
SCENARIOS = ["baseline", "fedlog", "adaptive", "dp", "sparse"]


def mask_1d(m, idx):
    """train/val/test 마스크가 [N] 또는 [N,k]일 수 있으므로 첫 split을 1D로."""
    m = m[idx]
    if m.dim() > 1:
        m = m[:, 0]
    return m


def build_clients(data, num_node_features):
    perm = torch.randperm(data.num_nodes)
    split = data.num_nodes // N_CLIENTS
    clients = []
    for i in range(N_CLIENTS):
        s = i * split
        e = (i + 1) * split if i != N_CLIENTS - 1 else data.num_nodes
        nidx = perm[s:e]
        sub_edge, _ = subgraph(nidx, data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
        cd = Data(
            x=data.x[nidx],
            edge_index=sub_edge,
            y=data.y[nidx],
            degrees=data.degrees[nidx],
            train_mask=mask_1d(data.train_mask, nidx),
            test_mask=mask_1d(data.test_mask, nidx),
        )
        clients.append(cd)
    return clients


def client_gradient(model, cd, scale):
    """클라이언트가 자신의 비밀 스케일로 로컬 학습 그래디언트를 계산."""
    model.feature_scale = scale
    model.zero_grad()
    out = model(cd.x, cd.edge_index, cd.degrees)
    tn = cd.train_mask.nonzero(as_tuple=True)[0]
    loss = F.cross_entropy(out[tn], cd.y[tn])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [g.detach().clone() for g in grads]


def fedavg_train(model, clients, scales, rounds=ROUNDS, lr=LOCAL_LR):
    """FedSGD식: 매 라운드 클라이언트 그래디언트를 평균해 글로벌 모델 갱신."""
    model.train()
    for _ in range(rounds):
        accum = None
        for cd, sc in zip(clients, scales):
            grads = client_gradient(model, cd, sc)
            if accum is None:
                accum = [g.clone() for g in grads]
            else:
                for a, g in zip(accum, grads):
                    a += g
        with torch.no_grad():
            for p, g in zip(model.parameters(), accum):
                p.data -= lr * (g / len(clients))


@torch.no_grad()
def eval_accuracy(model, clients, scales):
    """각 클라이언트의 test 노드 정확도를 해당 클라이언트 스케일로 평가 후 평균."""
    model.eval()
    accs = []
    for cd, sc in zip(clients, scales):
        tm = cd.test_mask.nonzero(as_tuple=True)[0]
        if len(tm) == 0:
            continue
        model.feature_scale = sc
        out = model(cd.x, cd.edge_index, cd.degrees)
        pred = out[tm].argmax(dim=1)
        accs.append((pred == cd.y[tm]).float().mean().item())
    return sum(accs) / max(1, len(accs))


def new_model(data, num_features):
    return FedLoGModel(in_channels=num_features, hidden_channels=32,
                       out_channels=int(data.y.max().item()) + 1).to(DEVICE)


def target_gradient(model, cd, target_idx, target_label, scale):
    model.eval()
    model.feature_scale = scale
    model.zero_grad()
    out = model(cd.x, cd.edge_index, cd.degrees)
    tloss = F.cross_entropy(out[target_idx:target_idx + 1],
                            torch.tensor([target_label], device=DEVICE))
    return [g.detach().clone() for g in
            torch.autograd.grad(tloss, model.parameters(), create_graph=False)]


def attack_and_record(model, cd, target_idx, target_label, sent, clean,
                      scen, optimize_scale, acc, true_scale, seed, writer):
    t0 = time.time()
    fidelity = common.gradient_fidelity(sent, clean)
    model.feature_scale = 1.0  # 서버는 비밀 스케일을 모른 채 1.0에서 출발
    _, rec_scale, m, c = dlg_attack_gnn(
        model=model, target_gradients=sent, data=cd,
        target_idx=target_idx, true_label=target_label,
        num_iterations=ATTACK_ITERS, attack_lr=0.05,
        num_restarts=ATTACK_RESTARTS, optimize_scale=optimize_scale,
    )
    scale_err = abs(rec_scale.item() - true_scale) if (optimize_scale and rec_scale is not None) else None
    rec = {
        "track": "gnn", "seed": seed, "scenario": scen,
        "mse": m, "cosine": c, "fidelity": fidelity, "accuracy": acc,
        "true_scale": true_scale,
        "stolen_scale": rec_scale.item() if (optimize_scale and rec_scale is not None) else None,
        "scale_err": scale_err, "secs": round(time.time() - t0, 1),
    }
    writer.write(json.dumps(rec) + "\n")
    writer.flush()
    print(f"[seed {seed}] {scen:9s} cos={c:.4f} mse={m:.6f} acc={acc:.3f} "
          f"fid={fidelity:.3f} ({rec['secs']}s)", flush=True)


def maybe_train(model, clients, scales, rounds):
    if rounds > 0:
        fedavg_train(model, clients, scales, rounds=rounds)


def attack_and_record(model, cd, target_idx, target_label, sent, clean,
                      scen, optimize_scale, acc, true_scale, seed, rounds, writer):
    t0 = time.time()
    fidelity = common.gradient_fidelity(sent, clean)
    model.feature_scale = 1.0  # 서버는 비밀 스케일을 모른 채 1.0에서 출발
    _, rec_scale, m, c = dlg_attack_gnn(
        model=model, target_gradients=sent, data=cd,
        target_idx=target_idx, true_label=target_label,
        num_iterations=ATTACK_ITERS, attack_lr=0.05,
        num_restarts=ATTACK_RESTARTS, optimize_scale=optimize_scale,
    )
    scale_err = abs(rec_scale.item() - true_scale) if (optimize_scale and rec_scale is not None) else None
    rec = {
        "track": "gnn", "seed": seed, "scenario": scen, "rounds": rounds,
        "mse": m, "cosine": c, "fidelity": fidelity, "accuracy": acc,
        "true_scale": true_scale,
        "stolen_scale": rec_scale.item() if (optimize_scale and rec_scale is not None) else None,
        "scale_err": scale_err, "secs": round(time.time() - t0, 1),
    }
    writer.write(json.dumps(rec) + "\n")
    writer.flush()
    print(f"[seed {seed} r{rounds}] {scen:9s} cos={c:.4f} mse={m:.6f} acc={acc:.3f} "
          f"fid={fidelity:.3f} ({rec['secs']}s)", flush=True)


def run_seed(seed, data, num_features, rounds, scenarios, writer):
    common.set_seed(seed)
    clients = build_clients(data, num_features)
    secret_scales = [random.uniform(0.1, 3.0) for _ in range(N_CLIENTS)]
    target_client = 0
    cd = clients[target_client]
    tn = cd.train_mask.nonzero(as_tuple=True)[0]
    if len(tn) == 0:
        return
    target_idx = tn[random.randint(0, len(tn) - 1)].item()
    target_label = cd.y[target_idx].item()

    # ===== baseline: 스케일=1, 방어 없음, 표준 공격 =====
    if "baseline" in scenarios:
        common.set_seed(seed)
        m_base = new_model(data, num_features)
        maybe_train(m_base, clients, [1.0] * N_CLIENTS, rounds)
        acc_base = eval_accuracy(m_base, clients, [1.0] * N_CLIENTS)
        clean_base = target_gradient(m_base, cd, target_idx, target_label, 1.0)
        attack_and_record(m_base, cd, target_idx, target_label, clean_base, clean_base,
                          "baseline", False, acc_base, 1.0, seed, rounds, writer)

    # ===== 방어 모델: 비밀 스케일 (fedlog/adaptive/dp/sparse 공유) =====
    defended_needed = [s for s in scenarios if s in ("fedlog", "adaptive", "dp", "sparse")]
    if defended_needed:
        common.set_seed(seed)
        m_def = new_model(data, num_features)
        maybe_train(m_def, clients, secret_scales, rounds)
        acc_def = eval_accuracy(m_def, clients, secret_scales)
        ts = secret_scales[target_client]
        clean = target_gradient(m_def, cd, target_idx, target_label, ts)

        if "fedlog" in defended_needed:
            attack_and_record(m_def, cd, target_idx, target_label, clean, clean,
                              "fedlog", False, acc_def, ts, seed, rounds, writer)
        if "adaptive" in defended_needed:
            attack_and_record(m_def, cd, target_idx, target_label, clean, clean,
                              "adaptive", True, acc_def, ts, seed, rounds, writer)
        if "dp" in defended_needed:
            sent_dp = common.dp_defense(clean, clip_norm=DP_CLIP, noise_multiplier=DP_NOISE_MULT)
            attack_and_record(m_def, cd, target_idx, target_label, sent_dp, clean,
                              "dp", True, acc_def, ts, seed, rounds, writer)
        if "sparse" in defended_needed:
            sent_sp = common.sparsify_defense(clean, sparsity=SPARSITY)
            attack_and_record(m_def, cd, target_idx, target_label, sent_sp, clean,
                              "sparse", True, acc_def, ts, seed, rounds, writer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=0,
                    help="공격 전 FedAvg 학습 라운드 수(0=미학습, worst-case 유출)")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS),
                    help="콤마구분 시나리오 (예: baseline 또는 baseline,fedlog,...)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "gnn.jsonl"))
    args = ap.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    print(f"Loading WikiCS ... (rounds={args.rounds}, scenarios={scenarios})", flush=True)
    ds = WikiCS(root=os.path.join(ROOT, "data", "WikiCS"))
    data = ds[0]
    data.degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    num_features = ds.num_node_features

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["scenario"], r.get("rounds", 0)))
            except Exception:
                pass

    with open(args.out, "a") as w:
        for seed in range(args.seeds):
            todo = [s for s in scenarios if (seed, s, args.rounds) not in done]
            if not todo:
                print(f"[seed {seed} r{args.rounds}] already done, skip", flush=True)
                continue
            run_seed(seed, data, num_features, args.rounds, todo, w)
    print("GNN experiments complete.", flush=True)


if __name__ == "__main__":
    main()

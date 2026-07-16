"""Controlled FedLoG privacy experiments on the official Cora partition.

This module reproduces the privacy-relevant parts of FedLoG that can be run on
CPU: a two-layer GraphSAGE encoder, global synthetic features, and the
class-wise feature-scaling equation from Eq. (5) of the ICLR 2025 paper.  It is
not a drop-in reproduction of the full prompt-generator/condensation pipeline;
that distinction is recorded in every result file and in the paper.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/FedLoG/closeset/Cora/4/local_graphs.pkl"
DEFAULT_OUT = ROOT / "experiments/results/fedlog_required.jsonl"
DEVICE = torch.device("cpu")


@dataclass
class Defense:
    name: str = "none"
    value: float = 0.0
    clip: float = 1.0


class PrivacySAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim * 2)
        self.conv2 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def load_graphs(path: Path = DEFAULT_DATA):
    with path.open("rb") as handle:
        graphs = pickle.load(handle)
    return [g.cpu() for g in graphs]


def train_indices(graph) -> torch.Tensor:
    return graph.train_mask.nonzero(as_tuple=True)[0]


def make_global_synthetic(graphs, num_proto: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Create public class prototypes with FedLoG's class-rate weighting.

    The official implementation learns local condensed features and aggregates
    them by class rate.  For this CPU study, class means stand in for the local
    condensed features so the aggregation and feature-scaling mechanisms can be
    isolated without claiming a full condensation reproduction.
    """
    num_classes = int(graphs[0].num_cls)
    features = []
    labels = []
    for cls in range(num_classes):
        local_means = []
        local_counts = []
        for graph in graphs[:-1]:
            idx = train_indices(graph)
            cls_idx = idx[graph.y[idx] == cls]
            if cls_idx.numel():
                local_means.append(graph.x[cls_idx].mean(0))
                local_counts.append(float(cls_idx.numel()))
        if local_means:
            weights = torch.tensor(local_counts)
            weights /= weights.sum()
            proto = torch.stack(local_means).mul(weights[:, None]).sum(0)
        else:
            proto = torch.zeros(graphs[0].x.size(1))
        for _ in range(num_proto):
            features.append(proto.clone())
            labels.append(cls)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def scale_synthetic(
    syn_x: torch.Tensor, syn_y: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """FedLoG Eq. (5): x_hat = x + gamma[c] * (mean(x) - x)."""
    mean_syn = syn_x.mean(0, keepdim=True)
    per_node_gamma = gamma[syn_y].unsqueeze(1)
    return syn_x + per_node_gamma * (mean_syn - syn_x)


def loss_for_update(
    model: PrivacySAGE,
    graph,
    batch_idx: torch.Tensor,
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    gamma: torch.Tensor,
    use_synthetic: bool,
    synthetic_weight: float = 1.0,
) -> torch.Tensor:
    logits = model(graph.x, graph.edge_index)
    loss = F.cross_entropy(logits[batch_idx], graph.y[batch_idx])
    if use_synthetic:
        scaled = scale_synthetic(syn_x, syn_y, gamma)
        empty_edge = torch.empty((2, 0), dtype=torch.long)
        syn_logits = model(scaled, empty_edge)
        loss = loss + synthetic_weight * F.cross_entropy(syn_logits, syn_y)
    return loss


def gradients(loss: torch.Tensor, model: nn.Module, create_graph: bool = False):
    return list(torch.autograd.grad(loss, tuple(model.parameters()), create_graph=create_graph))


def flatten(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def global_norm(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum((g.detach() ** 2).sum() for g in grads) + 1e-12)


def defend_gradients(
    grads: list[torch.Tensor], defense: Defense, seed: int
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    if defense.name == "none":
        return [g.detach().clone() for g in grads], None

    norm = global_norm(grads)
    scale = min(1.0, defense.clip / float(norm))
    clipped = [g.detach() * scale for g in grads]

    if defense.name == "noise":
        generator = torch.Generator().manual_seed(seed)
        std = defense.value * defense.clip
        return [g + torch.randn(g.shape, generator=generator) * std for g in clipped], None

    if defense.name == "sparse":
        all_values = flatten(clipped)
        keep = max(1, int(all_values.numel() * (1.0 - defense.value)))
        threshold = torch.topk(all_values.abs(), keep, largest=True).values[-1]
        masks = [(g.abs() >= threshold).to(g.dtype) for g in clipped]
        return [g * m for g, m in zip(clipped, masks)], masks

    raise ValueError(f"Unknown defense: {defense.name}")


def update_gamma(
    model: PrivacySAGE,
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    gamma: torch.Tensor,
    step: float = 0.001,
) -> torch.Tensor:
    empty_edge = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        pred = model(scale_synthetic(syn_x, syn_y, gamma), empty_edge).argmax(1)
        updated = gamma.clone()
        for cls in range(gamma.numel()):
            mask = syn_y == cls
            acc = (pred[mask] == cls).float().mean() if mask.any() else torch.tensor(0.0)
            updated[cls] += step if acc > 0.8 else -step
        return updated.clamp_(0.0, 1.0)


@torch.no_grad()
def evaluate(model: PrivacySAGE, graphs) -> tuple[float, float]:
    model.eval()
    ys = []
    preds = []
    for graph in graphs[:-1]:
        idx = graph.test_mask.nonzero(as_tuple=True)[0]
        if not idx.numel():
            continue
        pred = model(graph.x, graph.edge_index)[idx].argmax(1)
        ys.extend(graph.y[idx].tolist())
        preds.extend(pred.tolist())
    accuracy = float(np.mean(np.asarray(ys) == np.asarray(preds)))
    macro_f1 = float(f1_score(ys, preds, average="macro", zero_division=0))
    return accuracy, macro_f1


def federated_train(
    graphs,
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    seed: int,
    rounds: int,
    use_synthetic: bool,
    defense: Defense,
    lr: float = 0.05,
) -> tuple[PrivacySAGE, list[torch.Tensor], list[dict]]:
    set_seed(seed)
    model = PrivacySAGE(graphs[0].x.size(1), 32, int(graphs[0].num_cls)).to(DEVICE)
    gammas = [torch.zeros(int(graphs[0].num_cls)) for _ in graphs[:-1]]
    history = []

    for round_idx in range(rounds + 1):
        if round_idx in {0, max(1, rounds // 2), rounds}:
            accuracy, macro_f1 = evaluate(model, graphs)
            history.append({"round": round_idx, "accuracy": accuracy, "macro_f1": macro_f1})
        if round_idx == rounds:
            break

        client_grads = []
        for cid, graph in enumerate(graphs[:-1]):
            model.zero_grad(set_to_none=True)
            loss = loss_for_update(
                model, graph, train_indices(graph), syn_x, syn_y, gammas[cid], use_synthetic
            )
            raw = gradients(loss, model)
            sent, _ = defend_gradients(raw, defense, seed * 100000 + round_idx * 10 + cid)
            client_grads.append(sent)

        with torch.no_grad():
            for param_idx, param in enumerate(model.parameters()):
                avg = torch.stack([g[param_idx] for g in client_grads]).mean(0)
                param.add_(avg, alpha=-lr)
        if use_synthetic:
            gammas = [update_gamma(model, syn_x, syn_y, gamma, step=0.002) for gamma in gammas]

    return model, gammas, history


def attack_subgraph(graph, batch_idx: torch.Tensor):
    subset, sub_edge, mapping, _ = k_hop_subgraph(
        batch_idx, 2, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
    )
    subgraph = copy.copy(graph)
    subgraph.x = graph.x[subset].clone()
    subgraph.y = graph.y[subset].clone()
    subgraph.edge_index = sub_edge
    return subgraph, mapping, subset


def observed_update(
    model: PrivacySAGE,
    graph,
    batch_idx: torch.Tensor,
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    gamma: torch.Tensor,
    use_synthetic: bool,
    defense: Defense,
    seed: int,
):
    subgraph, mapping, subset = attack_subgraph(graph, batch_idx)
    model.zero_grad(set_to_none=True)
    loss = loss_for_update(model, subgraph, mapping, syn_x, syn_y, gamma, use_synthetic)
    raw = gradients(loss, model)
    sent, masks = defend_gradients(raw, defense, seed)
    return subgraph, mapping, subset, sent, masks, raw


def cosine_grad_loss(candidate: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
    a = flatten(candidate)
    b = flatten(target)
    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), eps=1e-12).squeeze()


def invert_update(
    model: PrivacySAGE,
    subgraph,
    mapping: torch.Tensor,
    observed: list[torch.Tensor],
    masks: list[torch.Tensor] | None,
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    attack_use_synthetic: bool,
    adaptive_gamma: bool,
    true_gamma: torch.Tensor,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    set_seed(seed)
    fixed_x = subgraph.x.detach().clone()
    true_x = fixed_x[mapping].clone()
    dummy_logits = torch.randn_like(true_x, requires_grad=True)
    params = [dummy_logits]
    if adaptive_gamma:
        gamma_logits = torch.zeros_like(true_gamma, requires_grad=True)
        params.append(gamma_logits)
    else:
        gamma_logits = None
    optimizer = torch.optim.Adam(params, lr=0.12)
    best_loss = math.inf
    best_x = None
    best_gamma = torch.zeros_like(true_gamma)

    unknown_mask = torch.zeros(subgraph.num_nodes, 1)
    unknown_mask[mapping] = 1.0
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        dummy_x = torch.sigmoid(dummy_logits)
        expanded = fixed_x.clone()
        expanded[mapping] = dummy_x
        candidate_graph = copy.copy(subgraph)
        candidate_graph.x = expanded
        if adaptive_gamma:
            candidate_gamma = torch.sigmoid(gamma_logits)
        else:
            candidate_gamma = torch.zeros_like(true_gamma)
        loss = loss_for_update(
            model,
            candidate_graph,
            mapping,
            syn_x,
            syn_y,
            candidate_gamma,
            attack_use_synthetic,
        )
        candidate = gradients(loss, model, create_graph=True)
        if masks is not None:
            candidate = [g * m for g, m in zip(candidate, masks)]
        objective = cosine_grad_loss(candidate, observed)
        density = true_x.mean().detach()
        objective = objective + 1e-4 * (dummy_x.mean() - density).abs()
        objective.backward()
        optimizer.step()
        value = float(objective.detach())
        if value < best_loss:
            best_loss = value
            best_x = dummy_x.detach().clone()
            best_gamma = candidate_gamma.detach().clone()

    assert best_x is not None
    return best_x, best_gamma, best_loss


def reconstruction_metrics(true_x: torch.Tensor, recovered: torch.Tensor) -> dict:
    true_flat = true_x.reshape(-1)
    rec_flat = recovered.reshape(-1)
    mse = F.mse_loss(rec_flat, true_flat).item()
    denom = true_flat.var(unbiased=False).item() + 1e-12
    cosine = F.cosine_similarity(rec_flat.unsqueeze(0), true_flat.unsqueeze(0)).item()
    true_binary = true_flat > 0.5
    k = max(1, int(true_binary.sum().item()))
    rec_top = torch.zeros_like(true_binary)
    rec_top[torch.topk(rec_flat, min(k, rec_flat.numel())).indices] = True
    overlap = (rec_top & true_binary).sum().item() / max(1, true_binary.sum().item())
    return {"cosine": cosine, "mse": mse, "nmse": mse / denom, "topk_overlap": overlap}


def select_batch(graph, seed: int, batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    idx = train_indices(graph)
    perm = torch.randperm(idx.numel(), generator=generator)
    return idx[perm[: min(batch_size, idx.numel())]]


def run_attack_suite(
    graphs,
    syn_x,
    syn_y,
    seed: int,
    rounds: int,
    batch_sizes: list[int],
    iterations: int,
    writer,
) -> None:
    model, gammas, history = federated_train(
        graphs, syn_x, syn_y, seed, rounds, True, Defense("none")
    )
    model.eval()
    graph = graphs[0]
    scenarios = [
        ("plain", False, Defense("none"), False, False),
        ("fedlog_naive", True, Defense("none"), False, False),
        ("fedlog_adaptive", True, Defense("none"), True, True),
        ("noise_0.0001", True, Defense("noise", 0.0001), True, True),
        ("noise_0.001", True, Defense("noise", 0.001), True, True),
        ("sparse_0.90", True, Defense("sparse", 0.90), True, True),
        ("sparse_0.95", True, Defense("sparse", 0.95), True, True),
    ]

    for batch_size in batch_sizes:
        batch = select_batch(graph, seed * 7919 + batch_size, batch_size)
        for name, observed_syn, defense, attack_syn, adaptive in scenarios:
            subgraph, mapping, _, sent, masks, raw = observed_update(
                model,
                graph,
                batch,
                syn_x,
                syn_y,
                gammas[0],
                observed_syn,
                defense,
                seed * 1000 + batch_size,
            )
            recovered, recovered_gamma, objective = invert_update(
                model,
                subgraph,
                mapping,
                sent,
                masks,
                syn_x,
                syn_y,
                attack_syn,
                adaptive,
                gammas[0],
                iterations,
                seed * 10000 + batch_size,
            )
            metrics = reconstruction_metrics(subgraph.x[mapping], recovered)
            row = {
                "kind": "attack",
                "implementation": "official-equation CPU reproduction",
                "seed": seed,
                "rounds": rounds,
                "batch_size": batch_size,
                "scenario": name,
                "gradient_fidelity": F.cosine_similarity(
                    flatten(sent).unsqueeze(0), flatten(raw).unsqueeze(0)
                ).item(),
                "gamma_mae": (recovered_gamma - gammas[0]).abs().mean().item()
                if adaptive
                else None,
                "objective": objective,
                **metrics,
            }
            writer.write(json.dumps(row) + "\n")
            writer.flush()
            print(
                f"attack seed={seed} b={batch_size} {name:16s} "
                f"cos={metrics['cosine']:.3f} nmse={metrics['nmse']:.3f}",
                flush=True,
            )


def run_utility_suite(graphs, syn_x, syn_y, seed: int, rounds: int, writer) -> None:
    configs = [
        ("plain", False, Defense("none")),
        ("fedlog_reproduction", True, Defense("none")),
        ("noise_0.0001", True, Defense("noise", 0.0001)),
        ("noise_0.001", True, Defense("noise", 0.001)),
        ("sparse_0.90", True, Defense("sparse", 0.90)),
        ("sparse_0.95", True, Defense("sparse", 0.95)),
    ]
    for name, use_synthetic, defense in configs:
        started = time.time()
        _, gammas, history = federated_train(
            graphs, syn_x, syn_y, seed, rounds, use_synthetic, defense
        )
        row = {
            "kind": "utility",
            "implementation": "official-equation CPU reproduction",
            "seed": seed,
            "rounds": rounds,
            "scenario": name,
            "accuracy": history[-1]["accuracy"],
            "macro_f1": history[-1]["macro_f1"],
            "history": history,
            "gamma_mean": torch.stack(gammas).mean().item(),
            "seconds": round(time.time() - started, 2),
        }
        writer.write(json.dumps(row) + "\n")
        writer.flush()
        print(
            f"utility seed={seed} {name:20s} acc={row['accuracy']:.3f} "
            f"f1={row['macro_f1']:.3f}",
            flush=True,
        )


def completed_keys(path: Path) -> set[tuple]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "utility":
            keys.add(("utility", row["seed"]))
        elif row.get("kind") == "attack":
            keys.add(("attack", row["seed"], row["batch_size"], row["scenario"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--phase", choices=["all", "utility", "attack"], default="all")
    args = parser.parse_args()

    graphs = load_graphs(args.data)
    syn_x, syn_y = make_global_synthetic(graphs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.out)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]

    with args.out.open("a") as writer:
        for seed in range(args.seeds):
            if args.phase in {"all", "utility"} and ("utility", seed) not in done:
                run_utility_suite(graphs, syn_x, syn_y, seed, args.rounds, writer)
            if args.phase in {"all", "attack"}:
                missing = [
                    b
                    for b in batch_sizes
                    if any(
                        ("attack", seed, b, scenario) not in done
                        for scenario in [
                            "plain",
                            "fedlog_naive",
                            "fedlog_adaptive",
                            "noise_0.0001",
                            "noise_0.001",
                            "sparse_0.90",
                            "sparse_0.95",
                        ]
                    )
                ]
                if missing:
                    run_attack_suite(
                        graphs, syn_x, syn_y, seed, args.rounds, missing, args.iterations, writer
                    )


if __name__ == "__main__":
    main()

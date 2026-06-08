import torch
import torch.nn.functional as F
from torch_geometric.datasets import WikiCS
from torch_geometric.utils import degree, subgraph
from torch_geometric.data import Data
import random
import argparse
import numpy as np

from gnn_model import FedLoGModel
from gnn_dlg import dlg_attack_gnn

def set_seed(seed=42):
    """재현성 확보: 모든 난수원 고정 (기존 코드의 결함 보완)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)
    print(f"=== Federated Learning & DLG Attack with FedLoGModel (seed={args.seed}) ===")
    if args.defense:
        print("[!] Defense Mechanisms (Data Condensation, Feature Scaling, Class Noise) ENABLED")
        if args.adaptive_attack:
            print("[!] ADAPTIVE ATTACK ENABLED: Server will jointly optimize dummy_x and dummy_scale!")
    else:
        print("[!] Defense Mechanisms DISABLED (Baseline Vulnerability)")
    
    # 1. Dataset Load
    print("Loading WikiCS Dataset...")
    dataset = WikiCS(root='./data/WikiCS')
    data = dataset[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Compute node degrees for FedLoGModel
    data.degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).to(device)
    data = data.to(device)
    
    # 2. Model Initialization
    in_channels = dataset.num_node_features
    out_classes = dataset.num_classes
    
    model = FedLoGModel(in_channels=in_channels, hidden_channels=32, out_channels=out_classes).to(device)
    
    # 3. Simulate Federated Learning Clients
    n_clients = 3
    print(f"Partitioning graph into {n_clients} clients...")
    
    indices = torch.randperm(data.num_nodes)
    split_size = data.num_nodes // n_clients
    
    clients_data = []
    for i in range(n_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != n_clients - 1 else data.num_nodes
        node_idx = indices[start:end]
        
        sub_edge_index, _ = subgraph(node_idx, data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
        
        client_data = Data(
            x=data.x[node_idx],
            edge_index=sub_edge_index,
            y=data.y[node_idx],
            degrees=data.degrees[node_idx],
            train_mask=data.train_mask[node_idx, 0] # Using first split
        )
        clients_data.append(client_data)
        
    # 4. Federated Training Simulation (Just 1 round for demonstration)
    print("\n[Federated Learning Phase]")
    
    if args.defense:
        print(">> Defense 1 & 3: Server aggregates Data Condensation with Class Distribution Noise...")
        # Simulate computing noisy class distribution
        true_dist = torch.bincount(clients_data[0].y, minlength=out_classes).float()
        noise = torch.randn_like(true_dist) * 0.5  # Gaussian Noise
        noisy_dist = F.softmax(true_dist + noise, dim=0)
        print(f"   -> Noisy Class Distribution generated: {noisy_dist[:3]}...")
        
        print(">> Defense 2: Client 0 applies Secret Feature Scaling to Global Synthetic Data...")
        secret_scale = random.uniform(0.1, 3.0)
        model.feature_scale = secret_scale
    else:
        model.feature_scale = 1.0
        
    # 실제 다중 클라이언트 FedAvg(FedSGD): 모든 클라이언트가 그래디언트를 계산하고
    # 서버가 평균내어 글로벌 모델을 갱신한다. (기존 코드는 클라이언트 0만 학습했음)
    print(f">> FedAvg training across {n_clients} clients for {args.rounds} rounds...")
    model.train()
    for rnd in range(args.rounds):
        accum, n_used = None, 0
        for cd in clients_data:
            train_nodes = cd.train_mask.nonzero(as_tuple=True)[0]
            if len(train_nodes) == 0:
                continue
            model.zero_grad()
            out = model(cd.x, cd.edge_index, cd.degrees)
            loss = F.cross_entropy(out[train_nodes], cd.y[train_nodes])
            grads = torch.autograd.grad(loss, model.parameters())
            if accum is None:
                accum = [g.detach().clone() for g in grads]
            else:
                for a, g in zip(accum, grads):
                    a += g.detach()
            n_used += 1
        if accum is not None:
            with torch.no_grad():
                for p, g in zip(model.parameters(), accum):
                    p.data -= 0.01 * (g / max(1, n_used))

    # 유용성(utility) 측정: 학습된 모델의 노드 분류 정확도
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for cd in clients_data:
            out = model(cd.x, cd.edge_index, cd.degrees)
            correct += (out.argmax(dim=1) == cd.y).sum().item()
            total += cd.y.numel()
    print(f"FedAvg training finished. Node classification accuracy: {correct/max(1,total):.4f}")
    
    # 5. Gradient computation for DLG Attack
    print("\n[DLG Attack Phase]")
    target_client_idx = 0
    attack_data = clients_data[target_client_idx]
    
    # Pick a random training node as the attack target
    train_indices = attack_data.train_mask.nonzero(as_tuple=True)[0]
    target_idx = train_indices[random.randint(0, len(train_indices)-1)].item()
    target_label = attack_data.y[target_idx].item()
    
    print(f"Malicious Server targets Client {target_client_idx}'s Node {target_idx} (True Label: {target_label})")
    
    # Client computes gradient for the target node
    model.eval()
    model.zero_grad()
    out = model(attack_data.x, attack_data.edge_index, attack_data.degrees)
    target_loss = F.cross_entropy(out[target_idx:target_idx+1], torch.tensor([target_label], dtype=torch.long, device=device))
    target_gradients = torch.autograd.grad(target_loss, model.parameters(), create_graph=False)
    target_gradients = [g.detach().clone() for g in target_gradients]
    
    # 5.5 Advanced Defense Mechanisms (DP and Sparsification)
    if args.defense_dp:
        print("\n[🛡️ Advanced Defense: Differential Privacy (DP-SGD style)]")
        # 기존 코드 결함 보완: 단순 노이즈가 아닌 (1)그래디언트 L2 클리핑으로 민감도를 C로
        # 제한한 뒤 (2)sigma = noise_multiplier*C 의 보정 가우시안 노이즈를 주입한다.
        clip_norm = 1.0
        noise_multiplier = 1.0
        total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in target_gradients) + 1e-12)
        scale = min(1.0, clip_norm / float(total_norm))
        sigma = noise_multiplier * clip_norm
        print(f">> Clipping grads to L2={clip_norm} (was {float(total_norm):.3f}), "
              f"then adding Gaussian noise sigma={sigma}...")
        target_gradients = [g * scale + torch.randn_like(g) * sigma for g in target_gradients]
        
    if args.defense_sparse:
        print("\n[🛡️ Advanced Defense: Extreme Sparsification]")
        sparsity_ratio = 0.95
        print(f">> Pruning {sparsity_ratio*100}% of the gradients (keeping only the top 5% magnitudes)...")
        pruned_gradients = []
        for g in target_gradients:
            flat_g = g.view(-1)
            num_drop = int(flat_g.numel() * sparsity_ratio)
            if num_drop > 0:
                _, drop_indices = torch.topk(torch.abs(flat_g), k=num_drop, largest=False)
                mask = torch.ones_like(flat_g)
                mask[drop_indices] = 0.0
                pruned_gradients.append((flat_g * mask).view(g.shape))
            else:
                pruned_gradients.append(g)
        target_gradients = pruned_gradients
    
    # 6. Malicious Server runs DLG Attack
    print("\n[Malicious Server Intercepts Gradients]")
    if args.defense:
        if args.adaptive_attack:
            print(">> Server DOES NOT know the client's secret Feature Scaling factor.")
            print(">> Server initiates ADAPTIVE DLG attack to jointly optimize Dummy Features & Secret Scale!")
        else:
            print(">> Server DOES NOT know the client's secret Feature Scaling factor.")
            print(">> Server attempts DLG attack using the standard model scale (1.0).")
        model.feature_scale = 1.0 # Server initial assumption
        
    recovered_feature, best_dummy_scale, mse, cosine_sim = dlg_attack_gnn(
        model=model,
        target_gradients=target_gradients,
        data=attack_data,
        target_idx=target_idx,
        true_label=target_label,
        num_iterations=200,
        attack_lr=0.05,
        num_restarts=3,
        optimize_scale=args.adaptive_attack
    )
    
    print("\n=== Attack Results ===")
    print(f"Target Label: {target_label}")
    if args.adaptive_attack and args.defense:
        print(f"True Secret Scale: {secret_scale:.6f}")
        print(f"Optimized (Stolen) Scale: {best_dummy_scale.item():.6f}")
    print(f"Recovered Feature MSE: {mse:.6f}")
    print(f"Recovered Feature Cosine Similarity: {cosine_sim:.6f} (Closer to 1 is better)")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLoG Federated Learning & DLG Simulation")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--rounds', type=int, default=10, help='Number of FedAvg training rounds before the attack')
    parser.add_argument('--defense', action='store_true', help='Enable FedLoG defense mechanisms (Feature Scaling & Noise)')
    parser.add_argument('--adaptive_attack', action='store_true', help='Enable Adaptive DLG attack to jointly optimize the secret scaling factor')
    parser.add_argument('--defense_dp', action='store_true', help='Advanced: Enable Differential Privacy Noise on gradients')
    parser.add_argument('--defense_sparse', action='store_true', help='Advanced: Enable Extreme Sparsification on gradients')
    args = parser.parse_args()
    main(args)

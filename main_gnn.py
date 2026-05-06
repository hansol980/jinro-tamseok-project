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

def main(args):
    print("=== Federated Learning & DLG Attack with FedLoGModel ===")
    if args.defense:
        print("[!] Defense Mechanisms (Data Condensation, Feature Scaling, Class Noise) ENABLED")
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
        
    client_0_data = clients_data[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(client_0_data.x, client_0_data.edge_index, client_0_data.degrees)
        # Only compute loss on train nodes
        train_nodes = client_0_data.train_mask.nonzero(as_tuple=True)[0]
        if len(train_nodes) > 0:
            loss = F.cross_entropy(out[train_nodes], client_0_data.y[train_nodes])
            loss.backward()
            optimizer.step()
    
    print("Client 0 finished local training.")
    
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
    
    # 6. Malicious Server runs DLG Attack
    print("\n[Malicious Server Intercepts Gradients]")
    if args.defense:
        print(">> Server DOES NOT know the client's secret Feature Scaling factor.")
        print(">> Server attempts DLG attack using the standard model scale (1.0).")
        model.feature_scale = 1.0 # Server assumes scale is 1.0
        
    recovered_feature, mse, cosine_sim = dlg_attack_gnn(
        model=model,
        target_gradients=target_gradients,
        data=attack_data,
        target_idx=target_idx,
        true_label=target_label,
        num_iterations=200,
        attack_lr=0.05,
        num_restarts=3
    )
    
    print("\n=== Attack Results ===")
    print(f"Target Label: {target_label}")
    print(f"Recovered Feature MSE: {mse:.6f}")
    print(f"Recovered Feature Cosine Similarity: {cosine_sim:.6f} (Closer to 1 is better)")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLoG Federated Learning & DLG Simulation")
    parser.add_argument('--defense', action='store_true', help='Enable FedLoG defense mechanisms (Feature Scaling & Noise)')
    args = parser.parse_args()
    main(args)

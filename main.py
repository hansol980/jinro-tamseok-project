import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from model import get_resnet8_modified, get_resnet56_modified
from client import FLClient
from server import FLServer
from dlg import dlg_attack
from utils import get_args, visualize_results

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar100_data(index=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    img, label = dataset[index]
    return img.unsqueeze(0), torch.tensor([label])

def main():
    set_seed(42)
    args = get_args()
    print(f"--- Federated Learning Simulation Started ---")
    print(f"Compression Method: {args.compression}\n")
    
    # 1. 모델 및 서버 초기화
    model = get_resnet8_modified(num_classes=100) 
    server = FLServer(model)
    
    num_clients = 3
    clients_ground_truth = []
    
    # 2. 다수의 클라이언트 시뮬레이션 (각기 다른 데이터를 보유)
    for i in range(num_clients):
        # 서로 다른 이미지를 할당하기 위해 index 변경
        gt_data, gt_label = get_cifar100_data(index=15 + i) 
        clients_ground_truth.append((gt_data, gt_label))
        
        client = FLClient(model, gt_data, gt_label)
        
        # 3. 각 클라이언트가 그래디언트를 계산 및 압축하여 서버로 전송
        compressed_grads, _, _ = client.compute_and_send_gradients(
            compression_method=args.compression, 
            sparsity=args.sparsity
        )
        
        server.receive_gradients(compressed_grads)
        print(f"Client {i} computed and sent gradients. (Label: {gt_label.item()})")
    
    # 4. 공격 페이즈: 악의적인 서버가 평균화 이전에 'Client 0'의 그래디언트를 빼돌려 DLG 공격 수행
    target_client_idx = 0
    target_grads = server.get_target_client_gradients(target_client_idx)
    target_gt_data, target_gt_label = clients_ground_truth[target_client_idx]
    
    print(f"\n[!] Malicious Server triggers DLG attack on Client {target_client_idx}'s isolated gradients...")
    recovered_data, recovered_label = dlg_attack(
        model=model, 
        target_gradients=target_grads, 
        data_shape=(1, 3, 32, 32), 
        num_classes=100,
        num_iterations=300
    )
    
    # 5. 정상적인 연합학습 페이즈: 서버가 모든 클라이언트의 그래디언트를 평균내어 글로벌 모델 업데이트
    server.aggregate_and_update(lr=0.01)
    
    # 6. 공격 결과 평가 및 시각화
    mse = torch.mean((recovered_data - target_gt_data) ** 2).item()
    print("\n--- Attack Results ---")
    print(f"Target Ground Truth Label: {target_gt_label.item()}")
    print(f"Recovered Label: {recovered_label.item()}")
    print(f"MSE between Original and Recovered Image: {mse:.6f}")
    
    title = f"DLG Attack on CIFAR-100 (Compression: {args.compression.capitalize()})"
    if args.compression == 'sparsification':
        title += f" | Sparsity: {args.sparsity}"
        
    visualize_results(target_gt_data, recovered_data, title=title)

if __name__ == "__main__":
    main()
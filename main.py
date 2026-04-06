import torch
import torchvision
import torchvision.transforms as transforms
from model import get_resnet8_modified, get_resnet56_modified
from client import FLClient
from server import FLServer
from dlg import dlg_attack
from utils import get_args, visualize_results

def get_cifar100_data(index=0):
    """CIFAR-100 데이터셋을 다운로드하고 특정 인덱스의 이미지를 반환합니다."""
    transform = transforms.Compose([
        transforms.ToTensor(), # 이미지를 0~1 사이의 텐서로 변환
    ])
    
    # 논문에서 사용한 CIFAR-100 데이터셋 로드
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    img, label = dataset[index]
    
    # 배치 차원 추가: (1, 3, 32, 32)
    return img.unsqueeze(0), torch.tensor([label])

def main():
    args = get_args()
    print(f"--- Federated Learning Simulation Started ---")
    print(f"Compression Method: {args.compression}")
    
    # 1. 모델 초기화 (테스트 속도를 위해 ResNet-8 사용. 논문과 동일한 깊이를 원하면 get_resnet56_modified 사용)
    # CIFAR-100의 클래스 개수는 100개
    model = get_resnet8_modified(num_classes=100) 
    server = FLServer(model)
    
    # 2. 클라이언트 데이터 생성 (CIFAR-100에서 첫 번째 이미지 추출)
    ground_truth_data, ground_truth_label = get_cifar100_data(index=15) 
    print(f"Loaded CIFAR-100 Image. True Label: {ground_truth_label.item()}")
    
    # 3. 클라이언트 동작: 그래디언트 계산 및 압축하여 서버로 전송
    client = FLClient(model, ground_truth_data, ground_truth_label)
    compressed_grads, gt_data, gt_label = client.compute_and_send_gradients(
        compression_method=args.compression, 
        sparsity=args.sparsity
    )
    
    # 서버가 그래디언트 수신
    server.receive_gradients(compressed_grads)
    print("Server received gradients from the client.")
    
    # 4. 악의적인 공격자(서버 스니핑 등)가 DLG 공격을 통해 데이터 복원 시도
    target_grads = server.get_received_gradients()
    recovered_data, recovered_label = dlg_attack(
        model=model, 
        target_gradients=target_grads, 
        data_shape=(1, 3, 32, 32), 
        num_classes=100,  # CIFAR-100은 클래스가 100개이므로 변경
        num_iterations=300
    )
    
    # 5. 결과 평가 및 시각화
    mse = torch.mean((recovered_data - gt_data) ** 2).item()
    print("\n--- Attack Results ---")
    print(f"Recovered Label: {recovered_label.item()} (Ground Truth: {gt_label.item()})")
    print(f"MSE between Original and Recovered Image: {mse:.6f}")
    
    # 압축 방식에 따른 시각화 제목 설정
    title = f"DLG Attack on CIFAR-100 (Compression: {args.compression.capitalize()})"
    if args.compression == 'sparsification':
        title += f" | Sparsity: {args.sparsity}"
        
    visualize_results(gt_data, recovered_data, title=title)

if __name__ == "__main__":
    main()
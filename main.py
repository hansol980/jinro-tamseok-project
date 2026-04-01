import argparse
import torch
from torch.utils.data import DataLoader, Subset

from server import FederatedServer
from client import FederatedClient
from dlg_attack import run_dlg
from utils import get_data, save_image
from models import get_mobilenet_v2

def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Compression and DLG Attack")
    parser.add_argument('--compress', type=str, default='none', 
                        choices=['none', 'quant', 'sparse'], 
                        help="Gradient 압축 방식: 'none'(원본), 'quant'(8비트 양자화), 'sparse'(Top-10% 희소화)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== 실험 모드: [{args.compress.upper()}] | 디바이스: {device} =====")

    # 1. 환경 및 데이터 초기화
    server = FederatedServer()
    dataset = get_data("CIFAR10")
    
    # DLG 타겟을 명확히 하기 위해 데이터 1개(인덱스 0)만 사용하는 클라이언트 구성
    client_loader = DataLoader(Subset(dataset, [0]), batch_size=1)
    client = FederatedClient(client_id=1, train_loader=client_loader)

    # 2. 연합학습(FL) 시뮬레이션
    global_weights = server.get_global_weights()
    local_weights, compressed_grads = client.update_local_model(global_weights, compress_mode=args.compress)
    server.aggregate_weights([local_weights])

    # 3. DLG 공격 시뮬레이션
    # 공격자는 배포된 모델 가중치와 클라이언트가 보낸 Gradient를 탈취했다고 가정
    attack_model = get_mobilenet_v2().to(device)
    attack_model.load_state_dict(global_weights)
    
    # BN(Batch Normalization) 레이어가 DLG에 미치는 영향을 고정하기 위해 eval() 사용
    attack_model.eval() 

    reconstructed_img = run_dlg(attack_model, compressed_grads, device=device)
    
    # 4. 결과 저장
    result_filename = f"dlg_result_{args.compress}.png"
    save_image(reconstructed_img, result_filename)
    print(f"===== 실험 종료: 복원된 이미지가 '{result_filename}'에 저장되었습니다. =====")

if __name__ == "__main__":
    main()
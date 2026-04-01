import torch
import copy
from models import get_mobilenet_v2

class FederatedServer:
    def __init__(self, num_classes=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = get_mobilenet_v2(num_classes).to(self.device)

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_weights(self, local_weights_list):
        # FedAvg: 가중치 산술 평균
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights_list)):
                avg_weights[key] += local_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
        
        self.global_model.load_state_dict(avg_weights)
        print("서버: 글로벌 모델(FedAvg) 업데이트 완료.")
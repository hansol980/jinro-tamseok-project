import torch
import torch.nn as nn
from models import get_mobilenet_v2
from utils import apply_compression

class FederatedClient:
    def __init__(self, client_id, train_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_mobilenet_v2().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def update_local_model(self, global_weights, compress_mode='none'):
        self.model.load_state_dict(global_weights)
        self.model.eval() 
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # DLG 실험을 위해 첫 번째 배치(1개 이미지)만 학습
        images, labels = next(iter(self.train_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        
        optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        
        # Gradient 추출 및 압축 적용 (양자화/희소화)
        raw_grads = [p.grad.detach().clone() for p in self.model.parameters()]
        compressed_grads = [apply_compression(g, mode=compress_mode) for g in raw_grads]
        
        optimizer.step()
        print(f"클라이언트 {self.client_id}: 학습 및 Gradient 압축({compress_mode}) 완료.")
        
        return self.model.state_dict(), compressed_grads
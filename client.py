import torch
import torch.nn.functional as F
from utils import compress_gradients
from pruning import apply_gradient_pruning

class FLClient:
    def __init__(self, model, data, label):
        self.model = model
        self.data = data
        self.label = label

    def compute_and_send_gradients(self, compression_method, sparsity):
        self.model.zero_grad()
        
        # 로컬 데이터로 그래디언트 계산
        output = self.model(self.data)
        loss = F.cross_entropy(output, self.label)
        loss.backward()
        
        # 모델의 파라미터 그래디언트 추출
        original_grads = [param.grad.clone() for param in self.model.parameters()]
        
        # 설정된 압축 방식 적용
        compressed_grads = compress_gradients(original_grads, method=compression_method, sparsity=sparsity)
        
        return compressed_grads, self.data, self.label # 시뮬레이션을 위해 원본 데이터도 반환 (실제로는 반환 안함)
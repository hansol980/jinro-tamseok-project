import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FedLoGModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_syn=20):
        super(FedLoGModel, self).__init__()
        # 2-Layer GraphSAGE 기반 통과 임베더로 캐파 확장
        self.gnn1 = SAGEConv(in_channels, hidden_channels * 2)
        self.gnn2 = SAGEConv(hidden_channels * 2, hidden_channels)
        
        # 학습 가능한 합성 노드 파라미터 [cite: 180, 182]
        self.syn_head = nn.Parameter(torch.randn(out_channels * num_syn, in_channels))
        self.syn_tail = nn.Parameter(torch.randn(out_channels * num_syn, in_channels))
        
        self.num_classes = out_channels
        self.s = num_syn
        self.feature_scale = 1.0  # 방어 기법(Feature Scaling)을 위한 스케일링 팩터

    def extract_embedding(self, x, edge_index):
        # 과적합 방지(Dropout)와 비선형성(ReLU)이 추가된 2계층 은닉망
        h = self.gnn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gnn2(h, edge_index)
        return h

    def forward(self, x, edge_index, node_degrees, lambda_val=3):
        # PyTorch Geometric에서 edge_index로 None을 허용하지 않으므로, 
        # 연결선이 없는 경우에는 크기(2, 0)의 빈 텐서를 생성해 사용합니다.
        device = x.device
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        # 로컬 노드 및 합성 노드 2-Layer 깊은 다중 임베딩
        h = self.extract_embedding(x, edge_index)
        
        # 특징 스케일링(Feature Scaling) 적용
        scaled_syn_head = self.syn_head * self.feature_scale
        scaled_syn_tail = self.syn_tail * self.feature_scale
        
        h_syn_h = self.extract_embedding(scaled_syn_head, empty_edge_index)
        h_syn_t = self.extract_embedding(scaled_syn_tail, empty_edge_index)

        # 프로토타입 기반 거리 계산 함수 [cite: 195, 199]
        def get_logits(emb, syn_emb):
            proto = syn_emb.view(self.num_classes, self.s, -1).mean(dim=1)
            return -torch.cdist(emb, proto, p=2)

        # Degree 기반 가중치 결합 (alpha) [cite: 201, 205]
        alpha = torch.sigmoid(node_degrees.float() - (lambda_val + 1)).view(-1, 1)
        return alpha * get_logits(h, h_syn_h) + (1 - alpha) * get_logits(h, h_syn_t)

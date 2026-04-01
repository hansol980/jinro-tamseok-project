import torch
import torch.nn.functional as F
from utils import cross_entropy_for_onehot

def run_dlg(target_model, target_gradients, device="cpu"):
    dummy_data = torch.randn((1, 3, 32, 32)).to(device).requires_grad_(True)
    dummy_label = torch.randn((1, 10)).to(device).requires_grad_(True)
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)
    
    print(">>> DLG 공격 최적화 시작...")
    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = target_model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
            
            # 더미 데이터로 Gradient 계산
            dummy_grads = torch.autograd.grad(dummy_loss, target_model.parameters(), create_graph=True)
            
            # 실제(혹은 압축된) Gradient와 가짜 Gradient 간의 L2 거리 계산
            grad_diff = 0
            for dg, tg in zip(dummy_grads, target_gradients):
                grad_diff += ((dg - tg) ** 2).sum()
            
            grad_diff.backward()
            return grad_diff
        
        optimizer.step(closure)
        if iters % 30 == 0:
            current_loss = closure()
            print(f"Iter {iters:3d}: Loss {current_loss.item():.8f}")

    print(">>> DLG 공격 최적화 완료.")
    return dummy_data.detach()
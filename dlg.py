import torch
import torch.nn.functional as F

def tv_loss(x):
    """Total Variation Loss: 인접 픽셀 간의 차이를 패널티로 부여하여 노이즈 억제"""
    diff_h = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    diff_w = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return diff_h + diff_w

def dlg_attack(model, target_gradients, data_shape, num_classes=100, num_iterations=500):
    print("\n--- Starting Improved DLG Attack (Cosine Similarity + TV Loss) ---")
    
    # 더미 데이터 초기화 (픽셀 발산을 막기 위해 0.5 주변에서 시작)
    dummy_data = torch.randn(data_shape, requires_grad=True)
    dummy_data.data = dummy_data.data * 0.1 + 0.5 
    
    dummy_label_logits = torch.randn((data_shape[0], num_classes), requires_grad=True)
    
    # L-BFGS 대신 파라미터 튜닝에 덜 민감하고 안정적인 Adam 사용
    optimizer = torch.optim.Adam([dummy_data, dummy_label_logits], lr=0.05)
    
    for iters in range(num_iterations):
        optimizer.zero_grad()
        
        # 더미 데이터로 그래디언트 계산
        dummy_pred = model(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label_logits, dim=-1)
        dummy_loss = torch.sum(-dummy_onehot_label * F.log_softmax(dummy_pred, dim=-1))
        
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        
        # 1. Cosine Similarity Loss (L2 Distance 대신 그래디언트의 방향성 매칭)
        grad_diff = 0
        for gx, gy in zip(dummy_grads, target_gradients):
            if gy is not None:
                # 코사인 유사도는 1에 가까울수록 좋으므로, Loss는 1에서 뺀 값을 최소화
                grad_diff += 1.0 - F.cosine_similarity(gx.flatten(), gy.flatten(), dim=0)
        
        # 2. Total Variation Loss (이미지를 부드럽게 만들어 아티팩트 감소)
        tv_penalty = 1e-4 * tv_loss(dummy_data)
        
        # 최종 Loss
        loss = grad_diff + tv_penalty
        loss.backward()
        
        optimizer.step()
        
        # 최적화 후 이미지 픽셀 값이 정상적인 색상 범위(0~1)를 벗어나지 않도록 클리핑
        with torch.no_grad():
            dummy_data.data.clamp_(0, 1)
        
        if iters % 50 == 0:
            print(f"Iteration {iters}: Total Loss = {loss.item():.6f} (Grad Diff: {grad_diff.item():.6f})")
            
    dummy_label = torch.argmax(dummy_label_logits, dim=-1)
    return dummy_data, dummy_label
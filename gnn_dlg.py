import torch
import torch.nn.functional as F

def cosine_similarity_loss(dummy_grads, true_grads):
    """Cosine Similarity 기반 목적 함수 (Geiping et al., 2020)"""
    flat_dummy = torch.cat([g.view(-1) for g in dummy_grads if g is not None])
    flat_true = torch.cat([g.view(-1) for g in true_grads if g is not None])
    cos_sim = F.cosine_similarity(flat_dummy.unsqueeze(0), flat_true.unsqueeze(0), eps=1e-8)
    return 1.0 - cos_sim.squeeze()

def dlg_attack_gnn(model, target_gradients, data, target_idx, true_label, num_iterations=200, attack_lr=0.05, num_restarts=1, optimize_scale=False):
    if optimize_scale:
        print(f"\n--- Starting ADAPTIVE DLG Attack for GNN (Node {target_idx}) ---")
    else:
        print(f"\n--- Starting Advanced DLG Attack for GNN (Node {target_idx}) ---")
    
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
        
    target_label_tensor = torch.tensor([true_label], dtype=torch.long, device=data.x.device)
    fixed_x = data.x.detach().clone()
    target_mask = torch.zeros((data.num_nodes, 1), device=data.x.device)
    target_mask[target_idx] = 1.0

    best_dummy_x = None
    best_dummy_scale = None
    best_loss = float('inf')

    for restart in range(num_restarts):
        # 복원할 더미 피처
        dummy_x = torch.randn((1, data.num_features), device=data.x.device, requires_grad=True)
        
        # Adaptive Attack: 피처 스케일링 계수까지 조인트 최적화 (Joint Optimization)
        if optimize_scale:
            # 스케일을 1.0부터 탐색 시작
            dummy_scale = torch.tensor([1.0], device=data.x.device, requires_grad=True)
            optimizer_params = [dummy_x, dummy_scale]
        else:
            dummy_scale = None
            optimizer_params = [dummy_x]
            
        optimizer = torch.optim.LBFGS(optimizer_params, lr=attack_lr, max_iter=20, history_size=20, line_search_fn='strong_wolfe')
        
        for iters in range(max(1, num_iterations // 20)):
            def closure():
                optimizer.zero_grad()
                expanded_dummy_x = dummy_x.expand(data.num_nodes, -1)
                mixed_x = fixed_x * (1.0 - target_mask) + expanded_dummy_x * target_mask
                
                # Adaptive Attack 적용 시, 추정한 dummy_scale을 모델에 덮어씌움
                if optimize_scale:
                    model.feature_scale = dummy_scale

                
                # 모델 포워드 (FedLoG 모델은 x, edge_index, node_degrees를 받음)
                out = model(mixed_x, data.edge_index, data.degrees)
                
                dummy_loss = F.cross_entropy(out[target_idx:target_idx+1], target_label_tensor)
                dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True, allow_unused=True)
                
                valid_dummy_grads = [g for g in dummy_grads if g is not None]
                valid_true_grads = [gt for gd, gt in zip(dummy_grads, target_gradients) if gd is not None]
                
                grad_diff = cosine_similarity_loss(valid_dummy_grads, valid_true_grads)
                
                # L2 정규화 (노이즈 발산 방지)
                alpha_l2 = 0.001
                l2_loss = alpha_l2 * torch.norm(dummy_x, p=2)
                
                loss = grad_diff + l2_loss
                loss.backward()
                return loss
            
            optimizer.step(closure)
            current_loss = closure().item()
            
            if not torch.isnan(torch.tensor(current_loss)) and current_loss < best_loss:
                best_loss = current_loss
                best_dummy_x = dummy_x.detach().clone()
                if optimize_scale:
                    best_dummy_scale = dummy_scale.detach().clone()
                
            if iters % 5 == 0 or iters == (num_iterations // 20) - 1:
                scale_str = f", Scale: {dummy_scale.item():.4f}" if optimize_scale else ""
                print(f"L-BFGS Step {iters*20:3d}/{num_iterations} | Loss: {current_loss:.6f}{scale_str}")

    if best_dummy_x is None:
        best_dummy_x = torch.zeros((1, data.num_features), device=data.x.device)

    # 평가 지표
    true_feature = data.x[target_idx:target_idx+1]
    mse = F.mse_loss(best_dummy_x, true_feature).item()
    cosine_sim = F.cosine_similarity(best_dummy_x, true_feature).item()
    
    return best_dummy_x, best_dummy_scale, mse, cosine_sim

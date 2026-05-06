import torch

def apply_gradient_pruning(gradients, prune_rate=0.2):
    """
    클라이언트가 서버로 전송하기 전, 기울기(gradients)의 하위 prune_rate 비율을 0으로 만듭니다.
    
    Args:
        gradients (list of torch.Tensor): 모델의 파라미터별 기울기 텐서들이 담긴 리스트
        prune_rate (float): 잘라낼 비율 (0.0 ~ 1.0). 예: 0.2면 하위 20%를 0으로 만듦
        
    Returns:
        pruned_gradients (list of torch.Tensor): 압축(가지치기)이 완료된 기울기 리스트
    """
    if prune_rate <= 0.0:
        return gradients

    flattened_grads = torch.cat([g.view(-1) for g in gradients])
    
    num_params = flattened_grads.numel()
    k = int(num_params * prune_rate)
    
    sorted_abs_grads, _ = torch.sort(torch.abs(flattened_grads))
    threshold = sorted_abs_grads[k]
    
    pruned_gradients = []
    for g in gradients:
        mask = (torch.abs(g) >= threshold).float()
        
        pruned_g = g * mask 
        pruned_gradients.append(pruned_g)
        
    return pruned_gradients
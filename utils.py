import argparse
import torch
import matplotlib.pyplot as plt
from pruning import apply_gradient_pruning

def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning with Compression and DLG Attack")
    parser.add_argument('--compression', type=str, default='none', 
                        choices=['none', 'sparsification', 'quantization', 'pruning'],
                        help="Choose gradient compression method (none, sparsification, quantization, pruning)")
    parser.add_argument('--sparsity', type=float, default=0.2, 
                        help="Sparsity ratio for sparsification (e.g., 0.2 means prune bottom 20%)")
    return parser.parse_args()

def compress_gradients(gradients, method, sparsity=0.2):
    compressed_grads = []
    if method == 'none':
        return gradients
        
    elif method == 'sparsification':
        for g in gradients:
            if g is None:
                compressed_grads.append(None)
                continue
            threshold = torch.quantile(torch.abs(g), sparsity)
            mask = torch.abs(g) >= threshold
            compressed_grads.append(g * mask)
            
    elif method == 'quantization':
        for g in gradients:
            if g is None:
                compressed_grads.append(None)
                continue
            compressed_grads.append(g.half().float())

    # DLG 논문의 기본 방어 기법인 pruning(가지치기) 적용
    elif method == 'pruning':
        valid_grads = [g for g in gradients if g is not None]
        pruned_valid = apply_gradient_pruning(valid_grads, sparsity)
        
        pruned_idx = 0
        for g in gradients:
            if g is None:
                compressed_grads.append(None)
            else:
                compressed_grads.append(pruned_valid[pruned_idx])
                pruned_idx += 1
            
    return compressed_grads

def visualize_results(original_data, recovered_data, title="DLG Attack Result"):
    """원본 이미지와 복원된 이미지를 비교하여 시각화합니다."""
    # 텐서를 numpy 배열로 변환 (1, 3, 32, 32) -> (32, 32, 3)
    orig_img = original_data[0].detach().cpu().permute(1, 2, 0).numpy()
    recv_img = recovered_data[0].detach().cpu().permute(1, 2, 0).numpy()

    # 화면 출력을 위해 [0, 1] 범위로 정규화
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
    recv_img = (recv_img - recv_img.min()) / (recv_img.max() - recv_img.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Ground Truth (Original)")
    axes[0].axis("off")

    axes[1].imshow(recv_img)
    axes[1].set_title("Recovered Image (DLG)")
    axes[1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
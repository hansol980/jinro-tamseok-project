import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_data(dataset_name="CIFAR10"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    if dataset_name == "CIFAR10":
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    return train_ds

def cross_entropy_for_onehot(inputs, targets):
    return torch.mean(torch.sum(-targets * torch.log_softmax(inputs, dim=-1), dim=-1))

def apply_compression(tensor, mode='none'):
    if mode == 'quant':
        # 8-bit 양자화 (Min-Max)
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.round((tensor - min_val) / scale)
        dequantized = (quantized * scale) + min_val
        return dequantized
        
    elif mode == 'sparse':
        # Top-10% 희소화 (나머지는 0으로 마스킹)
        k = max(1, int(tensor.numel() * 0.1))
        values, indices = torch.topk(torch.abs(tensor.view(-1)), k)
        mask = torch.zeros_like(tensor.view(-1))
        mask[indices] = 1.0
        return (tensor.view(-1) * mask).view(tensor.shape)
        
    return tensor

def save_image(tensor, path):
    # DLG 결과 텐서 (1, 3, H, W)를 이미지로 저장
    img = tensor[0].detach().cpu().permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min()) # 0~1 정규화
    plt.imshow(img.numpy())
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
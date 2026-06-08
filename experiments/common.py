# -*- coding: utf-8 -*-
"""
실험 공통 유틸리티.

- set_seed: 재현성 확보(난수 시드 고정)
- 방어 함수: DP-SGD식(그래디언트 클리핑 + 보정 노이즈), 극단적 희소화
- 지표: 그래디언트 충실도(utility proxy), 코사인/MSE 헬퍼

기존 코드가 빠뜨린 부분(시드 미고정, 비클리핑 노이즈, 유용성 미측정)을 보완한다.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    """모든 난수원(파이썬/넘파이/토치)을 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 방어 기법 (그래디언트 리스트에 적용)
# ---------------------------------------------------------------------------
def grad_global_norm(grads):
    sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return torch.sqrt(sq + 1e-12)


def dp_defense(grads, clip_norm=1.0, noise_multiplier=1.0):
    """
    DP-SGD 양식의 방어 (Abadi et al., 2016).

    기존 코드의 결함: 단순히 가우시안 노이즈만 더해 '차분 프라이버시'라 불렀으나,
    실제 DP-SGD는 (1) 그래디언트 L2 노름 클리핑으로 민감도를 C로 제한한 뒤
    (2) 표준편차 sigma = noise_multiplier * C 의 가우시안 노이즈를 더해야 한다.

    여기서는 (단일 타겟의) 전역 그래디언트 노름을 C로 클리핑한 뒤 노이즈를 주입한다.
    형식적 프라이버시 예산(epsilon)은 합성정리로 별도 회계해야 하며 본 구현은
    노이즈 배율(noise_multiplier)로 강도를 제어한다.
    """
    total_norm = grad_global_norm(grads)
    scale = min(1.0, clip_norm / float(total_norm))
    sigma = noise_multiplier * clip_norm
    out = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        gc = g * scale
        out.append(gc + torch.randn_like(gc) * sigma)
    return out


def sparsify_defense(grads, sparsity=0.95):
    """크기 기준 하위 sparsity 비율을 0으로 만들고 상위 (1-sparsity)만 전송."""
    out = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        flat = g.reshape(-1)
        k = int(flat.numel() * sparsity)
        if k > 0:
            _, idx = torch.topk(flat.abs(), k=k, largest=False)
            mask = torch.ones_like(flat)
            mask[idx] = 0.0
            out.append((flat * mask).reshape(g.shape))
        else:
            out.append(g.clone())
    return out


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def _flatten_aligned(defended, original):
    a, b = [], []
    for gd, go in zip(defended, original):
        if gd is None or go is None:
            continue
        a.append(gd.reshape(-1))
        b.append(go.reshape(-1))
    # float64로 캐스팅: 수백만 원소 벡터의 코사인은 float32 누적오차로 1을 초과할 수 있음
    return torch.cat(a).double(), torch.cat(b).double()


def _cos(a, b):
    """수치 안정 코사인 유사도(double 연산, [-1,1] 클램프)."""
    denom = a.norm() * b.norm()
    if denom <= 0:
        return 0.0
    return float(torch.clamp(torch.dot(a, b) / denom, -1.0, 1.0))


def gradient_fidelity(defended, original):
    """
    유용성(utility) 프록시: 방어가 적용된 그래디언트가 원본 그래디언트와
    방향적으로 얼마나 보존되는지(코사인 유사도). 1에 가까울수록 학습 신호
    손상이 적어 모델 유용성이 잘 보존됨을 의미한다.
    """
    a, b = _flatten_aligned(defended, original)
    return _cos(a, b)


def mse(a, b):
    return F.mse_loss(a, b).item()


def cosine(a, b):
    return _cos(a.reshape(-1).double(), b.reshape(-1).double())


def summarize(values):
    """리스트 -> (mean, std). std는 표본표준편차(ddof=1)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))

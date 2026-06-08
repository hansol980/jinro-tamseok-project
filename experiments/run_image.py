# -*- coding: utf-8 -*-
"""
이미지(CIFAR-100 + 수정 ResNet) 그래디언트 유출 — 멀티시드 엄밀 실험.

기존 fedAvg/main.py 대비 보완점
  1) 시드별로 서로 다른 타겟 이미지를 공격하여 n=1 단일 사례 편향 제거
  2) 방어 기법별 복원 MSE를 N_SEEDS회 측정해 mean±std 보고
  3) 라벨 복원 정확도, 그래디언트 충실도(유용성 프록시)까지 기록
  4) 결과를 JSONL로 체크포인트 저장

주의: CIFAR-100을 CPU에서 수렴까지 학습하는 것은 비현실적이므로, 본 트랙은
원본과 동일하게 (무학습) 초기 모델을 공격하는 worst-case(=표준 DLG) 설정을 쓴다.
학습 진행에 따른 유출 감소 추세는 GNN 트랙(run_gnn.py --rounds)에서 별도 검증한다.
"""
import os, sys, json, time, argparse, random
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fedAvg"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_resnet8_modified           # noqa: E402
from dlg import dlg_attack                        # noqa: E402
from utils import compress_gradients              # noqa: E402
from soteria import apply_soteria_defense         # noqa: E402
import common                                     # noqa: E402

DEVICE = torch.device("cpu")
ATTACK_ITERS = 300
# (이름, 방어 method, sparsity)
CONFIGS = [
    ("none",        "none",            0.0),
    ("sparse_0.2",  "sparsification",  0.2),
    ("sparse_0.5",  "sparsification",  0.5),
    ("sparse_0.8",  "sparsification",  0.8),
    ("quant",       "quantization",    0.0),
    ("pruning_0.2", "pruning",         0.2),
    ("soteria_0.2", "soteria",         0.2),
]


def load_dataset():
    tf = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR100(root=os.path.join(ROOT, "data"),
                                         train=True, download=True, transform=tf)


def defended_gradients(model, data, label, method, sparsity):
    """client.py 로직 재현: 그래디언트 계산 -> (soteria) -> 압축."""
    model.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, label)
    loss.backward()
    if method == "soteria":
        apply_soteria_defense(model, data, defended_layer_name="linear", prune_rate=sparsity)
    grads = [p.grad.clone() for p in model.parameters()]
    return compress_gradients(grads, method=method, sparsity=sparsity)


def run_seed(seed, dataset, writer, configs, iters):
    common.set_seed(seed)
    # 시드마다 다른 타겟 이미지
    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]
    data = img.unsqueeze(0).to(DEVICE)
    label_t = torch.tensor([label], device=DEVICE)

    # 동일 초기화 모델(시드 고정)에서 깨끗한 그래디언트(none) 기준 확보
    common.set_seed(seed)
    base_model = get_resnet8_modified(num_classes=100).to(DEVICE)
    clean = defended_gradients(base_model, data, label_t, "none", 0.0)

    for name, method, sparsity in configs:
        if (seed, name) in run_seed.done:
            continue
        t0 = time.time()
        # 매 설정마다 동일 초기화 모델 재생성(방어가 grad에 in-place 영향 주므로)
        common.set_seed(seed)
        model = get_resnet8_modified(num_classes=100).to(DEVICE)
        sent = defended_gradients(model, data, label_t, method, sparsity)
        fidelity = common.gradient_fidelity(sent, clean)

        rec_data, rec_label = dlg_attack(
            model=model, target_gradients=sent,
            data_shape=(1, 3, 32, 32), num_classes=100, num_iterations=iters,
        )
        mse = common.mse(rec_data, data)
        label_ok = int(rec_label.item() == label)
        rec = {
            "track": "image", "seed": seed, "config": name,
            "img_index": idx, "true_label": label,
            "mse": mse, "label_correct": label_ok, "fidelity": fidelity,
            "secs": round(time.time() - t0, 1),
        }
        writer.write(json.dumps(rec) + "\n")
        writer.flush()
        print(f"[seed {seed}] {name:11s} mse={mse:.6f} label_ok={label_ok} "
              f"fid={fidelity:.3f} ({rec['secs']}s)", flush=True)


run_seed.done = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=ATTACK_ITERS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "image.jsonl"))
    args = ap.parse_args()

    print("Loading CIFAR-100 ...", flush=True)
    dataset = load_dataset()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                run_seed.done.add((r["seed"], r["config"]))
            except Exception:
                pass

    with open(args.out, "a") as w:
        for seed in range(args.seeds):
            run_seed(seed, dataset, w, CONFIGS, args.iters)
    print("Image experiments complete.", flush=True)


if __name__ == "__main__":
    main()

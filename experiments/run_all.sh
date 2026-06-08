#!/bin/bash
# 전체 엄밀 실험을 순차 실행. 각 드라이버는 JSONL에 체크포인트하므로 중단 후 재실행 가능.
set -u
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
SEEDS="${1:-5}"
echo "### SEEDS=$SEEDS ###"

echo "### [1/3] GNN 방어 비교 (rounds=0, 미학습 worst-case) ###"
python3 -u experiments/run_gnn.py --seeds "$SEEDS" --rounds 0 \
    --out experiments/results/gnn.jsonl

echo "### [2/3] GNN 학습 진행 추세 (baseline, rounds=5,20) ###"
for R in 5 20; do
  python3 -u experiments/run_gnn.py --seeds "$SEEDS" --rounds "$R" \
      --scenarios baseline --out experiments/results/gnn.jsonl
done

echo "### [3/3] 이미지 방어 비교 (멀티시드) ###"
python3 -u experiments/run_image.py --seeds "$SEEDS" \
    --out experiments/results/image.jsonl

echo "ALL EXPERIMENTS COMPLETE"

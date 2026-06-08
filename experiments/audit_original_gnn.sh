#!/bin/bash
# 원본 main_gnn.py를 (시드 미고정 상태 그대로) 여러 번 실행해
# 보고된 단일 수치(0.9999 등)가 대표값인지 분산을 측정한다.
set -u
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
OUT="experiments/results/audit_gnn.txt"
: > "$OUT"

run_n () {
  local label="$1"; shift
  local n="$1"; shift
  for i in $(seq 1 "$n"); do
    echo "=== $label run $i ===" >> "$OUT"
    python3 main_gnn.py "$@" 2>/dev/null \
      | grep -E "Recovered Feature (MSE|Cosine)|Secret Scale|Stolen" >> "$OUT"
    echo "[$label $i] done"
  done
}

run_n baseline 5
run_n adaptive 5 --defense --adaptive_attack
echo "AUDIT COMPLETE"

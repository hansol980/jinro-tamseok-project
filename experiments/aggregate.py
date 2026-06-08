# -*- coding: utf-8 -*-
"""
실험 결과(JSONL + 원본 감사 로그)를 읽어 mean±std로 집계하고
results.json 및 마크다운 표로 출력한다.
"""
import os, json, re, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def dedup(rows, key_fields):
    """동일 key의 마지막 레코드만 사용(재실행 중복 제거)."""
    d = {}
    for r in rows:
        d[tuple(r.get(k) for k in key_fields)] = r
    return list(d.values())


def stat(vals):
    a = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))],
                 dtype=np.float64)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def fmt(s, prec=4):
    if s is None:
        return "—"
    return f"{s['mean']:.{prec}f} ± {s['std']:.{prec}f}"


# --------------------------------------------------------------------------
def agg_gnn():
    rows = dedup(load_jsonl(os.path.join(RES, "gnn.jsonl")), ["seed", "scenario", "rounds"])
    out = {"defense": {}, "trend": {}}
    # 방어 비교(rounds=0)
    for scen in ["baseline", "fedlog", "adaptive", "dp", "sparse"]:
        rs = [r for r in rows if r["scenario"] == scen and r.get("rounds", 0) == 0]
        if not rs:
            continue
        out["defense"][scen] = {
            "cosine": stat([r["cosine"] for r in rs]),
            "mse": stat([r["mse"] for r in rs]),
            "fidelity": stat([r["fidelity"] for r in rs]),
            "scale_err": stat([r["scale_err"] for r in rs]),
            "n": len(rs),
        }
    # 학습 추세(baseline, rounds별)
    rounds_set = sorted(set(r.get("rounds", 0) for r in rows if r["scenario"] == "baseline"))
    for rd in rounds_set:
        rs = [r for r in rows if r["scenario"] == "baseline" and r.get("rounds", 0) == rd]
        out["trend"][rd] = {
            "cosine": stat([r["cosine"] for r in rs]),
            "accuracy": stat([r["accuracy"] for r in rs]),
            "n": len(rs),
        }

    # 쌍체(시드 내 같은 타겟 노드) 효과: 절대값 분산이 커도 효과는 일관됨
    by = {}
    for r in rows:
        if r.get("rounds", 0) == 0:
            by.setdefault(r["seed"], {})[r["scenario"]] = r["cosine"]
    out["paired"] = {}
    for name, a, b in [("scaling_effect", "fedlog", "baseline"),
                       ("adaptive_break", "adaptive", "fedlog"),
                       ("dp_defense", "dp", "adaptive"),
                       ("sparse_defense", "sparse", "adaptive")]:
        deltas = [by[s][a] - by[s][b] for s in by if a in by[s] and b in by[s]]
        out["paired"][name] = {**(stat(deltas) or {}), "from": b, "to": a, "deltas": deltas}
    return out


def agg_image():
    rows = dedup(load_jsonl(os.path.join(RES, "image.jsonl")), ["seed", "config"])
    order = ["none", "sparse_0.2", "sparse_0.5", "sparse_0.8", "quant", "pruning_0.2", "soteria_0.2"]
    out = {}
    for cfg in order:
        rs = [r for r in rows if r["config"] == cfg]
        if not rs:
            continue
        out[cfg] = {
            "mse": stat([r["mse"] for r in rs]),
            "label_acc": stat([r["label_correct"] for r in rs]),
            "fidelity": stat([r["fidelity"] for r in rs]),
            "n": len(rs),
        }
    return out


def agg_audit():
    """원본 main_gnn.py 다중 실행 로그 파싱."""
    path = os.path.join(RES, "audit_gnn.txt")
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    blocks = re.split(r"=== (\w+) run \d+ ===", text)
    # blocks: ['', label, body, label, body, ...]
    data = {}
    for i in range(1, len(blocks), 2):
        label = blocks[i]
        body = blocks[i + 1]
        cos = re.search(r"Cosine Similarity:\s*([-\d.]+)", body)
        mse = re.search(r"MSE:\s*([\d.]+)", body)
        data.setdefault(label, {"cosine": [], "mse": []})
        if cos:
            data[label]["cosine"].append(float(cos.group(1)))
        if mse:
            data[label]["mse"].append(float(mse.group(1)))
    out = {}
    for label, d in data.items():
        out[label] = {"cosine": stat(d["cosine"]), "mse": stat(d["mse"])}
    return out


def main():
    result = {"gnn": agg_gnn(), "image": agg_image(), "audit": agg_audit()}
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n================ 원본 코드 변동성 감사 (main_gnn.py, 시드 미고정) ================")
    for label, d in result["audit"].items():
        print(f"  {label:9s}: cosine {fmt(d['cosine'])}  "
              f"(범위 {d['cosine']['min']:.3f}~{d['cosine']['max']:.3f}, n={d['cosine']['n']})"
              if d['cosine'] else f"  {label}: (no data)")

    print("\n================ GNN 방어 비교 (미학습 rounds=0, 멀티시드) ================")
    print(f"  {'scenario':10s} {'cosine(복원)':>20s} {'MSE':>20s} {'fidelity(유용성)':>20s}")
    for scen, d in result["gnn"]["defense"].items():
        print(f"  {scen:10s} {fmt(d['cosine']):>20s} {fmt(d['mse'],6):>20s} {fmt(d['fidelity'],3):>20s}  (n={d['n']})")

    print("\n================ GNN 학습 진행에 따른 유출 추세 (baseline) ================")
    for rd in sorted(result["gnn"]["trend"]):
        d = result["gnn"]["trend"][rd]
        print(f"  rounds={rd:3d}: cosine {fmt(d['cosine'])}  accuracy {fmt(d['accuracy'],3)}  (n={d['n']})")

    print("\n================ 이미지 방어별 복원 (멀티시드) ================")
    print(f"  {'config':12s} {'MSE':>20s} {'label복원률':>14s} {'fidelity':>16s}")
    for cfg, d in result["image"].items():
        print(f"  {cfg:12s} {fmt(d['mse'],6):>20s} {fmt(d['label_acc'],2):>14s} {fmt(d['fidelity'],3):>16s}  (n={d['n']})")

    print(f"\n저장: {os.path.join(RES, 'results.json')}")


if __name__ == "__main__":
    main()

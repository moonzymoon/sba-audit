"""E5 十种子训练队列 (预注册 §6 seed 协议 + R3 扩充).

预注册新种子 {0,1,2,3,4,5,6,8,9,10}; 既有独立种子 {7,31,42,97,123,2024}(第2篇 checkpoint).
本队列逐 (dataset, seed) 训练(跳过已有 checkpoint)并立即推理缓存分数, GPU 单流串行.
优先级: SMD(最快) -> PSM -> SWaT -> MSL -> SMAP -> WADI.
用法: python e5/train_queue.py [--max-runs N]
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/src")
sys.path.insert(0, r"D:/0科研/工作1/第6篇SCI/src")

CKPT_DIR = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/results/checkpoints"
CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
LOG = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results/e5_train_queue.log"

QUEUE = [
    ("SMD", [0, 1, 2, 3]),
    ("PSM", [0, 1, 2, 3]),
    ("SWaT", [0, 1, 2, 3]),
    ("MSL", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]),
    ("SMAP", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]),
    ("WADI", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=100)
    args = ap.parse_args()
    import torch
    from utils.repro import load_config
    cfg = load_config(r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"E5 训练队列启动, device={device}")
    from train import train_one_seed
    from cadms.scorers import cmh_mil_scores
    n = 0
    for ds, seeds in QUEUE:
        for seed in seeds:
            ckpt = os.path.join(CKPT_DIR, f"{ds}_seed{seed}.pt")
            cache = os.path.join(CACHE, f"cmhmil_{ds}_seed{seed}.npz")
            if not os.path.exists(ckpt):
                t0 = time.time()
                try:
                    m = train_one_seed(cfg, seed, ds, device=device)
                    log(f"TRAIN {ds} seed{seed}: {time.time() - t0:.0f}s "
                        f"AUROC={m.get('AUROC', float('nan')):.4f}")
                except Exception as e:  # noqa: BLE001
                    log(f"TRAIN FAIL {ds} seed{seed}: {e}\n{traceback.format_exc()[-500:]}")
                    continue
                n += 1
            if os.path.exists(ckpt) and not os.path.exists(cache):
                try:
                    s, y = cmh_mil_scores(ds, seed=seed, use_cache=False)
                    log(f"SCORE {ds} seed{seed}: T={len(s)} 已缓存")
                except Exception as e:  # noqa: BLE001
                    log(f"SCORE FAIL {ds} seed{seed}: {e}")
            if n >= args.max_runs:
                log(f"达到 max-runs={args.max_runs}, 队列退出")
                return
    log("队列全部完成")


if __name__ == "__main__":
    main()

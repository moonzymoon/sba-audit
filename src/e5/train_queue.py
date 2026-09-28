"""E5 zh:zh_9092seedtrainingzh:zh_4696 (pre-registered §6 seed zh:zh_1063 + R3 extended).

pre-registerednewzh:seed {0,1,2,3,4,5,6,8,9,10}; zh:zh_7758seed {7,31,42,97,123,2024}(paper 2 checkpoint).
zh:zh_1426 (dataset, seed) training(skipzh:zh_6379 checkpoint)zh:zh_2878i.e.zh_4655cachezh:score, GPU zh:zh_4017row.
zh:zh_5354: SMD(zh:zh_2235) -> PSM -> SWaT -> MSL -> SMAP -> WADI.
zh:zh_966: python e5/train_queue.py [--max-runs N]
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, r"D:/0keyan/gongzuo1/paper 2SCI/Contrastive_TopK_MIL/src")
sys.path.insert(0, r"D:/0keyan/gongzuo1/paper 6SCI/src")

CKPT_DIR = r"D:/0keyan/gongzuo1/paper 2SCI/Contrastive_TopK_MIL/results/checkpoints"
CACHE = r"D:/0keyan/gongzuo1/paper 6SCI/src/_score_cache"
LOG = r"D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/e5_train_queue.log"

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
    cfg = load_config(r"D:/0keyan/gongzuo1/paper 2SCI/Contrastive_TopK_MIL/configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"E5 trainingzh:zh_5486, device={device}")
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
                    log(f"SCORE {ds} seed{seed}: T={len(s)} cache")
                except Exception as e:  # noqa: BLE001
                    log(f"SCORE FAIL {ds} seed{seed}: {e}")
            if n >= args.max_runs:
                log(f"zh:zh_787 max-runs={args.max_runs}, zh:zh_1889")
                return
    log("zh:zh_4696alldone")


if __name__ == "__main__":
    main()

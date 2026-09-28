"""E5 准备: 为已有独立训练 checkpoint 补推理分数缓存 (预注册 R3 扩充).

第2篇 checkpoint 库中 SMD/PSM/SWaT 各有 6 个独立种子 {7,31,42,97,123,2024},
其中 31/97/2024 尚无分数缓存 -> 本脚本走 paper6 scorers 的标准推理路径补齐
(小模型前向, GPU 轻载, 逐个串行). 这些是真实独立训练, 与预注册 E5 的
{0..10} 新训练队列互补; 合计后 SMD/PSM/SWaT 将有 9+ 种子.
"""
import os
import sys
import time

sys.path.insert(0, r"D:/0科研/工作1/第6篇SCI/src")
from cadms.scorers import cmh_mil_scores  # noqa: E402

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"

JOBS = [(ds, seed) for ds in ["SMD", "PSM", "SWaT"] for seed in [31, 97, 2024]]


def main():
    for ds, seed in JOBS:
        f = os.path.join(CACHE, f"cmhmil_{ds}_seed{seed}.npz")
        if os.path.exists(f):
            print(f"[skip] {ds} seed{seed} 已有缓存")
            continue
        t0 = time.time()
        s, y = cmh_mil_scores(ds, seed=seed, use_cache=False)
        print(f"[done] {ds} seed{seed}: T={len(s)}, 用时 {time.time() - t0:.0f}s, "
              f"缓存 {os.path.exists(f)}")


if __name__ == "__main__":
    main()

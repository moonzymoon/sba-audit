"""E5 zh:zh_9233: is zh:zh_896training checkpoint zh:zh_2026scorecache (pre-registered R3 extended).

paper 2 checkpoint zh:zh_6398 SMD/PSM/SWaT zh:zh_6400 6  zh:zh_6802seed {7,31,42,97,123,2024},
zh:among them 31/97/2024 zh:zh_6024no zh:scorecache -> zh:zh_33 paper6 scorers  zh:zh_8512pathzh_1912
(zh:zh_5000modelzh_7268, GPU zh:zh_8617, zh:zh_2843 zh:zh_2090row). zh:thesezh_5050realzh:zh_6802training, withpre-registered E5  
{0..10} newtrainingzh:zh_7080; zh:zh_2521 SMD/PSM/SWaT zh:zh_8882 9+ zh:seed.
"""
import os
import sys
import time

sys.path.insert(0, r"D:/0keyan/gongzuo1/paper 6SCI/src")
from cadms.scorers import cmh_mil_scores  # noqa: E402

CACHE = r"D:/0keyan/gongzuo1/paper 6SCI/src/_score_cache"

JOBS = [(ds, seed) for ds in ["SMD", "PSM", "SWaT"] for seed in [31, 97, 2024]]


def main():
    for ds, seed in JOBS:
        f = os.path.join(CACHE, f"cmhmil_{ds}_seed{seed}.npz")
        if os.path.exists(f):
            print(f"[skip] {ds} seed{seed} zh:zh_6379cache")
            continue
        t0 = time.time()
        s, y = cmh_mil_scores(ds, seed=seed, use_cache=False)
        print(f"[done] {ds} seed{seed}: T={len(s)}, zh:zh_3776 {time.time() - t0:.0f}s, "
              f"cache {os.path.exists(f)}")


if __name__ == "__main__":
    main()

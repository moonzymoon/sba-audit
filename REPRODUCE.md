# REPRODUCE - reproduction guide for paper 15 (R58 edition, 2026-09-28)

## 0. Environment

- Python 3.11.9; NumPy 1.26.4 / SciPy 1.17.1 / scikit-learn 1.9.0 / pandas 3.0.5 / Matplotlib 3.11.1 (see `src/requirements.txt`).
- Scripts carry the authors' absolute paths (Windows, `D:\...`); on another machine, replace the `D:/...` prefixes in each script or mirror the same directory layout.
- Score caches (`caches/scores/*.npz`) and resample caches (`caches/resample/*.npz`) ship with the GitHub repository; this SOM zip carries the results layer (`results/`, all CSV/JSON) and all scripts.

## 1. Figure-to-script map

| Figure | Script | Notes |
|---|---|---|
| Fig.1 (tau map) | `python src/make_figures.py` | generates fig1/2/4 |
| Fig.2 (size vs r) | `python src/make_figures.py` | r-version, 43 cells / corr 0.931 |
| Fig.3 (power) | `python src/make_fig_direct.py` | generates fig3/5/6 (direct scale) |
| Fig.4 (win vs blk) | `python src/make_figures.py` | |
| Fig.5 (gap landscape) | `python src/make_fig_direct.py` | red = own-floor criterion (11 pairs) |
| Fig.6 (design chart) | `python src/make_fig_direct.py` | |
| Fig.7 (CR1) | `python src/make_fig789.py` | |
| Fig.8 (pair corr) | `python src/make_fig789.py` | |
| Fig.9 (var decomp) | `python src/make_fig789.py` | |
| Fig.10 (rank stab) | `python src/make_fig_rankstab.py` | |

## 2. Command-line tools

- `python src/audit.py <dataset> <scorer> [B]` — block inventory (raw + qualified K) and the Type I calibration
- `python src/power.py <bm> <pair> [gap]` / `--all` — direct floor and resolvability

## 3. Claim-to-artifact pointers

See the claim-to-artifact traceability table in the main text; every file lives in `results/`.

---


## 22. Friedman omnibus test (R5)
```
cd src && python probe/friedman_omnibus.py   # deterministic性, no resampling; 输出 results/friedman_omnibus.csv
```

## 23. synthetic factorial grid (R6)
```
cd src && python probe/synthetic_grid.py   # 18格 x B=5000, ~7分钟; 输出 results/synthetic_grid.csv
```

## 24. R7 bridge correction recompute (2π→2π(1−π))
```
cd src && python mixed/e4_mde.py && python mixed/e1_full.py   # bridge corrected; 输出 e4_mde_table/e1_full_components
python make_fig56.py && python make_figures.py                 # figs 3/5/6 (corrected) + figs 1/2/4
```
## 25. paired-scale robustness (R7)
```
cd src && python probe/pair_correlation.py   # 输出 results/pair_scale_analysis.csv
```

## 26. CRSE comparison and scenario analysis (R8)
```
cd src && python probe/crse_demo.py   # 输出 results/crse_vs_block.csv + %GRR场景 + 基准表数据
```

27. 图 7-9（CRSE 柱状 / 配对尺度散点 / 方差分解双面板）：`python src/make_fig789.py`（读取 results/ 下 crse_vs_block.csv、pair_scale_analysis.csv、e1_full_components.json，全部现算）。

28. Deep-derivation artifacts (R11): `python src/probe/pr_bridge_mc.py` (MC for AP-bridge nonexistence, 400 replicas x 3 cells -> results/pr_bridge_mc.csv); `python src/probe/length_effects.py` (block-length ratio-estimator effects -> results/length_effects.csv; source of the 18.3 / 5.9 / 126-52336 numbers in Appendix A).

29. deep-derivation artifacts (batch 2)（R14）：`python src/probe/cross_bench_synthesis.py`（cross-benchmark synthesis：FE 合并 MDE/RE 地板/41 套/种子噪声→cross_bench_synthesis.csv）；block concentration 75 对statistics saved by results/block_concentration.csv 保存（source of the anatomy paragraph and the honesty sentence）。

30. R15b direct-scale reproduction：`python src/probe/r15b_compute.py`（8 数据集 V_eval direct + AP + 75 对统计→e1_direct_components.json/r15b_summary.txt）；`python src/probe/bridge_replica_check.py`（bridge replica distortion quantification）；`python src/make_fig_direct.py`（fig3/5/6 direct）；make_fig789.py 现读 e1_direct_components.json（fig9）。

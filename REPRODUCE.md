# REPRODUCE — 第15篇论文复现指南（R58版, 2026-09-28）

## 0. 环境

- Python 3.11.9；NumPy 1.26.4 / SciPy 1.17.1 / scikit-learn 1.9.0 / pandas 3.0.5 / Matplotlib 3.11.1（见 `src/requirements.txt`）。
- 脚本中的路径为作者环境的绝对路径（Windows, `D:\...`）；换环境时替换各脚本的 `BASE/D:/0科研/...` 前缀或将目录摆成相同结构。
- 分数缓存与重采样缓存（`_score_cache/*.npz`、`_resample_cache/*.npz`）随正式仓库发布；本 SOM zip 携带结果层（`results/` 全部 CSV/JSON）与全部脚本。

## 1. 图→脚本映射（逐图）

| 图 | 脚本 | 说明 |
|---|---|---|
| Fig.1 (tau map) | `python src/make_figures.py` | 生成 fig1/2/4 |
| Fig.2 (size vs r) | `python src/make_figures.py` | r版, 43格/corr 0.931 |
| Fig.3 (power) | `python src/make_fig_direct.py` | 生成 fig3/5/6 (direct) |
| Fig.4 (win vs blk) | `python src/make_figures.py` | |
| Fig.5 (gap landscape) | `python src/make_fig_direct.py` | 红点=自身floor判据(11对) |
| Fig.6 (design chart) | `python src/make_fig_direct.py` | |
| Fig.7 (CR1) | `python src/make_fig789.py` | |
| Fig.8 (pair corr) | `python src/make_fig789.py` | |
| Fig.9 (var decomp) | `python src/make_fig789.py` | |
| Fig.10 (rank stab) | `python src/make_fig_rankstab.py` | |

## 2. 命令行工具

- `python src/audit.py <dataset> <scorer> [B]` — 块清单(原始+合格K)与Type I校准
- `python src/power.py <bm> <pair> [gap]` / `--all` — 直接floor与可认证性

## 3. 关键结论→产物

见主稿 Table (Claim-to-artifact traceability)；全部文件在 `results/`。

---


## 22. Friedman 全向检验 (R5)
```
cd src && python probe/friedman_omnibus.py   # 确定性, 无重采样; 输出 results/friedman_omnibus.csv
```

## 23. 合成因子网格 (R6)
```
cd src && python probe/synthetic_grid.py   # 18格 x B=5000, ~7分钟; 输出 results/synthetic_grid.csv
```

## 24. R7 桥接修正重算 (2π→2π(1−π))
```
cd src && python mixed/e4_mde.py && python mixed/e1_full.py   # 已修正桥接; 输出 e4_mde_table/e1_full_components
python make_fig56.py && python make_figures.py                 # 图3/5/6(修正) + 图1/2/4
```
## 25. 配对尺度鲁棒性 (R7)
```
cd src && python probe/pair_correlation.py   # 输出 results/pair_scale_analysis.csv
```

## 26. CRSE 对照与场景分析 (R8)
```
cd src && python probe/crse_demo.py   # 输出 results/crse_vs_block.csv + %GRR场景 + 基准表数据
```

27. 图 7-9（CRSE 柱状 / 配对尺度散点 / 方差分解双面板）：`python src/make_fig789.py`（读取 results/ 下 crse_vs_block.csv、pair_scale_analysis.csv、e1_full_components.json，全部现算）。

28. 深推导复现物（R11）：`python src/probe/pr_bridge_mc.py`（AP桥不存在性MC，400副本×3单元→results/pr_bridge_mc.csv）；`python src/probe/length_effects.py`（段长比估计器效应→results/length_effects.csv，Appendix A 引用的 18.3/5.9/126–52336 出处）。

29. 第二批深推导复现物（R14）：`python src/probe/cross_bench_synthesis.py`（跨基准合成：FE 合并 MDE/RE 地板/41 套/种子噪声→cross_bench_synthesis.csv）；块集中度 75 对统计由 results/block_concentration.csv 保存（第13轮解剖段+第14轮诚实句的出处）。

30. R15b 直接尺度复现：`python src/probe/r15b_compute.py`（8 数据集 V_eval direct + AP + 75 对统计→e1_direct_components.json/r15b_summary.txt）；`python src/probe/bridge_replica_check.py`（桥 replica 失真量化）；`python src/make_fig_direct.py`（fig3/5/6 direct）；make_fig789.py 现读 e1_direct_components.json（fig9）。

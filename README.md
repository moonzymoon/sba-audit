# Segment-Block Audit (SBA)

Code, caches, and results for
**"A Variance-Component and Power Audit for Benchmark Comparisons in
Multivariate Time-Series Anomaly Detection"**
(Zhang, Li, and Shuai; submitted to *Communications in Statistics -
Simulation and Computation*).

The paper audits the statistical capability of paired benchmark comparisons
in multivariate time-series anomaly detection (MTSAD): window-level tests
reject true nulls at high rates on clustered benchmarks, an exact
event-segment-level reference test restores calibration, and directly
resampled minimum detectable effects show most published-scale gaps are
uncertified at the planned power.

## Layout

| Path | Content |
|---|---|
| `src/` | All analysis code, incl. the two CLI entry points `src/audit.py` (block inventory + Type I certificate) and `src/power.py` (direct floor + resolvability) |
| `results/` | Every CSV/JSON behind the paper's tables and figures (see `REPRODUCE.md` and the paper's claim-to-artifact table) |
| `caches/scores/` | Cached detector scores and labels per dataset-detector cell (279 `.npz`) |
| `caches/resample/` | Segment-block resampling distributions (117 `.npz`, the 114 Type I cells) |
| `prereg/revision_log.txt` | Pre-registration hash chain and the full revision log (R1-R59) |
| `REPRODUCE.md` | Environment, per-figure script map, and reproduction commands |
| `requirements.txt` | Pinned Python environment |

## Quick start

```bash
pip install -r requirements.txt
python src/power.py SMD cmhmil_seed7-pca 19.0   # replica SD, direct floor, rho
python src/audit.py SMD pca 200                 # block inventory + Type I sizes
python src/power.py --all                       # all 75 audited pairs
```

Scripts use the authors' absolute paths (Windows); adjust the `BASE`/`D:/...`
prefixes or mirror the directory layout (see `REPRODUCE.md`).

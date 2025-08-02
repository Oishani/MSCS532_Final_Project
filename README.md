# **Quantifying Cache Locality Effects: A Python Prototype Using Memory Layouts**

## Repository Layout (relevant parts)
```graphql
cache_locality_project/
├── README.md
├── requirements.txt
├── src/
│ ├── benchmark.py
│ ├── plot_results.py
├── plots_and_tables/
│ ├── benchmark_results.csv # raw benchmark output
│ ├── benchmark5000_summary.png # summary bar chart
│ ├── benchmark5000_speedup.png # speedup chart
│ ├── benchmark5000_summary_table.csv # aggregated stats
│ ├── benchmark5000_alignment_summary.csv
│ ├── benchmark5000_speedup_table.csv
├── tests/
│ ├── test_correctness.py # correctness harness
│ └── test_correctness_log_*.txt # last run log(s)
```

## Requirements
```bash
pip install -r requirements.txt
```

## Benchmarking
Run the benchmark (example for large matrix):
```bash
python3 src/benchmark.py --n 5000 --trials 6 --output plots_and_tables/benchmark_results.csv
```
- Produces timing data for different layouts/access patterns, including numba-accelerated and vectorized variants.
- Ensure you warm up numba before interpreting its steady-state timings (see next section).

### Warm-up (to avoid JIT compile skew)
```bash
python - <<'PY'
import numpy as np
from src.benchmark import sum_rows_numpy_manual_numba, sum_columns_numpy_manual_numba
base = np.random.random((5000, 5000))
arr_c = np.ascontiguousarray(base)
arr_f = np.asfortranarray(base)
_ = sum_rows_numpy_manual_numba(arr_c)
_ = sum_columns_numpy_manual_numba(arr_c)
_ = sum_rows_numpy_manual_numba(arr_f)
_ = sum_columns_numpy_manual_numba(arr_f)
PY
```

## Plotting / Analysis
Use the enhanced plotting script to aggregate and visualize results:
```bash
python src/plot_results.py \
  --input plots_and_tables/benchmark_results.csv \
  --out plots_and_tables/benchmark5000 \
  --drop-numba-warmup \
  --baseline-method python_list \
  --baseline-layout row-major \
  --baseline-access column-wise \
  --baseline-alignment misaligned
```

### Outputs:
- *_summary_table.csv, *_alignment_summary.csv, *_speedup_table.csv
- *_summary.png and *_speedup.png

## Correctness Verification
```bash
python tests/test_correctness.py --n 500
```
Produces a log file in `tests/` (e.g., test_correctness_log_n500_*.txt) summarizing all sum variants and pass/fail status.

## Interpretation Guidance
- Baseline: `python_list` column-wise is worst-case (misaligned); row-wise is better by locality.
- Numba variants: Show layout-access alignment effects once warmed up (aligned vs. misaligned).
- Vectorized (`np.sum`) demonstrates best-case optimized performance.
- Speedups: Computed relative to the worst-case baseline to quantify benefits of alignment and optimization.

## Notes
- If `numba` is not installed, the JIT paths fallback to pure Python; you will see warnings and slower manual loop performance.
- The plotting script auto-labels aligned vs misaligned cases and allows flexible baseline selection.




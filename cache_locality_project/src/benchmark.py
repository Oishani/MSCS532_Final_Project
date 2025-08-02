import time
import random
import csv
import argparse
import numpy as np
import warnings

# Try to import numba; if missing, fall back gracefully.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("numba not installed; JIT-accelerated variants will fall back to pure Python.", UserWarning)


def generate_list_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]


def sum_rows_list(matrix):
    s = 0.0
    for row in matrix:
        for val in row:
            s += val
    return s


def sum_columns_list(matrix):
    n = len(matrix)
    s = 0.0
    for j in range(n):
        for i in range(n):
            s += matrix[i][j]
    return s


def sum_rows_numpy_manual_py(arr):
    n, m = arr.shape
    s = 0.0
    for i in range(n):
        for j in range(m):
            s += arr[i, j]
    return s


def sum_columns_numpy_manual_py(arr):
    n, m = arr.shape
    s = 0.0
    for j in range(m):
            for i in range(n):
                s += arr[i, j]
    return s


if NUMBA_AVAILABLE:
    @njit
    def sum_rows_numpy_manual_numba(arr):
        n, m = arr.shape
        s = 0.0
        for i in range(n):
            for j in range(m):
                s += arr[i, j]
        return s

    @njit
    def sum_columns_numpy_manual_numba(arr):
        n, m = arr.shape
        s = 0.0
        for j in range(m):
            for i in range(n):
                s += arr[i, j]
        return s
else:
    # Fallback to Python versions, but mark method differently upstream.
    def sum_rows_numpy_manual_numba(arr):
        return sum_rows_numpy_manual_py(arr)

    def sum_columns_numpy_manual_numba(arr):
        return sum_columns_numpy_manual_py(arr)


def benchmark(n, trials, out_csv):
    fieldnames = ['method', 'layout', 'access_pattern', 'trial', 'time_sec', 'n']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Baseline: Python list-of-lists
        for t in range(1, trials + 1):
            matrix = generate_list_matrix(n)
            start = time.perf_counter()
            sum_rows = sum_rows_list(matrix)
            end = time.perf_counter()
            writer.writerow({
                'method': 'python_list',
                'layout': 'row-major',
                'access_pattern': 'row-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })
            start = time.perf_counter()
            sum_cols = sum_columns_list(matrix)
            end = time.perf_counter()
            writer.writerow({
                'method': 'python_list',
                'layout': 'row-major',
                'access_pattern': 'column-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })
            if abs(sum_rows - sum_cols) > 1e-6:
                print(f'[WARNING] Python list sums differ: rows={sum_rows} cols={sum_cols}')

        # NumPy arrays: use shared base so comparisons isolate layout/access pattern.
        base = np.random.random((n, n))
        arr_c = np.ascontiguousarray(base, dtype=float)
        arr_f = np.asfortranarray(base, dtype=float)

        for t in range(1, trials + 1):
            # 1. Pure Python manual loops over NumPy arrays (slow, shows interpreter noise)
            # C-contiguous
            start = time.perf_counter()
            s1 = sum_rows_numpy_manual_py(arr_c)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_py',
                'layout': 'C',
                'access_pattern': 'row-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            start = time.perf_counter()
            s2 = sum_columns_numpy_manual_py(arr_c)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_py',
                'layout': 'C',
                'access_pattern': 'column-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            # Fortran-contiguous
            start = time.perf_counter()
            s3 = sum_rows_numpy_manual_py(arr_f)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_py',
                'layout': 'F',
                'access_pattern': 'row-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            start = time.perf_counter()
            s4 = sum_columns_numpy_manual_py(arr_f)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_py',
                'layout': 'F',
                'access_pattern': 'column-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            # 2. numba-accelerated manual loops (if available)
            start = time.perf_counter()
            sn1 = sum_rows_numpy_manual_numba(arr_c)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_numba',
                'layout': 'C',
                'access_pattern': 'row-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            start = time.perf_counter()
            sn2 = sum_columns_numpy_manual_numba(arr_c)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_numba',
                'layout': 'C',
                'access_pattern': 'column-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            start = time.perf_counter()
            sn3 = sum_rows_numpy_manual_numba(arr_f)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_numba',
                'layout': 'F',
                'access_pattern': 'row-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            start = time.perf_counter()
            sn4 = sum_columns_numpy_manual_numba(arr_f)
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_manual_numba',
                'layout': 'F',
                'access_pattern': 'column-wise',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            # 3. Fully vectorized NumPy sum (best-case aggregated)
            # C-contiguous
            start = time.perf_counter()
            vs1 = arr_c.sum()
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_vectorized',
                'layout': 'C',
                'access_pattern': 'full-sum',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })
            # Fortran-contiguous
            start = time.perf_counter()
            vs2 = arr_f.sum()
            end = time.perf_counter()
            writer.writerow({
                'method': 'numpy_vectorized',
                'layout': 'F',
                'access_pattern': 'full-sum',
                'trial': t,
                'time_sec': end - start,
                'n': n
            })

            # Sanity check: all sums should be approximately equal
            reference = s1  # python manual row-wise on C
            candidates = [s2, s3, s4, sn1, sn2, sn3, sn4, vs1, vs2]
            for val in candidates:
                if abs(reference - val) > 1e-6:
                    print("[WARNING] Sum mismatch detected across variants.")

    print(f"Benchmark complete. Results written to {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark cache locality effects with enhanced variants.')
    parser.add_argument('--n', type=int, default=2000, help='Matrix dimension (square)')
    parser.add_argument('--trials', type=int, default=3, help='Repetitions per configuration')
    parser.add_argument('--output', type=str, default='results.csv', help='CSV output path')
    args = parser.parse_args()
    random.seed(0)  # deterministic for reproducibility of baseline list-of-lists
    benchmark(args.n, args.trials, args.output)

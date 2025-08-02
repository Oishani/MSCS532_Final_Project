import sys
import pathlib
import random
import datetime

# ensure src is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from benchmark import (
    sum_rows_list,
    sum_columns_list,
    sum_rows_numpy_manual_py,
    sum_columns_numpy_manual_py,
    sum_rows_numpy_manual_numba,
    sum_columns_numpy_manual_numba,
)

def generate_list_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]

def approx_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol or abs(a - b) <= tol * max(abs(a), abs(b), 1.0)

def verify(n=100):
    # seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    matrix = generate_list_matrix(n)

    # Python list-of-lists sums
    row_list = sum_rows_list(matrix)
    col_list = sum_columns_list(matrix)

    # Shared base numpy arrays
    arr_c = np.ascontiguousarray(np.array(matrix, dtype=float))
    arr_f = np.asfortranarray(np.array(matrix, dtype=float))

    # Pure-Python manual NumPy loops
    py_c_row = sum_rows_numpy_manual_py(arr_c)
    py_c_col = sum_columns_numpy_manual_py(arr_c)
    py_f_row = sum_rows_numpy_manual_py(arr_f)
    py_f_col = sum_columns_numpy_manual_py(arr_f)

    # Warm up / JIT compile numba variants
    _ = sum_rows_numpy_manual_numba(arr_c)
    _ = sum_columns_numpy_manual_numba(arr_c)
    _ = sum_rows_numpy_manual_numba(arr_f)
    _ = sum_columns_numpy_manual_numba(arr_f)

    numba_c_row = sum_rows_numpy_manual_numba(arr_c)
    numba_c_col = sum_columns_numpy_manual_numba(arr_c)
    numba_f_row = sum_rows_numpy_manual_numba(arr_f)
    numba_f_col = sum_columns_numpy_manual_numba(arr_f)

    # Vectorized sums
    vec_c = arr_c.sum()
    vec_f = arr_f.sum()

    # Collect values
    reference = row_list
    results = {
        'list_row': row_list,
        'list_col': col_list,
        'numpy_py_C_row': py_c_row,
        'numpy_py_C_col': py_c_col,
        'numpy_py_F_row': py_f_row,
        'numpy_py_F_col': py_f_col,
        'numpy_numba_C_row': numba_c_row,
        'numpy_numba_C_col': numba_c_col,
        'numpy_numba_F_row': numba_f_row,
        'numpy_numba_F_col': numba_f_col,
        'vectorized_C': vec_c,
        'vectorized_F': vec_f,
    }

    mismatches = []
    for name, val in results.items():
        if not approx_equal(reference, val):
            mismatches.append((name, reference, val))

    success = len(mismatches) == 0
    return success, reference, results, mismatches

def format_log(success, reference, results, mismatches, n):
    now = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    lines = []
    lines.append(f"Test Correctness Log - {now}")
    lines.append(f"Matrix size: {n}x{n}")
    lines.append("")
    lines.append("Reference (list row-wise) sum:")
    lines.append(f"  {reference:.12f}")
    lines.append("")
    lines.append("All computed sums:")
    for name, val in results.items():
        diff = val - reference
        lines.append(f"  {name}: {val:.12f} (diff {diff:+.2e})")
    lines.append("")
    if success:
        lines.append("RESULT: PASS - all sums match within tolerance.")
    else:
        lines.append("RESULT: FAIL - mismatches detected:")
        for name, base, got in mismatches:
            lines.append(f"  {name}: {got:.12f} (expected ~{base:.12f})")
    lines.append("")
    return "\n".join(lines)

def write_log(content, n):
    script_dir = pathlib.Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_correctness_log_n{n}_{timestamp}.txt"
    path = script_dir / filename
    with open(path, "w") as f:
        f.write(content)
    return path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Verify correctness of summation variants and log output.")
    parser.add_argument('--n', type=int, default=100, help='Matrix dimension to test (square)')
    args = parser.parse_args()

    success, reference, results, mismatches = verify(n=args.n)
    log_content = format_log(success, reference, results, mismatches, args.n)
    log_path = write_log(log_content, args.n)

    print(log_content)
    print(f"Log written to: {log_path}")

    if not success:
        sys.exit(1)

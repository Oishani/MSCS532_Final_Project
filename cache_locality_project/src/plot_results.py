import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

def determine_alignment(row):
    if row['method'] == 'python_list':
        return 'aligned' if row['access_pattern'] == 'row-wise' else 'misaligned'
    if row['method'] in ('numpy_manual_py', 'numpy_manual_numba'):
        if (row['layout'] == 'C' and row['access_pattern'] == 'row-wise') or \
           (row['layout'] == 'F' and row['access_pattern'] == 'column-wise'):
            return 'aligned'
        else:
            return 'misaligned'
    if row['method'] == 'numpy_vectorized':
        return 'vectorized'
    return 'unknown'

def summarize(df):
    grouped = df.groupby(['method', 'layout', 'access_pattern', 'alignment'])['time_sec'] \
                .agg(['mean', 'std', 'count']).reset_index()
    return grouped

def plot_bar_summary(summary_df, out_prefix):
    labels = []
    means = []
    errors = []
    for _, row in summary_df.iterrows():
        label = f"{row['method']}|{row['layout']}|{row['access_pattern']}|{row['alignment']}"
        labels.append(label)
        means.append(row['mean'])
        errors.append(row['std'] if not pd.isna(row['std']) else 0)
    x = range(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=errors, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Benchmark: Average Time by Method/Layout/Access Pattern/Alignment')
    plt.tight_layout()
    fig_path = f"{out_prefix}_summary.png"
    fig.savefig(fig_path)
    print(f"Saved summary bar chart to {fig_path}")

def compute_speedups(summary_df, baseline_key):
    df = summary_df.copy()
    mask = (
        (df['method'] == baseline_key[0]) &
        (df['layout'] == baseline_key[1]) &
        (df['access_pattern'] == baseline_key[2]) &
        (df['alignment'] == baseline_key[3])
    )
    if not mask.any():
        # fallback: choose the slowest mean (worst-case) and warn
        print(f"Warning: Baseline {baseline_key} not found in summary; falling back to the slowest combination.", file=sys.stderr)
        worst = df.loc[df['mean'].idxmax()]
        baseline_key = (worst['method'], worst['layout'], worst['access_pattern'], worst['alignment'])
        base = worst['mean']
        print(f"Using fallback baseline: {baseline_key} with mean {base:.6f}", file=sys.stderr)
    else:
        base = df[mask]['mean'].iloc[0]
    df['speedup_vs_baseline'] = base / df['mean']
    return df, baseline_key

def plot_speedups(speedups_df, out_prefix, baseline_key):
    fig, ax = plt.subplots()
    speed_labels = [f"{r['method']}|{r['layout']}|{r['access_pattern']}|{r['alignment']}" for _, r in speedups_df.iterrows()]
    ax.bar(range(len(speed_labels)), speedups_df['speedup_vs_baseline'])
    ax.set_xticks(range(len(speed_labels)))
    ax.set_xticklabels(speed_labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Speedup vs baseline')
    ax.set_title(f"Speedup vs baseline: {baseline_key[0]}|{baseline_key[1]}|{baseline_key[2]}|{baseline_key[3]}")
    plt.tight_layout()
    speedup_fig_path = f"{out_prefix}_speedup.png"
    fig.savefig(speedup_fig_path)
    print(f"Saved speedup chart to {speedup_fig_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot benchmark results with locality-aware annotations.')
    parser.add_argument('--input', type=str, default='results.csv', help='CSV from benchmark')
    parser.add_argument('--out', type=str, default='plots/benchmark', help='Output prefix for plots/files')
    parser.add_argument('--baseline-method', type=str, default='python_list', help='Baseline method name')
    parser.add_argument('--baseline-layout', type=str, default='row-major', help='Baseline layout')
    parser.add_argument('--baseline-access', type=str, default='column-wise', help='Baseline access pattern')
    parser.add_argument('--baseline-alignment', type=str, choices=['aligned', 'misaligned', 'vectorized'], default='misaligned', help='Baseline alignment')
    parser.add_argument('--drop-numba-warmup', action='store_true',
                        help='Drop trial 1 of numpy_manual_numba variants (recommended to exclude compilation overhead)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.drop_numba_warmup:
        before = len(df)
        df = df[~((df.method == 'numpy_manual_numba') & (df.trial == 1))]
        after = len(df)
        print(f"Dropped {before - after} warm-up rows for numpy_manual_numba (trial 1 excluded).")

    df['alignment'] = df.apply(determine_alignment, axis=1)

    summary_df = summarize(df)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    summary_table_path = f"{args.out}_summary_table.csv"
    summary_df.to_csv(summary_table_path, index=False)
    alignment_summary = summary_df.groupby(['alignment', 'method'])['mean'].mean().reset_index()
    alignment_summary_path = f"{args.out}_alignment_summary.csv"
    alignment_summary.to_csv(alignment_summary_path, index=False)

    plot_bar_summary(summary_df, args.out)

    baseline_key = (args.baseline_method, args.baseline_layout, args.baseline_access, args.baseline_alignment)
    speedups, used_baseline = compute_speedups(summary_df, baseline_key=baseline_key)
    speedups_path = f"{args.out}_speedup_table.csv"
    speedups.to_csv(speedups_path, index=False)
    plot_speedups(speedups, args.out, used_baseline)

    print(f"Saved summary table to {summary_table_path}")
    print(f"Saved alignment summary to {alignment_summary_path}")
    print(f"Saved speedup table to {speedups_path}")

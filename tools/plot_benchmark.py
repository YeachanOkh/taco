#!/usr/bin/env python3
"""Plot benchmark CSV output from format_conversion_benchmark.

Usage: python3 tools/plot_benchmark.py results.csv -o out.png
Accepts one CSV with header: matrix,format,mean_ms,stdev_ms,median_ms,samples,vmrss_kb,storage_bytes,nnz
Optional: pass --baseline baseline.csv where baseline has columns matrix,format,mean_ms and will be plotted alongside.
"""
import sys
import csv
import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def group_by_matrix(rows):
    g = collections.OrderedDict()
    for r in rows:
        m = r['matrix']
        if m not in g:
            g[m] = []
        g[m].append(r)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--baseline', help='optional baseline CSV')
    p.add_argument('-o','--out', default='benchmark.png')
    args = p.parse_args()

    rows = read_csv(args.csv)
    baseline = None
    if args.baseline:
        baseline = read_csv(args.baseline)

    grouped = group_by_matrix(rows)
    nmat = len(grouped)
    fig, axes = plt.subplots(nmat, 1, figsize=(6, 3*nmat))
    if nmat == 1:
        axes = [axes]

    for ax, (matrix, entries) in zip(axes, grouped.items()):
        # sort by format name
        entries_sorted = sorted(entries, key=lambda e: e['format'])
        formats = [e['format'] for e in entries_sorted]
        means = [float(e.get('mean_ms', e.get('mean', 0))) for e in entries_sorted]
        errs = [float(e.get('stdev_ms', 0)) for e in entries_sorted]
        x = np.arange(len(formats))
        width = 0.35
        ax.bar(x - width/2, means, width, yerr=errs, label='taco')

        if baseline:
            # find baseline entries for this matrix
            bmap = { (b['matrix'], b['format']): float(b['mean_ms']) for b in baseline }
            bmeans = [bmap.get((matrix, f), np.nan) for f in formats]
            ax.bar(x + width/2, bmeans, width, label='baseline')

        ax.set_xticks(x)
        ax.set_xticklabels(formats)
        ax.set_ylabel('mean (ms)')
        ax.set_title(matrix)
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.out)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()

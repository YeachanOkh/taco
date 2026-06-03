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
    p.add_argument('--y', default='mean_ms', help='CSV column to plot (default: mean_ms)')
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

    ycol = args.y

    for ax, (matrix, entries) in zip(axes, grouped.items()):
        # sort by format name
        entries_sorted = sorted(entries, key=lambda e: e['format'])
        formats = [e['format'] for e in entries_sorted]
        # read numeric values from chosen column; missing -> nan
        def tofloat(v):
            try:
                return float(v)
            except Exception:
                return float('nan')

        means = [tofloat(e.get(ycol, e.get('mean_ms', e.get('mean', 'nan')))) for e in entries_sorted]
        x = np.arange(len(formats))
        width = 0.35
        # Plot taco bars (no edge lines and no error bars so no vertical line appears)
        ax.bar(x - width/2, means, width, label='taco', edgecolor='none', color='C0')

        if baseline:
            # find baseline entries for this matrix using the same y column when available
            bmap = {}
            for b in baseline:
                key = (b.get('matrix'), b.get('format'))
                bmap[key] = tofloat(b.get(ycol, b.get('mean_ms', b.get('mean', 'nan'))))
            bmeans = [bmap.get((matrix, f), float('nan')) for f in formats]
            # Convert NaNs to masked values so matplotlib won't draw bars or lines
            bmeans_masked = [np.nan if np.isnan(v) else v for v in bmeans]
            ax.bar(x + width/2, bmeans_masked, width, label='baseline', edgecolor='none', color='C1')

        ax.set_xticks(x)
        ax.set_xticklabels(formats)
        # Y axis label based on chosen column
        if ycol == 'mean_ms':
            ylabel = 'mean (ms)'
        elif ycol == 'storage_bytes':
            ylabel = 'storage (B)'
        else:
            ylabel = ycol
        ax.set_ylabel(ylabel)
        ax.set_title(matrix)
        # Improve aesthetics: remove top/right spines and add subtle grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.out)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()

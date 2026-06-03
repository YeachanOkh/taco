#!/usr/bin/env python3
"""Generate a baseline CSV by timing SciPy sparse format conversions.

Usage: python3 tools/sparskit_baseline.py [--repeat N] [--formats f1,f2,...]
    file1.mtx [file2.mtx ...]
Produces CSV with header: matrix,format,mean_ms,storage_bytes
"""
import sys
import argparse
import time
import csv

try:
    import numpy as np
    import scipy.io
    import scipy.sparse as sp
except Exception as e:
    sys.stderr.write("Missing dependency: %s\n" % e)
    sys.stderr.write("Install with: pip3 install --user numpy scipy\n")
    sys.exit(2)


def time_conversion(mat, fmt, repeat):
    # mat: scipy.sparse matrix
    times = []
    if fmt in ('csr','csc','coo'):
        for _ in range(repeat):
            t0 = time.perf_counter()
            if fmt == 'csr':
                _ = mat.tocsr()
            elif fmt == 'csc':
                _ = mat.tocsc()
            else:
                _ = mat.tocoo()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return float(np.mean(times))

    if fmt == 'ell':
        # perform COO -> ELL manual conversion timing
        coo = mat.tocoo()
        nrows = mat.shape[0]
        rows = [[] for _ in range(nrows)]
        for r, c, v in zip(coo.row, coo.col, coo.data):
            rows[r].append((int(c), float(v)))
        maxPerRow = max((len(rw) for rw in rows), default=0)

        for _ in range(repeat):
            t0 = time.perf_counter()
            M = maxPerRow * nrows
            idx = np.full(M, -1, dtype=np.int32)
            vals = np.zeros(M, dtype=np.float64)
            for i, rvec in enumerate(rows):
                for t, (c, val) in enumerate(rvec):
                    pos = t * nrows + i
                    idx[pos] = c
                    vals[pos] = val
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return float(np.mean(times))

    raise ValueError('Unsupported format: %s' % fmt)


def compute_storage_bytes(mat, fmt):
    """Estimate storage bytes for the given SciPy matrix in the target format.
    """
    if fmt == 'csr':
        M = mat.tocsr()
        return int(M.data.nbytes + M.indices.nbytes + M.indptr.nbytes)
    if fmt == 'csc':
        M = mat.tocsc()
        return int(M.data.nbytes + M.indices.nbytes + M.indptr.nbytes)
    if fmt == 'coo':
        M = mat.tocoo()
        return int(M.data.nbytes + M.row.nbytes + M.col.nbytes)
    if fmt == 'ell':
        coo = mat.tocoo()
        nrows = int(mat.shape[0])
        rows = [[] for _ in range(nrows)]
        for r, c, v in zip(coo.row, coo.col, coo.data):
            rows[int(r)].append((int(c), float(v)))
        maxPerRow = max((len(rw) for rw in rows), default=0)
        M = maxPerRow * nrows
        return int(M * (np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--formats', default='csc,ell')
    p.add_argument('-o', '--out', default='sparskit_baseline.csv')
    p.add_argument('files', nargs='+')
    args = p.parse_args()

    formats = [f.strip() for f in args.formats.split(',') if f.strip()]

    rows = []
    for f in args.files:
        try:
            mat = scipy.io.mmread(f)
        except Exception as e:
            sys.stderr.write('Failed to read %s: %s\n' % (f, e))
            continue
        # ensure sparse
        if not sp.isspmatrix(mat):
            mat = sp.coo_matrix(mat)
        for fmt in formats:
            try:
                mean_ms = time_conversion(mat, fmt, args.repeat)
                storage_bytes = compute_storage_bytes(mat, fmt)
            except Exception as e:
                sys.stderr.write('Conversion failed for %s -> %s: %s\n' % (f, fmt, e))
                continue
            rows.append({'matrix': f, 'format': fmt, 'mean_ms': '%.6f' % mean_ms, 'storage_bytes': str(int(storage_bytes))})
            sys.stdout.write('Measured %s %s: %.6f ms, storage: %d B\n' % (f, fmt, mean_ms, storage_bytes))

    # write CSV
    with open(args.out, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['matrix', 'format', 'mean_ms', 'storage_bytes'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    sys.stdout.write('Wrote baseline CSV to %s\n' % args.out)


if __name__ == '__main__':
    main()

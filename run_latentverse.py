"""
Latentverse analysis of the latent space.

Usage:
  run_latentverse.py -h | --help
  run_latentverse.py <data_fname> <labels_fname>

Options:
  -h        Show this screen.
  --help    Show this screen.
"""

from __future__ import annotations
from docopt import docopt
import sys
import time
from typing import Any

import numpy as np
from latentverse import (
    run_clustering,
    run_disentanglement,
    run_expressiveness,
    run_probing,
    run_robustness,
)


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------
def fmt_metric(v: Any) -> str:
    if v is None:
        return '—'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k}={fmt_metric(x)}' for k, x in v.items()) + '}'
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v).ravel()
        if arr.dtype.kind in 'fiu':  # only summarise numeric arrays
            return f'[{arr.shape[0]}d, mean={float(arr.mean()):.4f}]'
        # non-numeric (e.g. model-complexity labels): show length + first 3 entries
        head = ', '.join(repr(x) for x in arr[:3].tolist())
        if arr.size > 3:
            head += ', ...'
        return f'[{arr.size}: {head}]'
    try:
        return f'{float(v):.4f}'
    except (TypeError, ValueError):
        return repr(v)


def section(title: str) -> None:
    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def demo_clustering(X, y) -> None:
    section('1. Clusterability')
    out = run_clustering(X, y, random_state=42)
    print('  Metrics:')
    for name, value in out['results'].items():
        print(f'    {name:35s} {fmt_metric(value)}')
    print(f'  PCA available     : {out["pca_available"]}')
    print(f'  KMeans assignments: shape={out["cluster_labels"].shape}')


def demo_disentanglement(X, y) -> None:
    section('2. Disentanglement')
    out = run_disentanglement(X, y, random_state=42)
    metrics = out['metrics']
    print(f'    DCI            : {fmt_metric(metrics.get("DCI"))}')
    print(f'    MIG            : {fmt_metric(metrics.get("MIG"))}')
    print(f'    SAP            : {fmt_metric(metrics.get("SAP"))}')
    print(f'    TC             : {fmt_metric(metrics.get("TC"))}')


def demo_expressiveness(X, y) -> None:
    section('3. Expressiveness')
    out = run_expressiveness(
        X,
        y,
        percent_to_remove_list=[0, 10, 30, 60],
        plots=False,
        random_state=42,
    )
    print(f'    metrics keys: {list(out["metrics"].keys())[:6]}')


def demo_probing(X, y) -> None:
    section('4. Probing')
    out = run_probing(X, y, random_state=42)
    if isinstance(out, tuple):
        metrics, _ = out
    else:
        metrics = out
    print(f'    metrics: {fmt_metric(metrics)}')


def demo_robustness(X, y) -> None:
    section('5. Robustness')
    out = run_robustness(
        X,
        y,
        noise_levels=[0.0, 0.25, 0.5, 1.0],
        metric='clustering',
        plots=False,
        random_state=42,
    )
    print(f'    metric keys: {list(out["metrics"].keys())[:6]}')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(data, labels) -> int:
    t0 = time.time()
    print('Latentverse analysis of the latent space')
    print('-' * 60)

    demos = [
        ('clusterability', demo_clustering),
        ('disentanglement', demo_disentanglement),
        ('expressiveness', demo_expressiveness),
        ('probing', demo_probing),
        ('robustness', demo_robustness),
    ]

    failures: list[tuple[str, Exception]] = []
    for name, fn in demos:
        try:
            fn(data, labels)
        except Exception as e:  # pragma: no cover — surfaces in CLI output
            failures.append((name, e))
            print(f'\n  FAILED {name!r}: {e!r}')

    print()
    print('=' * 60)
    print(f'Done in {time.time() - t0:.1f}s')
    if failures:
        print(f'{len(failures)} failure(s):')
        for name, e in failures:
            print(f'  - {name}: {e}')
        return 1
    print('All entrypoints executed successfully.')
    return 0


if __name__ == '__main__':
    args = docopt(__doc__)
    data_fname = args['<data_fname>']
    labels_fname = args['<labels_fname>']
    data = np.load(data_fname)
    labels = np.load(labels_fname)
    sys.exit(main(data, labels))

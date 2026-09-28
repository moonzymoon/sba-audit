# -*- coding: utf-8 -*-
"""SBA audit command: per-cell block inventory and Type I certificate.

Usage:  python audit.py <dataset> <scorer> [B]
Example: python audit.py SMD pca 10000
Prints the raw and qualified block counts (K) and the window/block empirical
sizes (t, Wilcoxon, exact sign-permutation reference).  Thin wrapper over
bootstrap/e3_clean.run() and common.blocks.build_blocks().
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap.e3_clean import run, load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ds, sc = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    y = load_goodness(sc, ds)[1]
    blocks = build_blocks(events_from_binary(y), len(y))
    k_qual = sum(1 for k, s, e in blocks if k == 'event' and e - s >= 10) + \
             sum(1 for k, s, e in blocks if k == 'normal')
    print('cell:', ds, sc, 'B=', B)
    print('  raw blocks          %d' % len(blocks))
    print('  qualified K         %d' % k_qual)
    r = run(ds, sc, B)
    for k, v in r.items():
        if k.startswith('size_'):
            print('  %-22s %s' % (k, round(v, 4) if isinstance(v, float) else v))

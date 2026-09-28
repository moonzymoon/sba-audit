# -*- coding: utf-8 -*-
"""SBA power command: direct floor and resolvability for a claimed gap.

Usage:  python power.py <benchmark> <pair> [gap_pts]
        python power.py --all
Example: python power.py SMD cmhmil_seed7-pca 19.0
Reads results/direct_scale_verdict.csv (shared-block direct pricing, B=400 per
pair), prints the pair's replica SD, direct floor (2.8 x SD, power .8), and
rho = gap / floor with the certification verdict.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', '02shiyanjilu'.encode('ascii', 'ignore').decode('ascii'))

def find_res():
    cand = [os.path.join(HERE, '..', '02_shiyanjilu', 'results', 'direct_scale_verdict.csv'),
            os.path.join(HERE, 'results', 'direct_scale_verdict.csv'),
            'direct_scale_verdict.csv']
    for c in cand:
        if os.path.exists(c):
            return c
    sys.exit('direct_scale_verdict.csv not found; run from repo root')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = list(csv.DictReader(open(find_res(), encoding='utf-8-sig')))
    if sys.argv[1] == '--all':
        for r in rows:
            g, m = float(r['gap_pts']), float(r['mde_direct_pts'])
            print('%-5s %-24s gap=%5.1f floor=%5.1f rho=%.2f %s'
                  % (r['bm'], r['pair'], g, m, g / m,
                     'CERTIFIABLE' if g >= m else 'uncertified'))
        sys.exit(0)
    bm, pair = sys.argv[1], sys.argv[2]
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else None
    for r in rows:
        if r['bm'] == bm and r['pair'] == pair:
            sd, m = float(r['sd_direct_pts']), float(r['mde_direct_pts'])
            g = gap if gap is not None else float(r['gap_pts'])
            print('%s %s: replica SD=%.2f pts' % (bm, pair, sd))
            print('  direct floor (power .8): %.1f pts' % m)
            print('  claimed gap %.1f -> rho=%.2f -> %s'
                  % (g, g / m, 'CERTIFIABLE' if g >= m else 'uncertified at planned power'))
            break
    else:
        sys.exit('pair not found; run power.py --all to list')

#!/usr/bin/env python3
"""Plot UFM cache comparisons; all ratios are candidate / reference.

Usage: MPLCONFIGDIR=/tmp/jet-mpl python3 plot_ufm_jet_cache.py RESULT_DIRECTORY
Points summarize only matched, successful runs. Missing pairs remain visible in
plot-data.json. Small dots are per-graph geometric means across seeds; large
markers are geometric means across all matched runs. Whiskers are 95% graph
bootstrap intervals, preserving all seeds within each sampled graph.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

COMPARISONS = (('compact', 'stock'), ('jet', 'compact'), ('jet', 'stock'))
LABELS = ('Corrected compact\n/ stock', 'Jet\n/ corrected compact', 'Jet\n/ stock')
METRICS = (
    ('runtime_s', 'Partitioning runtime'),
    ('cut', 'Edge cut'),
    ('max_rss_kib', 'Peak process memory'),
    ('ufm_time_s', 'Time inside UFM'),
)


def geometric(values):
    if not values:
        return None
    if any(x == 0 for x in values):
        return 0.0
    return math.exp(statistics.mean(math.log(x) for x in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    source = args.root / 'runs.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    keyed = {(r['graph'], r['k'], r['seed'], r['arm']): r for r in rows}
    assert len(keyed) == len(rows), 'Duplicate measurements'
    plan_path = args.root / 'plan.json'
    planned = json.loads(plan_path.read_text())['jobs'] if plan_path.exists() else rows
    cells = sorted({(r['graph'], r['k'], r['seed']) for r in planned})
    ks = sorted({k for _, k, _ in cells})
    palette = ('#176e96', '#c35d21')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    fig.suptitle('Unconstrained FM: adaptive cache versus compact hashing', fontsize=15)
    data = {'runs_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'matplotlib_version': matplotlib.__version__, 'series': []}
    for ax, (metric, title) in zip(axes.flat, METRICS):
        ax.set_title(title, fontsize=12, loc='left', fontweight='bold')
        ax.axhline(1, color='#666666', linewidth=1, linestyle='--')
        for ki, k in enumerate(ks):
            for ai, (arm, reference) in enumerate(COMPARISONS):
                ratios = {}
                missing = []
                series = dict(metric=metric, arm=arm, reference=reference, k=k, valid_pairs=0,
                              geometric_mean=None, graph_bootstrap_95pct=None,
                              per_graph={}, missing_pairs=missing,
                              applicable=True)
                data['series'].append(series)
                if not series['applicable']:
                    continue
                for graph, cell_k, seed in cells:
                    if cell_k != k:
                        continue
                    ref = keyed.get((graph, k, seed, reference))
                    candidate = keyed.get((graph, k, seed, arm))
                    reason = None
                    if not ref or not candidate:
                        reason = 'missing measurement'
                    elif ref['status'] != 'ok' or candidate['status'] != 'ok':
                        reason = 'unsuccessful measurement'
                    elif ref.get(metric) is None or candidate.get(metric) is None:
                        reason = 'missing metric'
                    elif not all(math.isfinite(r[metric]) and r[metric] >= 0
                                 for r in (ref, candidate)):
                        reason = 'invalid metric'
                    elif ref[metric] == 0 and (metric != 'cut' or candidate[metric] != 0):
                        reason = 'undefined ratio to zero'
                    if reason:
                        missing.append(dict(graph=graph, seed=seed, reason=reason))
                        continue
                    ratio = candidate[metric] / ref[metric] if ref[metric] else 1.0
                    ratios.setdefault(graph, []).append(ratio)
                pairs = [v for values in ratios.values() for v in values]
                if not pairs:
                    continue
                center = geometric(pairs)
                rng = random.Random(42)
                graphs = sorted(ratios)
                bootstrap = sorted(geometric([x for g in rng.choices(graphs, k=len(graphs))
                                               for x in ratios[g]]) for _ in range(2000))
                lo, hi = bootstrap[50], bootstrap[1949]
                x = ai + (ki - (len(ks) - 1) / 2) * .28
                means = [geometric(ratios[g]) for g in graphs]
                jitters = [(i - (len(means) - 1)/2)*.012 for i in range(len(means))]
                ax.scatter([x + j for j in jitters], means, color=palette[ki], s=18, alpha=.32, zorder=3)
                ax.vlines(x, lo, hi, color=palette[ki], lw=1.8, zorder=4)
                ax.scatter([x], [center], color=palette[ki], marker='D', s=45, zorder=5,
                           label=f'k = {k}' if ai == 0 else None)
                ax.annotate(f'{center:.3f}', (x, center), xytext=(0, 10 if ki == 0 else -16),
                            textcoords='offset points', ha='center', fontsize=8, color=palette[ki])
                series.update(valid_pairs=len(pairs), geometric_mean=center,
                              graph_bootstrap_95pct=[lo, hi],
                              per_graph={g: geometric(ratios[g]) for g in graphs})
        if metric not in ('cut', 'max_rss_kib'):
            ax.set_yscale('log', base=2)
            ax.yaxis.set_major_locator(FixedLocator([.5, .75, 1, 1.25, 1.5, 2, 3]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.set_ylabel('Candidate / reference')
        ax.set_xticks(range(len(COMPARISONS)), LABELS, fontsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['right', 'top']].set_visible(False)
        ax.margins(y=.25, x=.15)
    axes[0, 0].legend(loc='best', frameon=False, fontsize=9)
    fig.supxlabel('Smaller ratios are better. Six graphs; three seeds; 32 physical cores.\n'
                  'Diamonds: matched-run geometric means. Whiskers: 95% graph bootstrap intervals. '
                  'Dots: graph means.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(args.root / f'comparison.{ext}', dpi=180)
    (args.root / 'plot-data.json').write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()

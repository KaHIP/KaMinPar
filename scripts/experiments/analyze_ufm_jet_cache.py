#!/usr/bin/env python3
"""Compute complete-case UFM cache ratios, graph summaries, and phase totals."""
import argparse
import hashlib,json,math,statistics
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
root=parser.parse_args().root
source=root/'runs.jsonl';rows=[json.loads(x) for x in source.read_text().splitlines()]
plan=json.loads((root/'plan.json').read_text())
assert len(rows)==len(plan['jobs']), 'Analysis requires the complete plan'
assert all(r['status']=='ok' for r in rows)
keyed={(r['graph'],r['k'],r['seed'],r['arm']):r for r in rows};assert len(keyed)==len(rows)
metrics=['runtime_s','cut','max_rss_kib','ufm_time_s']
comparisons=[('compact','stock'),('jet','compact'),('jet','stock')]
def gm(values):
 assert all(x>0 for x in values)
 return math.exp(statistics.mean(math.log(x) for x in values))
def compare(cand,ref,k=None,graph=None):
 pairs=[(r,keyed[(r['graph'],r['k'],r['seed'],ref)]) for r in rows if r['arm']==cand and (k is None or r['k']==k) and (graph is None or r['graph']==graph)]
 return dict(candidate=cand,reference=ref,k=k,graph=graph,pairs=len(pairs),ratios={m:gm([a[m]/b[m] for a,b in pairs]) for m in metrics},runtime_wins=sum(a['runtime_s']<b['runtime_s'] for a,b in pairs),cut_wins=sum(a['cut']<b['cut'] for a,b in pairs),equal_cuts=sum(a['cut']==b['cut'] for a,b in pairs))
result=dict(runs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),overall=[compare(*c) for c in comparisons],per_k=[compare(*c,k=k) for k in plan['config']['ks'] for c in comparisons],per_graph=[compare(*c,k=k,graph=g) for g in sorted({r['graph'] for r in rows}) for k in plan['config']['ks'] for c in comparisons])
result['phases']={}
for k in plan['config']['ks']:
 result['phases'][str(k)]={}
 for arm in ['stock','compact','jet']:
  rr=[r for r in rows if r['k']==k and r['arm']==arm]
  phases={p:sum(r['ufm_phases_s'].get(p,0) for r in rr) for p in sorted({p for r in rr for p in r['ufm_phases_s']})}
  result['phases'][str(k)][arm]=dict(runtime_s=sum(r['runtime_s'] for r in rr),ufm_s=sum(r['ufm_time_s'] for r in rr),phases_s=phases,min_rss_mib=min(r['max_rss_kib']/1024 for r in rr),max_rss_mib=max(r['max_rss_kib']/1024 for r in rr))
(root/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(root/'analysis.json')

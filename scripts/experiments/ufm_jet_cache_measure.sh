#!/usr/bin/env bash
#SBATCH --job-name=ufm-cache-ablation
#SBATCH --partition=dijkstra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
set -euo pipefail
UFM_CACHE_ROOT=$(realpath "$1")
UFM_CACHE_GCC=/nfs/software/spack-installs/upstream/opt/spack/linux-rocky9-x86_64/gcc-14.2.0/gcc-13.3.0-akehmu65vgbtliwqsjkqj5trscvfmfmh
export PATH="$UFM_CACHE_GCC/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$UFM_CACHE_GCC/lib64"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_PRELOAD TSAN_OPTIONS ASAN_OPTIONS UBSAN_OPTIONS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[[ -n ${SLURM_JOB_ID:-} && $(hostname -s) == dijkstra ]] || exit 2
python3 - "$UFM_CACHE_ROOT" <<'PY'
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys
root=Path(sys.argv[1])
binaries={role: root/role/'build/apps/KaMinPar' for role in ('stock','candidate')}
for role,binary in binaries.items():
    provenance=json.loads((root/role/'provenance.json').read_text())
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==provenance['binary_sha256']
smoke=root/'smoke'; smoke.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('runner', root/'ufm_jet_cache.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
# Weighted ring of dense groups, deterministic and symmetric.
n=512; edges=[{} for _ in range(n)]
for u in range(n):
    for delta in (1,3,7,31):
        v=(u+delta)%n; weight=1+(u+v)%7
        edges[u][v]=weight;edges[v][u]=weight
p=smoke/'weighted.metis'
p.write_text(f'{n} {sum(map(len,edges))//2} 1\n'+'\n'.join(' '.join(f'{v+1} {w}' for v,w in sorted(row.items())) for row in edges)+'\n')
records=[]
for k in (2,16,64):
    for arm,opts in runner.ARMS.items():
        binary=binaries['stock' if arm=='stock' else 'candidate']
        blocks=smoke/f'{arm}-k{k}.blocks'; log=smoke/f'{arm}-k{k}.log'
        command=[str(binary),'-G',str(p),'-k',str(k),'-t','8','-e','0.03','--seed','1','-v','-T','--disable-colors','--output-block-sizes',str(blocks)]+opts
        with log.open('w') as out:
            run=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=60)
        parsed=runner.parse_log(log,run.returncode)
        parsed['balance_audit']=runner.audit_blocks(blocks,n,k,.03)
        assert parsed['status']=='ok' and parsed['balance_audit']['exact_feasible'],(arm,k,parsed)
        assert parsed['ufm_time_s'] is not None and parsed['result_k']==k
        assert parsed['reported_gain_cache']==runner.CACHE_STRATEGIES[arm]
        records.append(dict(arm=arm,k=k,command=command,**parsed))
(root/'smoke/summary.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
PY
python3 -B "$UFM_CACHE_ROOT/ufm_jet_cache.py" run \
  --root "$UFM_CACHE_ROOT/results" --binary "$UFM_CACHE_ROOT/candidate/build/apps/KaMinPar" \
  --provenance "$UFM_CACHE_ROOT/candidate/provenance.json" \
  --stock-binary "$UFM_CACHE_ROOT/stock/build/apps/KaMinPar" \
  --stock-provenance "$UFM_CACHE_ROOT/stock/provenance.json" \
  --stage-dir /dev/shm/seemaier-ufm-jet-cache-20261022 \
  --host dijkstra --threads 32 --ks 16 1024 --seeds 1 2 3 --timeout 300

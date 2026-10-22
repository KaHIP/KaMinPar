#!/usr/bin/env bash
#SBATCH --job-name=jet-reduced-ufm
#SBATCH --partition=dijkstra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
set -euo pipefail
JET_ROOT=$(realpath "$1")
JET_GCC=/nfs/software/spack-installs/upstream/opt/spack/linux-rocky9-x86_64/gcc-14.2.0/gcc-13.3.0-akehmu65vgbtliwqsjkqj5trscvfmfmh
export PATH="$JET_GCC/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$JET_GCC/lib64"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_PRELOAD TSAN_OPTIONS ASAN_OPTIONS UBSAN_OPTIONS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[[ -n ${SLURM_JOB_ID:-} && $(hostname -s) == dijkstra ]] || exit 2
python3 - "$JET_ROOT" <<'PY'
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys
root=Path(sys.argv[1]); binary=root/'build/apps/KaMinPar'
provenance=json.loads((root/'provenance.json').read_text())
assert hashlib.sha256(binary.read_bytes()).hexdigest()==provenance['binary_sha256']
smoke=root/'smoke'; smoke.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('runner', root/'jet_reduced_ufm.py')
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
for k in (2,16):
    for arm,opts in runner.ARMS.items():
        blocks=smoke/f'{arm}-k{k}.blocks'; log=smoke/f'{arm}-k{k}.log'
        command=[str(binary),'-G',str(p),'-k',str(k),'-t','8','-e','0.03','--seed','1','-v','-T','--disable-colors','--output-block-sizes',str(blocks)]+opts
        with log.open('w') as out:
            run=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=60)
        parsed=runner.parse_log(log,run.returncode)
        parsed['balance_audit']=runner.audit_blocks(blocks,n,k,.03)
        assert parsed['status']=='ok' and parsed['balance_audit']['exact_feasible'],(arm,k,parsed)
        if arm != 'eco_ufm': assert parsed['jet_time_s'] is not None
        records.append(dict(arm=arm,k=k,command=command,**parsed))
(root/'smoke/summary.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
PY
python3 -B "$JET_ROOT/jet_reduced_ufm.py" run \
  --root "$JET_ROOT/results" --binary "$JET_ROOT/build/apps/KaMinPar" \
  --provenance "$JET_ROOT/provenance.json" \
  --stage-dir /dev/shm/seemaier-jet-adaptive-20261022 \
  --host dijkstra --threads 32 --ks 16 1024 --seeds 1 2 3 --timeout 300

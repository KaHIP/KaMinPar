#!/usr/bin/env bash
#SBATCH --job-name=ufm-cache-build
#SBATCH --partition=dijkstra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
set -euo pipefail
UFM_CACHE_ROOT=$(realpath "$1")
UFM_CACHE_DEPS=/nfs/work/seemaier/experiments/dkaminpar-single-host-20261021/build/_deps
UFM_CACHE_GCC=/nfs/software/spack-installs/upstream/opt/spack/linux-rocky9-x86_64/gcc-14.2.0/gcc-13.3.0-akehmu65vgbtliwqsjkqj5trscvfmfmh
UFM_CACHE_CMAKE=/nfs/software/spack-installs/upstream/opt/spack/linux-rocky9-x86_64/gcc-14.2.0/cmake-3.30.5-7odlnvs2xcai2cpf45qw2lrp4frolugx
UFM_CACHE_MAKE=/nfs/software/spack-installs/upstream/opt/spack/linux-rocky9-x86_64/gcc-14.2.0/gmake-4.4.1-exjizzwwqrospljlql3bg2x47c6qajpj
export PATH="$UFM_CACHE_CMAKE/bin:$UFM_CACHE_MAKE/bin:$UFM_CACHE_GCC/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$UFM_CACHE_GCC/lib64"
export CC="$UFM_CACHE_GCC/bin/gcc" CXX="$UFM_CACHE_GCC/bin/g++"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_PRELOAD TSAN_OPTIONS ASAN_OPTIONS UBSAN_OPTIONS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[[ -n ${SLURM_JOB_ID:-} && $(hostname -s) == dijkstra ]] || exit 2
mkdir -p "$UFM_CACHE_ROOT/build-records"
python3 - "$UFM_CACHE_ROOT" "$UFM_CACHE_DEPS" "$SLURM_JOB_ID" <<'PY'
from pathlib import Path
import hashlib, json, sys
root, deps, job = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def tree(path):
    files={str(f.relative_to(path)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.rglob('*')) if f.is_file() and '.git' not in f.relative_to(path).parts}
    return dict(files=files, sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest())
manifest={name:tree(path) for name,path in [('source',root/'source'),('tbb',deps/'tbb-src'),('kassert',deps/'kassert-src')]}
snapshot=json.loads((root/'source-manifest.json').read_text())
assert manifest['source']['files']==snapshot['files'], 'Uploaded source differs from frozen snapshot'
assert manifest['source']['sha256']==snapshot['tree_sha256'], 'Frozen source hash differs'
(root/'build-records'/('source-'+job+'.json')).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
cmake -S "$UFM_CACHE_ROOT/source" -B "$UFM_CACHE_ROOT/build" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS= -DCMAKE_EXE_LINKER_FLAGS= \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG' '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKAMINPAR_BUILD_APPS=ON -DKAMINPAR_BUILD_DISTRIBUTED=OFF \
  -DKAMINPAR_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  -DKAMINPAR_BUILD_TOOLS=OFF -DKAMINPAR_BUILD_BENCHMARKS=OFF \
  -DKAMINPAR_64BIT_IDS=OFF -DKAMINPAR_64BIT_NODE_IDS=OFF \
  -DKAMINPAR_64BIT_EDGE_IDS=ON -DKAMINPAR_64BIT_WEIGHTS=ON \
  -DKAMINPAR_BUILD_WITH_MTUNE_NATIVE=ON -DKAMINPAR_BUILD_WITH_CCACHE=OFF \
  -DKAMINPAR_BUILD_WITH_DEBUG_SYMBOLS=OFF \
  -DKAMINPAR_ENABLE_STATISTICS=OFF -DKAMINPAR_ENABLE_HEAP_PROFILING=OFF \
  -DKAMINPAR_ENABLE_PAGE_PROFILING=OFF -DKAMINPAR_ENABLE_TIMERS=ON \
  -DKAMINPAR_BUILD_WITH_ASAN=OFF -DKAMINPAR_BUILD_WITH_UBSAN=OFF -DKAMINPAR_BUILD_WITH_PG=OFF \
  -DKAMINPAR_DOWNLOAD_TBB=ON -DKAMINPAR_DOWNLOAD_KASSERT=ON \
  -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_TBB="$UFM_CACHE_DEPS/tbb-src" \
  -DFETCHCONTENT_SOURCE_DIR_KASSERT="$UFM_CACHE_DEPS/kassert-src" \
  > "$UFM_CACHE_ROOT/build-records/configure-$SLURM_JOB_ID.log" 2>&1
cmake --build "$UFM_CACHE_ROOT/build" --target KaMinParApp -j 32 > "$UFM_CACHE_ROOT/build-records/build-$SLURM_JOB_ID.log" 2>&1
python3 - "$UFM_CACHE_ROOT" "$UFM_CACHE_DEPS" "$SLURM_JOB_ID" <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys
root,deps,job=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
record=json.loads((root/'build-records'/('source-'+job+'.json')).read_text())
for name,path in [('source',root/'source'),('tbb',deps/'tbb-src'),('kassert',deps/'kassert-src')]:
    actual={str(f.relative_to(path)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.rglob('*')) if f.is_file() and '.git' not in f.relative_to(path).parts}
    assert actual == record[name]['files'], 'Source changed during build: '+name
binary=root/'build/apps/KaMinPar'
ldd=subprocess.check_output(['ldd',str(binary)],text=True)
assert all(lib not in ldd for lib in ('libtsan','libasan','libubsan'))
assert '-fsanitize' not in (root/'build/compile_commands.json').read_text()
record['build_job_id']=job
record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
record['ldd']=ldd
record['cmake_cache']=(root/'build/CMakeCache.txt').read_text()
record['compiler']=subprocess.check_output(['g++','--version'],text=True)
snapshot=json.loads((root/'source-manifest.json').read_text())
record['source_base_commit']=snapshot['baseline_commit']
record['source_runtime_commit']=snapshot['head_at_snapshot']
record['source_snapshot_tree_sha256']=snapshot['tree_sha256']
(root/'provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print('BUILT',record['binary_sha256'])
PY

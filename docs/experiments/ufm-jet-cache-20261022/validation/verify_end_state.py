#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,datetime,re
root=Path('/nfs/work/seemaier/experiments/ufm-jet-cache-20261022')
plan=json.loads((root/'results/plan.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
record=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),roles={})
for role,bundle in plan['config']['binaries'].items():
    source=root/role/'source';manifest=json.loads((root/role/'source-manifest.json').read_text())
    files={str(f.relative_to(source)):digest(f) for f in sorted(source.rglob('*')) if f.is_file() and '.git' not in f.relative_to(source).parts}
    assert files==manifest['files'], 'Frozen source files changed: '+role
    assert digest(root/role/'source.tar.gz')==manifest['archive_sha256'], 'Source archive changed: '+role
    binary=Path(bundle['binary']);assert digest(binary)==bundle['binary_sha256'], 'Measured executable changed: '+role
    assert bundle['source_provenance']['source_snapshot_tree_sha256']==manifest['tree_sha256']
    libraries={name:digest(Path(name)) for name in re.findall(r'(/\S+)',bundle['ldd']) if Path(name).is_file()}
    record['roles'][role]=dict(source_commit=manifest['head_at_snapshot'],source_tree_sha256=manifest['tree_sha256'],source_archive_sha256=manifest['archive_sha256'],source_files_unchanged=True,source_archive_unchanged=True,binary_sha256=bundle['binary_sha256'],binary_bytes=binary.stat().st_size,binary_unchanged=True,retained_binary_path=str(binary),retained_source_archive=str(root/role/'source.tar.gz'),linked_library_hashes_after_measurement=libraries)
assert digest(root/'ufm_jet_cache.py')==plan['config']['runner_sha256'],'Frozen runner changed'
rows=[json.loads(line) for line in (root/'results/runs.jsonl').read_text().splitlines()]
record.update(expected_calls=len(plan['jobs']),completed_calls=len(rows),manifest_sha256=digest(root/'results/runs.jsonl'),plan_sha256=digest(root/'results/plan.json'),runner_sha256=plan['config']['runner_sha256'],all_planned_calls_recorded=len(rows)==len(plan['jobs']))
(root/'end-state-verification.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='roles'},indent=2))

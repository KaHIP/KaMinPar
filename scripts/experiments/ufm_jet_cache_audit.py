#!/usr/bin/env python3
"""Independently re-read frozen UFM cache-ablation logs and exact block counts after collection."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--runner', type=Path, required=True)
    args=parser.parse_args()
    root=args.root
    plan=json.loads((root/'plan.json').read_text())
    assert digest(args.runner)==plan['config']['runner_sha256'], 'Wrong analysis runner'
    spec=importlib.util.spec_from_file_location('runner',args.runner)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    rows=[json.loads(line) for line in (root/'runs.jsonl').read_text().splitlines()]
    jobs={job['id']:job for job in plan['jobs']}
    assert len(jobs)==len(plan['jobs'])
    assert len({row['id'] for row in rows})==len(rows)
    failures=[]
    for row in rows:
        job=jobs[row['id']]
        assert all(row[key]==value for key,value in job.items()), row['id']
        assert row['binary_sha256']==plan['config']['binaries'][job['binary_role']]['binary_sha256'], row['id']
        log=root/'raw'/Path(row['log_path']).name
        assert digest(log)==row['log_sha256'], row['id']
        parsed=runner.parse_log(log,row['returncode'])
        for key,value in parsed.items():
            if key!='status': assert row[key]==value,(row['id'],key)
        if row['returncode']==0:
            invalid = (row['status'] != 'ok' or parsed['status'] != 'ok'
                       or parsed.get('result_k') != job['k']
                       or parsed.get('reported_gain_cache') != job['gain_cache']
                       or parsed.get('ufm_time_s') is None)
            resource=root/'resource'/(row['id']+'.txt')
            try:
                rss=int(resource.read_text().splitlines()[-1])
                if rss <= 0 or not isinstance(row.get('max_rss_kib'), int) or row['max_rss_kib'] <= 0:
                    raise ValueError('Missing or invalid recorded RSS')
            except (OSError, ValueError, IndexError):
                assert invalid, (row['id'], 'Valid row lacks valid resource evidence')
                invalid=True
            else:
                assert rss==row['max_rss_kib'], (row['id'], 'Resource evidence changed')
            blocks=root/'blocks'/Path(job['block_path']).name
            if blocks.exists() and 'block_sizes_sha256' in row:
                assert digest(blocks)==row['block_sizes_sha256'], (row['id'], 'Block evidence changed')
            else:
                assert invalid, (row['id'], 'Valid row lacks block evidence')
                invalid=True
            try:
                actual=runner.audit_blocks(blocks,row['n'],row['k'],plan['config']['epsilon'])
            except (OSError, ValueError):
                assert invalid, (row['id'], 'Valid row has invalid block evidence')
                invalid=True
            else:
                if 'balance_audit' in row:
                    assert actual==row['balance_audit'], (row['id'], 'Block audit changed')
                else:
                    assert invalid, (row['id'], 'Valid row lacks block audit')
                    invalid=True
                invalid = invalid or not actual['exact_feasible']
            if invalid:
                failures.append(row['id'])
        else:
            failures.append(row['id'])
    result=dict(complete=len(rows)==len(jobs), expected_invocations=len(jobs),
                audited_invocations=len(rows), failed_or_invalid=failures,
                missing_ids=sorted(set(jobs)-{row['id'] for row in rows}),
                manifest_sha256=digest(root/'runs.jsonl'), plan_sha256=digest(root/'plan.json'),
                runner_sha256=digest(args.runner), audit_script_sha256=digest(Path(__file__)),
                source_base_commits={role: bundle['source_provenance']['source_base_commit'] for role, bundle in plan['config']['binaries'].items()},
                binary_sha256={role: bundle['binary_sha256'] for role, bundle in plan['config']['binaries'].items()})
    (root/'independent-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
    if not result['complete'] or failures: raise SystemExit(1)

if __name__=='__main__':main()

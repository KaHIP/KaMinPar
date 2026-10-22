#!/usr/bin/env python3
"""Independently re-read frozen Jet logs and exact block counts after collection."""
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
        assert row['binary_sha256']==plan['config']['binary_sha256'], row['id']
        log=root/'raw'/Path(row['log_path']).name
        assert digest(log)==row['log_sha256'], row['id']
        parsed=runner.parse_log(log,row['returncode'])
        for key,value in parsed.items():
            if key!='status': assert row[key]==value,(row['id'],key)
        if row['returncode']==0:
            resource=root/'resource'/(row['id']+'.txt')
            assert int(resource.read_text().splitlines()[-1])==row['max_rss_kib']>0,row['id']
            blocks=root/'blocks'/Path(job['block_path']).name
            assert digest(blocks)==row['block_sizes_sha256'],row['id']
            actual=runner.audit_blocks(blocks,row['n'],row['k'],plan['config']['epsilon'])
            assert actual==row['balance_audit'],row['id']
            if (row['status'] != 'ok' or parsed['status'] != 'ok'
                    or parsed.get('result_k') != job['k'] or not actual['exact_feasible']):
                failures.append(row['id'])
        else:
            failures.append(row['id'])
    result=dict(complete=len(rows)==len(jobs), expected_invocations=len(jobs),
                audited_invocations=len(rows), failed_or_invalid=failures,
                missing_ids=sorted(set(jobs)-{row['id'] for row in rows}),
                manifest_sha256=digest(root/'runs.jsonl'), plan_sha256=digest(root/'plan.json'),
                runner_sha256=digest(args.runner), audit_script_sha256=digest(Path(__file__)),
                source_base_commit=plan['config']['source_provenance']['source_base_commit'],
                binary_sha256=plan['config']['binary_sha256'])
    (root/'independent-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
    if not result['complete'] or failures: raise SystemExit(1)

if __name__=='__main__':main()

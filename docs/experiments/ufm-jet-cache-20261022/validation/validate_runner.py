#!/usr/bin/env python3
"""Synthetic runner/auditor checks. These are not experimental measurements."""
from pathlib import Path
from collections import Counter
import copy,hashlib,importlib.util,json,math,re,subprocess,sys,tempfile
RUNNER=Path('/private/tmp/ufm_jet_cache.py'); AUDITOR=Path('/private/tmp/ufm_jet_cache_audit.py')
s=importlib.util.spec_from_file_location('r',RUNNER);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
evidence={'kind':'synthetic_runner_validation_not_measurements','checks':[]}
def passed(name): evidence['checks'].append(name)
root=Path('/private/tmp/jet-final-v2')
rows=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()]; eco=[x for x in rows if x['arm']=='eco_ufm']
parsed=[r.parse_log(root/'raw'/Path(x['log_path']).name,x['returncode']) for x in eco]
assert len(parsed)==36 and all(x['status']=='ok' and x['reported_gain_cache']=='compact-hashing' for x in parsed)
evidence['old_eco_parser_checks']={'logs':36,'ufm_seconds':sum(x['ufm_time_s'] for x in parsed)}
passed('36 archived UFM logs parse with explicit compact-hashing and UFM timers')
config={'ks':[16,1024],'seeds':[1,2,3],'order_seed':20261022,'threads':32,'epsilon':.03,
'binaries':{'stock':{'binary':'/stock/KaMinPar','binary_sha256':'a'*64,'source_provenance':{'source_base_commit':'stockfixture'}},
'candidate':{'binary':'/candidate/KaMinPar','binary_sha256':'b'*64,'source_provenance':{'source_base_commit':'candidatefixture'}}},
'runner_sha256':r.sha256(RUNNER)}
jobs=r.make_jobs(config,[dict(graph=g,n=2000,path='/input/'+g+'.metis') for g in r.GRAPHS],Path('/results'))
assert len(jobs)==108 and set(Counter(x['arm'] for x in jobs).values())=={36}
assert set(Counter((x['arm'],x['position']) for x in jobs).values())=={12}
assert len(Counter(tuple(x['arm'] for x in jobs[i:i+3]) for i in range(0,len(jobs),3)))==6
assert set(Counter(tuple(x['arm'] for x in jobs[i:i+3]) for i in range(0,len(jobs),3)).values())=={6}
assert all(x['command'][0]==('/stock/KaMinPar' if x['arm']=='stock' else '/candidate/KaMinPar') for x in jobs)
assert all(x['command'][-1]==r.CACHE_STRATEGIES[x['arm']] for x in jobs)
passed('108 matched calls with 36 per arm, 12 per arm-position, six of each order, correct executable/cache routing')
fixture=Path(tempfile.mkdtemp(prefix='ufm-cache-runner-synthetic-',dir='/private/tmp'))
for name in ['raw','blocks','resource']: (fixture/name).mkdir()
original=next(x for x in eco if x['k']==16)
text=(root/'raw'/Path(original['log_path']).name).read_text()
blocks=(root/'blocks'/Path(original['block_path']).name).read_bytes()
fixture_config=copy.deepcopy(config);fixture_config.update(ks=[16],seeds=[original['seed']])
fixture_jobs=r.make_jobs(fixture_config,[dict(graph=original['graph'],n=original['n'],path='/synthetic/input')],fixture)
fixture_rows=[]
for job in fixture_jobs:
    log=fixture/'raw'/(job['id']+'.log');log.write_text(text.replace('Gain cache:                  compact-hashing','Gain cache:                  jet') if job['arm']=='jet' else text)
    if job['arm']=='jet': log.write_text(re.sub(r'(Gain cache:\s*)compact-hashing',r'\1jet',log.read_text()))
    blockfile=Path(job['block_path']);blockfile.write_bytes(blocks)
    (fixture/'resource'/(job['id']+'.txt')).write_text('12345\n')
    row=dict(job,returncode=0,log_path=str(log),log_sha256=r.sha256(log),binary_sha256=fixture_config['binaries'][job['binary_role']]['binary_sha256'],
        max_rss_kib=12345,block_sizes_sha256=r.sha256(blockfile),balance_audit=r.audit_blocks(blockfile,job['n'],job['k'],.03),elapsed_s=1.,**r.parse_log(log,0))
    fixture_rows.append(row)
r.write_json(fixture/'plan.json',dict(config=fixture_config,jobs=fixture_jobs))
def audit(records,expect):
    (fixture/'runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    result=subprocess.run([sys.executable,'-B',str(AUDITOR),str(fixture),'--runner',str(RUNNER)],capture_output=True,text=True)
    assert result.returncode==expect,(result.stdout,result.stderr)
    return result
assert all(x['status']=='ok' for x in fixture_rows)
audit(fixture_rows,0);passed('auditor accepts valid three-arm fixture with two binary hashes')
mut=copy.deepcopy(fixture_rows);mut[0]['status']='failed_exact_balance';audit(mut,1);passed('auditor rejects non-ok row status')
job=fixture_jobs[0];log=fixture/'raw'/(job['id']+'.log');saved=log.read_text()
log.write_text(re.sub(r'(RESULT[^\n]*?\bk=)16\b',r'\g<1>17',saved));mut=copy.deepcopy(fixture_rows);mut[0].update(r.parse_log(log,0));mut[0]['log_sha256']=r.sha256(log)
assert mut[0]['result_k']==17;result=audit(mut,1);assert 'Traceback' not in result.stderr;passed('auditor rejects wrong RESULT k with otherwise valid block counts')
log.write_text(saved)
log.write_text(re.sub(r'(Gain cache:\s*)compact-hashing',r'\1jet',saved));mut=copy.deepcopy(fixture_rows);mut[0].update(r.parse_log(log,0));mut[0]['log_sha256']=r.sha256(log)
audit(mut,1);passed('auditor rejects mismatched reported gain-cache selection');log.write_text(saved)
mut=copy.deepcopy(fixture_rows);mut[0]['binary_sha256']='b'*64;audit(mut,1);passed('auditor rejects swapped stock/candidate binary hash')
mut=copy.deepcopy(fixture_rows);mut[0]['returncode']=124;mut[0]['status']='timeout';result=audit(mut,1);assert 'Traceback' not in result.stderr
assert mut[0]['id'] in json.loads((fixture/'independent-audit.json').read_text())['failed_or_invalid'];passed('auditor retains timeout as invalid row without raising')
audit(fixture_rows,0)
resource=fixture/'resource'/(job['id']+'.txt'); resource.unlink()
mut=copy.deepcopy(fixture_rows);mut[0]['status']='invalid_resource_record';mut[0].pop('max_rss_kib')
result=audit(mut,1);assert 'Traceback' not in result.stderr;passed('auditor retains missing RSS evidence for an invalid row')
resource.write_text('invalid\n');mut[0]['max_rss_kib']=None
result=audit(mut,1);assert 'Traceback' not in result.stderr;passed('auditor retains malformed RSS evidence for an invalid row');resource.write_text('12345\n')
blockfile=Path(job['block_path']);blockfile.unlink();mut=copy.deepcopy(fixture_rows);mut[0]['status']='missing_block_audit';mut[0].pop('block_sizes_sha256');mut[0].pop('balance_audit')
result=audit(mut,1);assert 'Traceback' not in result.stderr;passed('auditor retains missing block evidence for an invalid row')
blockfile.write_text('not block counts\n');result=audit(mut,1);assert 'Traceback' not in result.stderr;passed('auditor retains malformed block evidence for an invalid row');blockfile.write_bytes(blocks)
audit(fixture_rows,0)
log.write_text(saved+'\nUsing the default gain cache strategy\n');assert r.parse_log(log,0)['status']=='unsupported_cache';passed('parser rejects cache fallback')
log.write_text(re.sub(r'^\s*Gain cache:[^\n]*$', '', saved,flags=re.MULTILINE));assert r.parse_log(log,0)['status']=='missing_ufm_configuration';passed('parser rejects absent cache confirmation')
log.write_text('Gain cache: compact-hashing\nRESULT cut=0 imbalance=0 feasible=1 k=16\nTIME partitioning=9 partitioning.uncoarsening.refinement.fm=4 partitioning.uncoarsening.refinement.fm.initialize_gain_cache=2 partitioning.uncoarsening.refinement.fm.initialize_gain_cache.allocate=1 partitioning.uncoarsening.coarse_levels.refinement.fm=3 partitioning.uncoarsening.coarse_levels.refinement.fm.initialize_gain_cache=1\n')
p=r.parse_log(log,0);assert p['status']=='ok' and p['ufm_time_s']==7 and p['ufm_phases_s']=={'initialize_gain_cache':3} and p['ufm_cache_initialization_phases_s']=={'allocate':1}
passed('timer roots and direct phases avoid descendant double-counting');log.write_text(saved)
ratio_rows=[dict(graph='synthetic',k=16,seed=1,arm=a,status='ok',cut=c,runtime_s=1,max_rss_kib=1,ufm_time_s=1) for a,c in [('stock',2),('jet',0)]]
c=r.comparison(ratio_rows,'test','stock','jet');assert c['cut_ratio_geomean']==0 and c['cut_ratio_zero_count']==1
ratio_rows[0]['cut']=0;c=r.comparison(ratio_rows,'test','stock','jet');assert c['cut_ratio_geomean']==1
passed('zero candidate cut preserved; zero/zero cut ratio is one')
evidence.update(runner_sha256=r.sha256(RUNNER),auditor_sha256=r.sha256(AUDITOR),fixture_directory=str(fixture),passed=True)
Path('/private/tmp/ufm-cache-runner-validation.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))

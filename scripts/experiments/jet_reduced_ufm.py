#!/usr/bin/env python3
"""Matched reduced UFM-corpus Jet ablation; exclusive Slurm compute nodes only.

The four Jet arms share one executable and differ only in gain-cache/balancer
selection. The fifth arm is KaMinPar eco (ULP + UFM), a quality/runtime reference.
Ratios are candidate / reference: values below one are better. Each seed is
matched but parallel execution is not deterministic. This is a reduced screening
study, not a replacement for a full-corpus experiment.
"""
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

ARMS = {
    "sparse_overload": ["-P", "jet", "--r-jet-gain-cache", "sparse", "--r-jet-balancer", "overload"],
    "jet_overload": ["-P", "jet", "--r-jet-gain-cache", "jet", "--r-jet-balancer", "overload"],
    "sparse_jet": ["-P", "jet", "--r-jet-gain-cache", "sparse", "--r-jet-balancer", "jet"],
    "jet_jet": ["-P", "jet", "--r-jet-gain-cache", "jet", "--r-jet-balancer", "jet"],
    "eco_ufm": ["-P", "eco"],
}
GRAPHS = ["kron_g500-simple-logn19", "recomp_dna1GB_9", "recomp_sources1GB_9",
          "Long_Coup_dt6", "com-lj.ungraph", "soc-flickr-und"]
COMPARISONS = {
    "cache_with_overload": ("sparse_overload", "jet_overload"),
    "cache_with_jet": ("sparse_jet", "jet_jet"),
    "balancer_with_sparse": ("sparse_overload", "sparse_jet"),
    "balancer_with_jet": ("jet_overload", "jet_jet"),
    "combined_vs_original": ("sparse_overload", "jet_jet"),
    "combined_vs_ufm": ("eco_ufm", "jet_jet"),
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FIELDS = re.compile(r"([^\s=]+)=([^\s]+)")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def graph_header(path):
    with path.open() as stream:
        for line in stream:
            if line.strip() and not line.startswith("%"):
                words = line.split()
                n, m = map(int, words[:2])
                fmt = int(words[2]) if len(words) > 2 else 0
                if fmt not in (0, 1):
                    raise ValueError("Exact block audit expects unit node weights: " + str(path))
                return dict(n=n, m=m, fmt=fmt)
    raise ValueError("Missing graph header: " + str(path))


def physical_cores():
    rows = csv.reader(io.StringIO(subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)))
    chosen = {}
    for row in rows:
        if not row or row[0].startswith("#"):
            continue
        cpu, core, package, numa = map(int, row)
        if cpu in os.sched_getaffinity(0):
            chosen.setdefault((package, core), dict(cpu=cpu, core=core, socket=package, numa=numa))
    return [chosen[key] for key in sorted(chosen)]


def parse_log(path, returncode):
    content = ANSI.sub("", path.read_text(errors="replace"))
    results = [dict(FIELDS.findall(x)) for x in re.findall(r"\bRESULT\s+([^\n]+)", content)]
    times = [dict(FIELDS.findall(x)) for x in re.findall(r"\bTIME\s+([^\n]+)", content)]
    parsed = dict(result_count=len(results), time_count=len(times))
    if returncode == 124:
        parsed["status"] = "timeout"
    elif returncode:
        parsed["status"] = "failed"
    elif len(results) != 1 or len(times) != 1:
        parsed["status"] = "invalid_summary"
    else:
        try:
            r, t = results[0], times[0]
            parsed.update(cut=int(r["cut"]), imbalance=float(r["imbalance"]),
                          feasible=r["feasible"] in ("1", "true"), result_k=int(r["k"]),
                          runtime_s=float(t["partitioning"]),
                          timers={key: float(value) for key, value in t.items()})
            jet_roots = {key: value for key, value in parsed["timers"].items()
                         if key.split(".")[-1] == "jet_refiner"}
            parsed["jet_time_s"] = sum(jet_roots.values()) if jet_roots else None
            parsed["jet_phases_s"] = {phase: sum(value for key, value in parsed["timers"].items()
                if key.endswith(".jet_refiner." + phase)) for phase in
                ("initialization", "allocation", "find_moves", "filter_moves", "execute_moves",
                 "rebalance", "update_best_partition", "rollback")} if jet_roots else {}
            parsed["status"] = "ok" if (parsed["feasible"] and parsed["cut"] >= 0
                and math.isfinite(parsed["runtime_s"]) and parsed["runtime_s"] > 0
                and math.isfinite(parsed["imbalance"])) else "invalid_result"
        except (KeyError, ValueError):
            parsed["status"] = "invalid_summary"
    if "ThreadSanitizer" in content or "WARNING: data race" in content:
        parsed["status"] = "instrumented_or_racy"
    return parsed


def audit_blocks(path, n, k, epsilon):
    values = list(map(int, path.read_text().split()))
    upper = math.floor((1.0 + epsilon) * math.ceil(n / k))
    return dict(blocks=len(values), total=sum(values), minimum=min(values, default=0),
                maximum=max(values, default=0), upper=upper,
                exact_feasible=len(values) == k and sum(values) == n
                and all(0 <= v <= upper for v in values))


def run(args):
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Requires an exclusive Slurm Linux compute allocation.")
    if socket.gethostname().split(".")[0] != args.host:
        raise SystemExit("Unexpected compute host")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cores = physical_cores()
    if len(cores) != args.threads:
        raise SystemExit(f"Expected {args.threads} physical cores, found {len(cores)}")
    os.sched_setaffinity(0, [x["cpu"] for x in cores])
    binary = args.binary.resolve(strict=True)
    deps = re.sub(r"\(0x[0-9a-fA-F]+\)", "(ASLR)", subprocess.check_output(["ldd", str(binary)], text=True))
    if any(x in deps for x in ("libtsan", "libasan", "libubsan")):
        raise SystemExit("Instrumented executable cannot be used for performance measurements")
    config = dict(binary=str(binary), binary_sha256=sha256(binary), ldd=deps,
                  source_provenance=json.loads(args.provenance.read_text()),
                  runner_sha256=sha256(Path(__file__)), graphs=sorted(args.graphs),
                  ks=args.ks, seeds=args.seeds, threads=args.threads, epsilon=args.epsilon,
                  timeout_s=args.timeout, arms=ARMS, host=args.host,
                  graph_dir=str(args.graph_dir.resolve()), stage_dir=str(args.stage_dir.resolve()),
                  order_seed=args.order_seed)
    plan_file = root / "plan.json"
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        if plan["config"] != config:
            raise SystemExit("Frozen configuration changed on resume")
    else:
        inputs = []
        for name in sorted(args.graphs):
            src = args.graph_dir / (name + ".metis")
            stat = src.stat()
            digest = sha256(src)
            if (src.stat().st_size, src.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise SystemExit("Input mutated while hashing: " + str(src))
            inputs.append(dict(graph=name, source=str(src), path=str(args.stage_dir / src.name),
                               sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                               **graph_header(src)))
        cells = [dict(graph=g["graph"], n=g["n"], input=g["path"], k=k, seed=seed)
                 for g in inputs for k in args.ks for seed in args.seeds]
        random.Random(args.order_seed).shuffle(cells)
        jobs = []
        # Cyclic positions are balanced within every five cells; reversed cycles
        # limit monotonic drift. All treatments of a cell are adjacent in time.
        arm_names = list(ARMS)
        for index, cell in enumerate(cells):
            cycle = arm_names if (index // 5) % 2 == 0 else list(reversed(arm_names))
            shift = index % len(cycle)
            order = cycle[shift:] + cycle[:shift]
            for pos, arm in enumerate(order):
                job_id = f"{index:03d}-{cell['graph']}-k{cell['k']}-s{cell['seed']}-{arm}"
                block_path = root / "blocks" / (job_id + ".txt")
                command = [str(binary), "-G", cell["input"], "-k", str(cell["k"]),
                           "-t", str(args.threads), "--seed", str(cell["seed"]),
                           "-e", str(args.epsilon), "-v", "-T", "--disable-colors",
                           "--output-block-sizes", str(block_path)] + ARMS[arm]
                jobs.append(dict(cell, id=job_id, cell=index, arm=arm,
                                 position=pos, block_path=str(block_path), command=command))
        plan = dict(config=config, inputs=inputs, jobs=jobs)
        write_json(plan_file, plan)
    for graph in plan["inputs"]:
        src, staged = Path(graph["source"]), Path(graph["path"])
        if sha256(src) != graph["sha256"]:
            raise SystemExit("Input content changed: " + str(src))
        if not staged.exists() or sha256(staged) != graph["sha256"]:
            shutil.copyfile(src, staged)
        if sha256(staged) != graph["sha256"]:
            raise SystemExit("Staged graph hash mismatch")
    write_json(root / ("hardware-" + os.environ["SLURM_JOB_ID"] + ".json"),
               dict(host=socket.gethostname(), job_id=os.environ["SLURM_JOB_ID"],
                    cores=cores, affinity=sorted(os.sched_getaffinity(0)),
                    lscpu=subprocess.check_output(["lscpu"], text=True),
                    allocation=subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]], text=True)))
    for subdir in ("raw", "blocks", "resource"):
        (root / subdir).mkdir(exist_ok=True)
    manifest = root / "runs.jsonl"
    previous = [json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    completed = {x["id"] for x in previous}
    if len(completed) != len(previous):
        raise SystemExit("Duplicate records")
    for job in plan["jobs"]:
        if job["id"] in completed:
            continue
        if sha256(binary) != config["binary_sha256"]:
            raise SystemExit("Executable changed")
        log = root / "raw" / (job["id"] + ".log")
        resource = root / "resource" / (job["id"] + ".txt")
        if log.exists():
            raise SystemExit("Unrecorded log exists: inspect interrupted run " + str(log))
        print("START " + job["id"], flush=True)
        start = time.monotonic()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open("x") as output:
            # Fresh wrapper has exactly one child, so RUSAGE_CHILDREN.ru_maxrss
            # is this invocation's Linux peak RSS, not a cumulative runner peak.
            wrapper = ("import pathlib,resource,subprocess,sys; "
                       "child=subprocess.run(sys.argv[2:]); "
                       "pathlib.Path(sys.argv[1]).write_text(str(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)+'\\n'); "
                       "sys.exit(child.returncode if child.returncode>=0 else 128-child.returncode)")
            command = [sys.executable, "-c", wrapper, str(resource)] + job["command"]
            proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                    env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
            try:
                rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                rc = 124
        row = dict(job, started_utc=started, elapsed_s=time.monotonic()-start, returncode=rc,
                   log_path=str(log), log_sha256=sha256(log), binary_sha256=config["binary_sha256"],
                   slurm_job_id=os.environ["SLURM_JOB_ID"], **parse_log(log, rc))
        if resource.exists():
            try:
                row["max_rss_kib"] = int(resource.read_text().splitlines()[-1])
            except (ValueError, IndexError):
                row["max_rss_kib"] = None
        if rc == 0:
            try:
                row["balance_audit"] = audit_blocks(Path(job["block_path"]), job["n"], job["k"], args.epsilon)
                row["block_sizes_sha256"] = sha256(Path(job["block_path"]))
                if not row["balance_audit"]["exact_feasible"] or row.get("result_k") != job["k"]:
                    row["status"] = "failed_exact_balance"
            except (OSError, ValueError):
                row["status"] = "missing_block_audit"
        with manifest.open("a") as out:
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        print("DONE " + job["id"] + " " + row["status"] + " " + str(row.get("runtime_s")), flush=True)
        report(root)
    if sha256(binary) != config["binary_sha256"]:
        raise SystemExit("Executable changed after experiment")
    report(root)


def geometric(values):
    if not values:
        return None
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(statistics.mean(map(math.log, values)))


def comparison(rows, name, reference, candidate):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["graph"], row["k"], row["seed"]), {})[row["arm"]] = row
    paired = []
    incomplete = []
    for key, arms in grouped.items():
        a, b = arms.get(reference), arms.get(candidate)
        if a and b and a["status"] == b["status"] == "ok":
            paired.append(dict(graph=key[0], k=key[1], seed=key[2],
                               runtime_ratio=b["runtime_s"]/a["runtime_s"],
                               cut_ratio=b["cut"]/a["cut"] if a["cut"] else (1.0 if b["cut"] == 0 else None),
                               rss_ratio=b.get("max_rss_kib", 0)/a["max_rss_kib"] if a.get("max_rss_kib") and b.get("max_rss_kib") else None,
                               jet_ratio=b["jet_time_s"]/a["jet_time_s"] if a.get("jet_time_s") and b.get("jet_time_s") else None,
                               reference_s=a["runtime_s"], candidate_s=b["runtime_s"]))
        else:
            incomplete.append(dict(graph=key[0], k=key[1], seed=key[2],
                                   reference_status=a["status"] if a else "missing",
                                   candidate_status=b["status"] if b else "missing"))
    result = dict(name=name, reference=reference, candidate=candidate, valid_pairs=len(paired),
                  incomplete_pairs=incomplete, pairs=paired)
    for metric in ("runtime_ratio", "cut_ratio", "rss_ratio", "jet_ratio"):
        values = [r[metric] for r in paired if r[metric] is not None and r[metric] >= 0]
        result[metric + "_geomean"] = geometric(values)
        result[metric + "_zero_count"] = sum(value == 0 for value in values)
        result[metric + "_undefined_count"] = sum(r[metric] is None for r in paired)
    if paired:
        result["aggregate_runtime_ratio"] = sum(r["candidate_s"] for r in paired)/sum(r["reference_s"] for r in paired)
        # Cluster bootstrap over graphs, preserving all matched k/seed cells.
        graphs = sorted({r["graph"] for r in paired})
        rng = random.Random(42)
        sampled = []
        for _ in range(2000):
            selected = rng.choices(graphs, k=len(graphs))
            vals = [r["runtime_ratio"] for g in selected for r in paired if r["graph"] == g]
            sampled.append(geometric(vals))
        sampled.sort()
        result["runtime_ratio_graph_bootstrap_95pct"] = [sampled[50], sampled[1949]]
    return result


def report(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    runs_file = root / "runs.jsonl"
    rows = [json.loads(x) for x in runs_file.read_text().splitlines()] if runs_file.exists() else []
    summary = dict(planned=len(plan["jobs"]), completed=len(rows),
                   statuses=dict(Counter(r["status"] for r in rows)),
                   complete=len(rows) == len(plan["jobs"]),
                   arm_totals={arm: dict(valid_runs=sum(r["arm"] == arm and r["status"] == "ok" for r in rows),
                       runtime_s=sum(r.get("runtime_s", 0) for r in rows if r["arm"] == arm and r["status"] == "ok"),
                       jet_time_s=sum(r.get("jet_time_s") or 0 for r in rows if r["arm"] == arm and r["status"] == "ok"),
                       jet_phases_s={phase: sum(r.get("jet_phases_s", {}).get(phase, 0) for r in rows
                            if r["arm"] == arm and r["status"] == "ok") for phase in
                            ("initialization", "allocation", "find_moves", "filter_moves", "execute_moves",
                             "rebalance", "update_best_partition", "rollback")}) for arm in ARMS},
                   comparisons=[comparison(rows, name, *arms) for name, arms in COMPARISONS.items()],
                   per_k={str(k): [comparison([r for r in rows if r["k"] == k], name, *arms)
                                    for name, arms in COMPARISONS.items()] for k in plan["config"]["ks"]})
    write_json(root / "summary.json", summary)
    columns = ["graph", "k", "seed", "arm", "status", "runtime_s", "cut", "imbalance", "max_rss_kib", "jet_time_s", "elapsed_s", "id"]
    with (root / "runs.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Reduced UFM Jet experiment", "", f"Completed {len(rows)}/{len(plan['jobs'])}; statuses: {summary['statuses']}.", "",
             "All ratios below are candidate/reference; smaller is better. Runtime is total partitioning time, excluding input/output; RSS is process-wide peak resident memory.", "",
             "| Comparison | Matched runs | Runtime ratio | Cut ratio | RSS ratio | Jet time ratio |", "|---|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"
    for c in summary["comparisons"]:
        lines.append(f"| {c['name']} | {c['valid_pairs']} | {fmt(c['runtime_ratio_geomean'])} | {fmt(c['cut_ratio_geomean'])} | {fmt(c['rss_ratio_geomean'])} | {fmt(c['jet_ratio_geomean'])} |")
    lines += ["", "The study uses six selected UFM corpus graphs, k=16/1024 and three seeds, with 32 physical cores on one exclusive allocation. The eco arm is the repository's unconstrained LP + UFM preset. Graph selection is fixed before measurements. This screening subset does not establish full-corpus superiority.", "",
              "`plan.json` records every command, source/executable/input hashes and arm order. `runs.jsonl` retains failed, timed-out and infeasible runs; `summary.json` lists missing/invalid comparison pairs. Every successful run has an independently checked exact block-size output. All four Jet arms use the same binary. The legacy arm uses explicit sparse cache + overload balancer within the candidate source, not a separately compiled pre-change program. All Jet arms also share the best-partition feasibility safety correction. Per-level Jet timers are summed without double-counting descendants; phase totals are included in summary.json.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--root", type=Path, required=True)
    run_p.add_argument("--binary", type=Path, required=True)
    run_p.add_argument("--provenance", type=Path, required=True)
    run_p.add_argument("--graph-dir", type=Path, default=Path("/nfs/work/graph_benchmark_sets/ufm_paper/combined"))
    run_p.add_argument("--stage-dir", type=Path, required=True)
    run_p.add_argument("--graphs", nargs="+", default=GRAPHS)
    run_p.add_argument("--ks", nargs="+", type=int, default=[16, 1024])
    run_p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    run_p.add_argument("--threads", type=int, default=32)
    run_p.add_argument("--host", default="dijkstra")
    run_p.add_argument("--epsilon", type=float, default=.03)
    run_p.add_argument("--timeout", type=int, default=300)
    run_p.add_argument("--order-seed", type=int, default=20261022)
    report_p = sub.add_parser("report")
    report_p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        run(args)
    else:
        report(args.root)

if __name__ == "__main__":
    main()

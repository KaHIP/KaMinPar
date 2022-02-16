#!/bin/zsh

BENCHMARK_DIR="${BENCHMARK_DIR}"
BINARY_CURRENT="${KAMINPAR_CURRENT}"
BINARY_BASELINE="${KAMINPAR_STABLE}"

if [[ ! -d "$BENCHMARK_DIR" ]]; then
	echo "ERROR: benchmark directory does not exist"
	echo "ERROR: expected benchmark instances at $BENCHMARK_DIR"
	exit 1
fi
if [[ ! -x "$BINARY_CURRENT" ]]; then
	echo "ERROR: current build does not exist"
	echo "ERROR: expected build at $BINARY_CURRENT"
	exit 1
fi
if [[ ! -x "$BINARY_BASELINE" ]]; then
	echo "ERROR: baseline build does not exist"
	echo "ERROR: expected build at $BINARY_BASELINE"
	exit 1
fi

# Experiment configuration
KS=(2 64)
CORES=(1 4)
SEEDS=(1 2 3 4 5)

function get_cut {
	echo "$1" | cut -d ',' -f 1
}
function get_feasible {
	echo "$1" | cut -d ',' -f 3
}
function get_time {
	echo "$1" | cut -d ',' -f 4
}
function get_ok {
	echo "$1" | cut -d ',' -f 5
}

fail=0

for k in $KS; do
	for core in $CORES; do
		# Highest core used for thread pinning via taskset
		max_core=$((core - 1))

		echo "#==============================================================================="
		echo "# $core cores, $k blocks"
		echo "#==============================================================================="

		num_runs=0
		total_cut_current=0
		total_cut_baseline=0
		total_time_current=0
		total_time_baseline=0

		for seed in $SEEDS; do
			for graph in "${BENCHMARK_DIR}"/*.graph; do
				num_runs=$((num_runs + 1))

				# Partition graph with current build and baseline build
				log_current="$(taskset -c 0-$max_core "$BINARY_CURRENT" -G "$graph" -k $k -s $seed -t $core)"
				out_current=$(echo "$log_current" | gawk -f parse_output.awk)
				log_baseline="$(taskset -c 0-$max_core "$BINARY_BASELINE" -G "$graph" -k $k -s $seed -t $core)"
				out_baseline=$(echo "$log_baseline" | gawk -f parse_output.awk)

				ok_current=$(get_ok "$out_current")
				ok_baseline=$(get_ok "$out_baseline")
				if (( ok_current == 0 || ok_baseline == 0 )); then
					echo "ERROR: run failed: current=${ok_current} baseline=${ok_baseline}"
					if (( ok_baseline == 0 )); then
						echo "ERROR: failed baseline run:"
						echo "$log_baseline"
					else
						echo "ERROR: failed current run:"
						echo "$log_current"
					fi
					continue
				fi

				# Parse CSV output
				cut_current=$(get_cut "$out_current")
				total_cut_current=$((total_cut_current + cut_current))
				feasible_current=$(get_feasible "$out_current")
				time_current=$(get_time "$out_current")
				total_time_current=$((total_time_current + time_current))

				cut_baseline=$(get_cut "$out_baseline")
				total_cut_baseline=$((total_cut_baseline + cut_baseline))
				feasible_baseline=$(get_feasible "$out_current")
				time_baseline=$(get_time "$out_current")
				total_time_baseline=$((total_time_baseline + time_baseline))
				
				# Only report single instances if the difference is very large to the baseline
				if [[ $feasible_baseline == 1 && $feasible_current == 0 ]]; then
					echo "ERROR: current build computes infeasible partition for $graph"
					echo "ERROR: output current: $out_current -- feasible: $feasible_current"
					echo "ERROR: output baseline: $out_baseline -- feasible: $feasible_baseline"
					fail=1
				fi
				if (( cut_current > 1.2 * cut_baseline )); then
					echo "WARNING: edge cut (${cut_current}) on single instance $graph is more than 20% worse than the baseline edge cut (${cut_baseline})"
				fi
				if (( time_current > 1.05 * time_baseline )); then
					echo "WARNING: time (${time_current}) on single instance $graph is more than 10% worse than the baseline time (${time_baseline})"
				fi
			done
		done
		
		# Report small deviations in the mean edge cut or running time
		avg_cut_current=$((total_cut_current / num_runs))
		avg_time_current=$((total_time_current / num_runs))
		avg_cut_baseline=$((total_cut_baseline / num_runs))
		avg_time_baseline=$((total_time_baseline / num_runs))

		if (( avg_cut_current > 1.03 * avg_cut_baseline )); then
			echo "ERROR: average cut (${avg_cut_current}) is more than 3% worse than the baseline cut (${avg_cut_baseline})"
			fail=1
		fi
		if (( avg_time_current > 1.03 * avg_time_baseline )); then
			echo "ERROR: average time (${avg_time_current}) is more than 3% slower than the baseline time (${avg_time_baseline})"
			fail=1
		fi
	done
done

exit $fail

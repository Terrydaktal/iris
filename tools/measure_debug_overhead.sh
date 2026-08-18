#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

output_dir=${1:-"$root/.debuggability-overhead"}
mkdir -p "$output_dir"
if [[ -n ${IRIS_OVERHEAD_SHARED_TARGET:-} ]]; then
	minimal_target=$IRIS_OVERHEAD_SHARED_TARGET
	observable_target=$IRIS_OVERHEAD_SHARED_TARGET
	cleanup_targets=false
else
	minimal_target=$(mktemp -d)
	observable_target=$(mktemp -d)
	cleanup_targets=true
fi
if [[ $cleanup_targets == true ]]; then
	trap 'rm -rf "$minimal_target" "$observable_target"' EXIT
fi

build_and_measure() {
	local name=$1
	local target_dir=$2
	shift 2
	printf '[%s] building release artifact\n' "$name" >&2
	CARGO_TARGET_DIR="$target_dir" cargo build --release "$@" >/dev/null
	local binary="$target_dir/release/iris"
	local size
	size=$(stat -c '%s' "$binary")
	printf '[%s] binary_bytes=%s\n' "$name" "$size" | tee -a "$output_dir/summary.txt"
	for run in 1 2 3; do
		python3 - "$binary" "$name" "$run" >>"$output_dir/cli-runs.txt" <<'PY'
import resource
import subprocess
import sys
import time

binary, name, run = sys.argv[1:]
started = time.monotonic()
subprocess.run([binary, "--build-info"], check=True, stdout=subprocess.DEVNULL)
elapsed = time.monotonic() - started
rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"[{name}] cli_run={run} elapsed_s={elapsed:.6f} max_rss_kib={rss_kib}")
PY
	done
}

: >"$output_dir/summary.txt"
: >"$output_dir/cli-runs.txt"
build_and_measure minimal "$minimal_target"
build_and_measure observable "$observable_target" --features diagnostics

printf 'Measurements written to %s\n' "$output_dir"
printf 'This compares artifact size and CLI build-info startup only; use a representative GUI/indexer workload before making a performance claim.\n'

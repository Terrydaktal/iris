#!/usr/bin/env bash
set -euo pipefail

usage() {
	printf 'Usage: %s PID [OUTPUT_DIR]\n' "$(basename "$0")" >&2
	printf 'Capture bounded process, thread, debugger, profiler, and Iris diagnostic evidence.\n' >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
	usage
	exit 64
fi

pid=$1
if [[ ! $pid =~ ^[0-9]+$ || ! -d "/proc/$pid" ]]; then
	printf 'Process does not exist: %s\n' "$pid" >&2
	exit 1
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_dir=${2:-"./iris-diagnostic-$pid-$timestamp"}
max_bundle_bytes=${IRIS_DIAGNOSTICS_MAX_BUNDLE_BYTES:-67108864}
if [[ ! $max_bundle_bytes =~ ^[0-9]+$ || $max_bundle_bytes -eq 0 ]]; then
	printf 'IRIS_DIAGNOSTICS_MAX_BUNDLE_BYTES must be a positive integer\n' >&2
	exit 64
fi
if [[ -e $output_dir && ! -d $output_dir ]]; then
	printf 'Output path is not a directory: %s\n' "$output_dir" >&2
	exit 1
fi
mkdir -p "$output_dir"
first_entry=$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)
if [[ -n $first_entry ]]; then
	printf 'Output directory must be empty: %s\n' "$output_dir" >&2
	exit 1
fi

check_bundle_budget() {
	local current_bytes
	current_bytes=$(du -sb "$output_dir" | awk '{print $1}')
	if ((current_bytes > max_bundle_bytes)); then
		printf 'Diagnostic bundle exceeded %s bytes (currently %s)\n' "$max_bundle_bytes" "$current_bytes" >&2
		exit 1
	fi
}

if ! exe=$(readlink -f "/proc/$pid/exe"); then
	printf 'Cannot resolve executable for process %s\n' "$pid" >&2
	exit 1
fi

working_directory=$(readlink "/proc/$pid/cwd" || true)
printf '%s\n' "$working_directory" >"$output_dir/replay-working-directory.txt"

printf 'Capturing PID %s into %s\n' "$pid" "$output_dir" >&2

{
	printf 'captured_at_utc=%s\n' "$timestamp"
	printf 'pid=%s\n' "$pid"
	printf 'exe=%s\n' "$exe"
	printf 'command_line='
	tr '\0' ' ' <"/proc/$pid/cmdline"
	printf '\n'
	printf 'working_directory=%s' "$working_directory"
	printf '\n'
	printf '\n[build-id]\n'
	if command -v readelf >/dev/null 2>&1; then
		readelf -n "$exe" 2>&1 || true
	else
		printf 'readelf unavailable\n'
	fi
	printf '\n[versions]\n'
	printf 'kernel='
	uname -srvm
	if command -v gdb >/dev/null 2>&1; then gdb --version | head -n 1; else printf 'gdb unavailable\n'; fi
	if command -v perf >/dev/null 2>&1; then perf --version; else printf 'perf unavailable\n'; fi
} >"$output_dir/identity.txt"

if command -v sha256sum >/dev/null 2>&1; then
	sha256sum "$exe" >"$output_dir/artifact-sha256.txt"
else
	printf 'sha256sum unavailable\n' >"$output_dir/artifact-sha256.txt"
fi
check_bundle_budget

if [[ -r "/proc/$pid/cmdline" ]]; then
	mapfile -d '' command_parts <"/proc/$pid/cmdline" || true
	if ((${#command_parts[@]} > 0)); then
		printf '%q ' "${command_parts[@]}" >"$output_dir/replay-command.txt"
		printf '\n' >>"$output_dir/replay-command.txt"
	else
		printf 'command line unavailable\n' >"$output_dir/replay-command.txt"
	fi
else
	printf 'command line unavailable\n' >"$output_dir/replay-command.txt"
fi

if [[ -r "/proc/$pid/environ" ]]; then
	if [[ ${IRIS_CAPTURE_FULL_ENV:-0} == 1 ]]; then
		tr '\0' '\n' <"/proc/$pid/environ" >"$output_dir/replay-environment.txt"
	else
		tr '\0' '\n' <"/proc/$pid/environ" |
			grep -E '^(HOME=|PATH=|USER=|IRIS_|RUST_BACKTRACE=|RUST_LOG=|WAYLAND_DISPLAY=|DISPLAY=|XDG_RUNTIME_DIR=|XDG_SESSION_TYPE=|LD_LIBRARY_PATH=)' \
				>"$output_dir/replay-environment.txt" || true
	fi
else
	printf 'environment unavailable\n' >"$output_dir/replay-environment.txt"
fi

for proc_file in status limits schedstat smaps_rollup oom_score oom_score_adj cgroup; do
	if [[ -r "/proc/$pid/$proc_file" ]]; then
		cp "/proc/$pid/$proc_file" "$output_dir/proc-$proc_file.txt"
	fi
done
check_bundle_budget

capture_cgroup_memory_events() {
	local hierarchy controllers relative cgroup_root events_path
	if [[ ! -r "/proc/$pid/cgroup" ]]; then
		printf 'cgroup_path=unavailable\n'
		printf 'cgroup_memory_events=unavailable\n'
		return
	fi
	while IFS=: read -r hierarchy controllers relative; do
		[[ -n $hierarchy && -n $relative ]] || continue
		if [[ -n $controllers && ",$controllers," != *,memory,* ]]; then
			continue
		fi
		[[ $relative != *..* ]] || continue
		cgroup_root=/sys/fs/cgroup${relative}
		events_path=$cgroup_root/memory.events
		if [[ -r $events_path ]]; then
			printf 'cgroup_path=%s\n' "$relative"
			printf 'cgroup_memory_events_path=%s\n' "$events_path"
			cat "$events_path"
			return
		fi
	done <"/proc/$pid/cgroup"
	printf 'cgroup_path=unavailable\n'
	printf 'cgroup_memory_events=unavailable\n'
}

{
	printf '[process-memory-kill-evidence]\n'
	if [[ -r "/proc/$pid/oom_score" ]]; then
		printf 'oom_score='
		cat "/proc/$pid/oom_score"
	else
		printf 'oom_score=unavailable\n'
	fi
	if [[ -r "/proc/$pid/oom_score_adj" ]]; then
		printf 'oom_score_adj='
		cat "/proc/$pid/oom_score_adj"
	else
		printf 'oom_score_adj=unavailable\n'
	fi
	capture_cgroup_memory_events
} >"$output_dir/oom-evidence.txt"

if [[ -r "/proc/$pid/maps" ]]; then
	cp "/proc/$pid/maps" "$output_dir/proc-maps.txt"
fi

{
	for fd in "/proc/$pid/fd"/*; do
		[[ -e "$fd" ]] || continue
		printf '%s -> %s\n' "${fd##*/}" "$(readlink "$fd" || true)"
	done
} >"$output_dir/proc-fd.txt"

capture_command() {
	local name=$1
	shift
	if ! command -v "$1" >/dev/null 2>&1; then
		printf '%s unavailable\n' "$1" >"$output_dir/$name.txt"
		return 0
	fi
	timeout 20s "$@" >"$output_dir/$name.txt" 2>&1 || {
		printf '\n[command exited with status %s]\n' "$?" >>"$output_dir/$name.txt"
	}
}

capture_command nvidia-smi nvidia-smi
check_bundle_budget

iris_bin=${IRIS_BIN:-iris}
if command -v "$iris_bin" >/dev/null 2>&1; then
	capture_iris_diagnostic() {
		local name=$1
		shift
		local output="$output_dir/iris-$name.json"
		local status
		if timeout 5s "$iris_bin" --diagnose-pid "$pid" "$@" >"$output" 2>&1; then
			status=0
		else
			status=$?
		fi
		printf '%s_exit_status=%s\n' "$name" "$status" >>"$output_dir/iris-validation.txt"
		if command -v jq >/dev/null 2>&1 && [[ $status -eq 0 ]]; then
			local observed_pid
			observed_pid=$(jq -r '.process.pid // empty' "$output" 2>/dev/null || true)
			if [[ $name == snapshot && $observed_pid != "$pid" ]]; then
				printf '%s_pid_mismatch=expected:%s observed:%s\n' "$name" "$pid" "${observed_pid:-missing}" >>"$output_dir/iris-validation.txt"
			fi
		fi
	}
	capture_iris_diagnostic snapshot snapshot
	capture_iris_diagnostic events events 512
	check_bundle_budget
else
	printf 'iris command unavailable\n' >"$output_dir/iris-snapshot.json"
	printf 'iris command unavailable\n' >"$output_dir/iris-events.json"
fi

if command -v gdb >/dev/null 2>&1; then
	timeout 15s gdb -batch -p "$pid" \
		-ex 'set pagination off' \
		-ex 'set confirm off' \
		-ex 'info threads' \
		-ex 'thread apply all bt full' \
		>"$output_dir/gdb-thread-backtraces.txt" 2>&1 || true
else
	printf 'gdb unavailable\n' >"$output_dir/gdb-thread-backtraces.txt"
fi
check_bundle_budget

if command -v perf >/dev/null 2>&1; then
	timeout 20s perf record --call-graph dwarf -o "$output_dir/perf.data" -p "$pid" -- sleep 5 \
		>"$output_dir/perf-record.txt" 2>&1 || true
	if [[ -s "$output_dir/perf.data" ]]; then
		timeout 20s perf report --stdio -i "$output_dir/perf.data" \
			>"$output_dir/perf-report.txt" 2>&1 || true
	fi
else
	printf 'perf unavailable\n' >"$output_dir/perf-record.txt"
fi
check_bundle_budget

printf 'Capture complete: %s\n' "$output_dir" >&2
{
	printf 'capture_schema=2\n'
	printf 'pid=%s\n' "$pid"
	printf 'replay_command=replay-command.txt\n'
	printf 'replay_working_directory=replay-working-directory.txt\n'
	printf 'replay_environment=replay-environment.txt\n'
	printf 'artifact_sha256=artifact-sha256.txt\n'
	printf 'oom_evidence=oom-evidence.txt\n'
	printf 'max_bundle_bytes=%s\n' "$max_bundle_bytes"
	printf 'bundle_bytes=%s\n' "$(du -sb "$output_dir" | awk '{print $1}')"
	printf 'process_alive_at_completion=%s\n' "$(if [[ -d "/proc/$pid" ]]; then printf true; else printf false; fi)"
} >"$output_dir/capture-manifest.txt"
check_bundle_budget

#!/usr/bin/env bash
set -euo pipefail

usage() {
	printf 'Usage: %s CAPTURE_DIR [--run]\n' "$(basename "$0")" >&2
	printf 'Show the exact command captured from a live Iris process; --run replays it explicitly.\n' >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
	usage
	exit 64
fi

capture_dir=$1
run=false
if [[ ${2:-} == "--run" ]]; then
	run=true
elif [[ $# -eq 2 ]]; then
	usage
	exit 64
fi

command_file=$capture_dir/replay-command.txt
if [[ ! -r $command_file ]]; then
	printf 'Replay command is unavailable: %s\n' "$command_file" >&2
	exit 1
fi

working_directory_file=$capture_dir/replay-working-directory.txt
environment_file=$capture_dir/replay-environment.txt

print_context() {
	if [[ -r $working_directory_file ]]; then
		local working_directory
		working_directory=$(<"$working_directory_file")
		if [[ -n $working_directory ]]; then
			printf 'cd -- %q\n' "$working_directory"
		fi
	fi
	if [[ -r $environment_file ]]; then
		local environment_line
		while IFS= read -r environment_line; do
			[[ $environment_line == *=* ]] || continue
			printf 'export %q\n' "$environment_line"
		done <"$environment_file"
	fi
	cat "$command_file"
}

print_context
if [[ $run == true ]]; then
	if [[ ! -r $working_directory_file ]]; then
		printf 'Replay working directory is unavailable: %s\n' "$working_directory_file" >&2
		exit 1
	fi
	working_directory=$(<"$working_directory_file")
	if [[ -z $working_directory || ! -d $working_directory ]]; then
		printf 'Replay working directory does not exist: %s\n' "${working_directory:-<empty>}" >&2
		exit 1
	fi
	cd -- "$working_directory"
	if [[ -r $environment_file ]]; then
		while IFS= read -r environment_line; do
			[[ $environment_line == *=* ]] || continue
			export "${environment_line?}"
		done <"$environment_file"
	fi
	printf 'Replaying the captured command from %s.\n' "$working_directory" >&2
	bash -c "$(cat "$command_file")"
fi

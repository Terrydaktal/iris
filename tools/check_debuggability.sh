#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

cargo fmt --all -- --check
cargo check
cargo check --features diagnostics
cargo test --bin iris
cargo test --bin iris --features diagnostics

if [[ -n ${DEBUGGABILITY_VALIDATOR:-} ]]; then
	python3 "$DEBUGGABILITY_VALIDATOR" --strict debuggability.toml
else
	python3 - <<'PY'
import tomllib
from pathlib import Path

contract = tomllib.loads(Path("debuggability.toml").read_text(encoding="utf-8"))
assert contract["schema_version"] == 5
assert contract["template"] is False
assert contract["profile"] == "stateful"
print("basic debuggability contract check passed")
PY
fi

if command -v shellcheck >/dev/null 2>&1; then
	shellcheck tools/diagnose_iris.sh tools/replay_iris_capture.sh tools/measure_debug_overhead.sh tools/check_debuggability.sh
fi

replay_test_dir=$(mktemp -d)
trap 'rm -rf "$replay_test_dir"' EXIT
printf '%s\n' "$root" >"$replay_test_dir/replay-working-directory.txt"
printf 'IRIS_REPLAY_TEST=ok\n' >"$replay_test_dir/replay-environment.txt"
printf '%s\n' "printf \"%s\\n\" \"\$IRIS_REPLAY_TEST\"" >"$replay_test_dir/replay-command.txt"
tools/replay_iris_capture.sh "$replay_test_dir" --run | grep -qx 'ok'

printf 'debuggability checks passed\n'

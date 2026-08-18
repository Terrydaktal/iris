# Debugging Iris

Iris has two build modes. The normal release binary keeps the diagnostic code out
of the hot path. The observable release adds bounded runtime evidence and a
private local control socket. The socket is always available in the observable
build, but collection remains disabled unless requested.

## Build Modes

Minimal release, for normal use:

```bash
cargo build --release
```

Observable release, for a run that may need live diagnosis:

```bash
cargo build --release --features diagnostics
IRIS_DIAGNOSTICS=1 IRIS_DIAGNOSTICS_TTL_SECS=900 \
  target/release/iris --no-daemon
```

The observable build still uses the release optimizer. It keeps line tables in
the release artifact so GDB and `perf` can resolve optimized stacks without
adding runtime instrumentation to the minimal build.

Build identity is available without starting a window:

```bash
target/release/iris --build-info
target/release/iris --capabilities
```

The build identity includes the package version, Git revision, dirty-tree state,
Rust compiler version, target, profile, and whether diagnostics were compiled.

## Live Evidence

From another terminal, query an observable process:

```bash
target/release/iris --diagnose-pid PID snapshot
target/release/iris --diagnose-pid PID events 512
target/release/iris --diagnose-pid PID activate 900
target/release/iris --diagnose-pid PID set heartbeat off
target/release/iris --diagnose-pid PID set events on
```

Each observable process gets its own socket under `$XDG_RUNTIME_DIR` when
available, named with the owning user's sanitized name and PID, and has mode
`0600`. It is removed when the process exits. Requests and responses are
bounded. A process started without `IRIS_DIAGNOSTICS=1` can be activated later
through this socket. The
snapshot is best effort rather than an atomic stop-the-world state: it reports
the GUI state, heartbeat age, frame count, active background count, thumbnail
activity, database/grid loading flags, task counters, active named tasks, task
and thumbnail high-water marks, metadata lock-wait totals, process RSS/OOM
evidence, the last structured task failure, build identity, and unavailable
subsystems. It intentionally does not include media paths or OCR/image content.

Diagnostic categories are independently switchable:

- `heartbeat`: GUI-loop liveness and lightweight state counters.
- `events`: bounded semantic event ring, capacity 512.
- `tasks`: named task registry and task counters.

All diagnostic data is in memory. The event ring drops the oldest event when it
is full and reports the overwrite count. Configuration changes are also events.
An observable build supports a safe synthetic probe for validating capture
tooling:

```bash
target/release/iris --diagnose-pid PID inject task_failure synthetic_probe synthetic_code synthetic_reason
```

No diagnostic file is written by Iris unless `IRIS_DIAGNOSTICS_FAILURE_FILE` is
explicitly set; then the last structured task failure is written atomically to
that path.

## Capture A Running Process

The repository includes a bounded external capture script. It does not need a
debug flag inside Iris and targets exactly one PID:

```bash
tools/diagnose_iris.sh PID ./iris-diagnostic-capture
```

It captures, when permitted and installed:

- process identity, executable Build ID, command line, limits, scheduler data,
  memory summary, mappings, and file descriptors;
- `nvidia-smi` output;
- Iris snapshot and event output;
- OOM score, cgroup memory events, replay command, and selected diagnostic environment;
- all-thread GDB backtraces;
- a five-second `perf` recording and text report.

The script uses bounded `timeout` calls and records unavailable tools instead of
failing the whole capture. GDB/perf attachment can be refused by the kernel's
ptrace policy or user permissions; that limitation is retained in the output.
Use `IRIS_BIN=/path/to/observable/iris` when `iris` in `PATH` is not the
observable build.

The capture also writes the shell-escaped command, working directory, selected
environment, and SHA-256 identity of the executable. Print the replay context
without executing it:

```bash
tools/replay_iris_capture.sh ./iris-diagnostic-PID-TIMESTAMP
```

Replay is intentionally opt-in because the command can reopen private media or
start another GUI process. `--run` restores the captured working directory and
selected environment before executing the command:

```bash
tools/replay_iris_capture.sh ./iris-diagnostic-PID-TIMESTAMP --run
```

The default environment capture is limited to runtime and Iris variables. Set
`IRIS_CAPTURE_FULL_ENV=1` before running `diagnose_iris.sh` if a full environment
is required; review it for secrets before sharing the bundle. Set
`IRIS_DIAGNOSTICS_MAX_BUNDLE_BYTES` to change the default 64 MiB bundle limit.

## Crash And Post-Mortem Evidence

Iris does not install a crash handler or write a persistent diagnostic journal.
On this system, core dumps are handled by systemd-coredump. Use the system
tools to locate a preserved dump, then run the capture script only while the
process is still alive; for a finished process use GDB directly against the
exact executable and core. The executable's Build ID and `--build-info` output
must match the core's captured process.

The external capture bundle can contain command lines, mappings, executable
paths, cgroup paths, and file descriptor targets. Treat it as private. Do not
upload it without reviewing those files first. The bundle identifies the exact
executable with its path, Build ID, and SHA-256; it does not copy the executable
into the bundle.

## Indexer Status

The Python indexer writes one atomic status file per collection and run under
`embedimages-status/`, using a unique run ID so concurrent collections or runs
cannot overwrite each other. Repeated per-file status updates are coalesced and
written at most every 0.5 seconds by default, while stage completion, failure,
fatal GPU errors, and database flush states are written immediately. Set
`EMBEDIMAGES_STATUS_INTERVAL_SECONDS` to change the interval, with a minimum of
50 ms. The JSON includes a schema number, process ID, run ID, current stage,
and current file/frame state. This bounds status I/O without removing the last
known operation from a live diagnostic. Each status record also includes bounded
`effective_configuration` and rerun/stage policy provenance, including model,
device, batch, threshold, worker, and force-rerun decisions.

## Known Limits

- The snapshot is not a global atomic view across egui, Rayon, LanceDB, and GPU
  runtimes.
- Expensive Iris workers are registered by name; aggregate GUI activity is
  reported separately. Small helper threads may still only appear in aggregate
  state.
- The diagnostic event ring is not a durable audit trail and does not record
  image edits or database mutations.
- A crash or power loss can leave the last status file up to the configured
  coalescing interval behind the actual operation.
- The OOM/cgroup fields are evidence, not proof: kernels, cgroup versions, and
  permission policy may omit them.

## Assurance Checks

Run the repository-level build, test, shell, and contract checks with:

```bash
tools/check_debuggability.sh
```

If the local debugging skill validator is available, set
`DEBUGGABILITY_VALIDATOR` to its `validate_contract.py` path before running the
script to perform strict schema validation as well.

To measure the observable-build overhead separately from a real workload:

```bash
tools/measure_debug_overhead.sh
```

It builds isolated minimal and diagnostics release artifacts and records binary
size plus three `--build-info` startup/RSS samples. This is a smoke measurement,
not a substitute for measuring a representative gallery or indexing run.

# Application Build Versioning

## Layout

```text
APP-VERSION                         tracked manual epoch: 0.HISTORY_GENERATION
build-support/
  app_version.py                    commit count, locked counter and frozen metadata
  app_version.rs                    Cargo build-script adapter
  app_version_runtime.rs            early, headless --version/-V handler
  AppVersion.cmake                  CMake executable integration
  stamp_bundle.cjs                   Node/Electron/web bundle build stamp
  build_go.py                        Go linker-version build entry point
  app_backend.py                     Python wheel/sdist/editable build adapter
  python_runtime.py                  template for the packaged version reader
  AppVersion.gradle.kts              Android variant/default versionName adapter
  AppVersionTask.gradle.kts           cached configuration preview + build reservation
  app-version.json                   optional generated-state location override
  README.md                         this build contract
.app-version/                       ignored, machine-local counter database and lock
```

These files are maintained in the `version` repository and vendored into consuming
repositories. Copies require no sibling checkout or network access. Python 3.11+
and Git are build dependencies only; built native applications do not invoke them.

## Version Fields

The executable version is `0.HISTORY_GENERATION.COMMIT_COUNT-BUILD_COUNT`:

- `0` is reserved, not an upstream rebase counter.
- `HISTORY_GENERATION` starts at zero. Change `APP-VERSION` manually only after an
  intentional whole-history reconstruction, not an amend or a small fixup pass.
- `COMMIT_COUNT` is `git rev-list --count HEAD`, including all reachable commits.
  It is the actual current count, not the anticipated next commit or remote count.
  An initialized Git branch with no commits has count zero and the explicit source
  identity `unborn`; the first commit starts its own counter at zero.
- `BUILD_COUNT` starts at zero for each full Git commit and build component. It
  advances across debug/release profiles and output directories in this checkout.
  Amendments start a new full-SHA counter even if the commit count stays unchanged.
  Dirty edits and manual epoch changes do not reset an existing commit's counter.

This is a custom display scheme, not SemVer compatibility promises. Dependency,
ABI, Python distribution and Android numeric installation versions remain separate.
Generation zero is the explicit initial baseline; no historical rewrites are guessed.

## Build Pipeline

1. The native build adapter reads the tracked `APP-VERSION` and complete Git history.
2. `app_version.py` locks `.app-version/counter.lock`, reserves the next number for
   the full commit and component, then atomically persists the incremented state.
3. It freezes that reservation into generated compiler inputs and JSON metadata.
4. The compiler embeds the version in the executable. `--version` never counts a
   build, queries Git, reads mutable source metadata or changes running applications.

Cargo invokes the adapter only when its build script is invalidated. A cached,
no-op Cargo build therefore does not consume another reservation. CMake's build
target deliberately reserves and relinks on each build invocation. Only its small
generated version object recompiles; the stable public header does not invalidate
application source files. Failed attempts
after reservation consume their number rather than risk reusing an artifact ID.
Different Cargo packages or build-system components have independent counters;
multiple executables in one Cargo package share its reservation.

Counter files must not be deleted to reset numbering. They are local to this
checkout and cannot reconstruct unknown builds on another machine. Corrupt state,
shallow histories, missing Git history and fork repositories fail explicitly rather
than silently producing misleading zeros. Preserve `.app-version/` when moving an
existing checkout; vendored fork repositories keep their separate version scheme.
Projects which require generated data under `out/` can set
`{"state_directory":"out/.app-version"}` in `build-support/app-version.json`.
That path must remain inside the checkout. Preserve that directory instead.
The display version is not an exact source identifier: an amendment can preserve
the commit count and restart its build counter without changing the manual history
generation. Keep the full source commit and native build ID when distinguishing
those builds; uniqueness is scoped to a full commit and build component.

Inspect the next build without reserving it:

```bash
python3 build-support/app_version.py --root . --component rust:. --version-only
```

Use the adapter's exact component name; packages within a workspace are independent.

For Cargo, the adapter sets compiler environment versions without rewriting
`Cargo.toml`, dependency constraints or the lockfile's package release numbers.
CMake keeps `project(VERSION ...)` separate and embeds `.program_metadata` records.
Build JSON includes the full source commit and dirty status for exact provenance.

## Other Build Systems

- Node/Electron: the normal `npm run build` variants reserve a version before
  compilation. Electron freezes it into `dist/build-info.json`; the application
  launcher's `--version` reads that artifact without starting Electron or rebuilding.
  Web builds can copy their frozen record into a generated public asset.
- Go: use `python3 build-support/build_go.py`, `make build` where provided, or
  Wordelites' `just go-build`. Deployment scripts use that same entry point. It
  injects the version with linker flags and writes adjacent build-info JSON. Raw
  `go build` has no pre-build hook and deliberately reports unversioned development
  rather than inventing a build count. `--output`, `--ldflags` and `--trimpath` are
  supported; relative output paths are relative to the invoking working directory.
- Python: the normal PEP 517 build invokes the original Hatch/setuptools backend,
  then embeds `_app_version.json` in the wheel and updates every wheel RECORD hash.
  Distribution release versions remain valid PEP 440 versions; the application
  reports the separate `0.HISTORY.COMMITS-BUILD` version. Source distributions carry
  the frozen identity and backend so their subsequent wheel build does not need Git
  or reserve a second number for the same source-build pipeline. Editable builds
  freeze a local stamp as well. Running or importing the package never invokes Git.
- Android: configuration reads the next version without reserving it and supplies
  the default and each APK variant's `versionName`. A `preBuild` dependency reserves
  and verifies that exact identity before compilation, never during IDE sync or
  `gradle help`. Configuration-cache inputs track the non-mutating query. Concurrent
  builds or source changes which invalidate the preview fail rather than stamp
  an APK with an unreserved or reused identity. Numeric
  `versionCode` and signing settings are unchanged. A working Android SDK and the
  project's original Gradle/Android plugin dependencies are still required.

Native CMake targets define `APP_VERSION_BUILD` for their version handlers. Shared
C++ files compile without generated desktop headers when reused by Android or
standalone library/test builds. No system libraries or running applications are
updated or restarted by these adapters.

## Verification

The canonical `version` checkout contains the focused regression suite:

```bash
UV_CACHE_DIR=/data/.cache/uv uv sync --dev
UV_CACHE_DIR=/data/.cache/uv uv run python -m unittest discover -s tests -p test_app_version.py -v
```

It exercises actual Cargo/CMake/Go release artifacts and Python packages, cached
builds, local dependency edits, Node stamps, full-SHA resets, zero-commit branches,
concurrent allocation, corruption refusal, generated-directory policy, and commit
counts beyond 99. Vendored consumers do not need the test-only Python dependencies.

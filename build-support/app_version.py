#!/usr/bin/env python3
"""Freeze 0.history.commits-build into an application at build time."""

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        timeout=30,
        check=True,
    )
    text = result.stdout.decode("utf-8", errors="surrogateescape")
    return text if "-z" in args else text.strip()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def read_state(path):
    if not path.exists():
        return {"schema": 1, "commits": {}}
    state = json.loads(path.read_text())
    if not isinstance(state, dict) or type(state.get("schema")) is not int or state["schema"] != 1:
        raise ValueError("unsupported build-counter state; refusing to reset it")
    commits = state.get("commits")
    if not isinstance(commits, dict):
        raise ValueError("invalid build-counter commits")
    for commit, components in commits.items():
        if commit != "unborn" and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise ValueError("invalid build-counter commit")
        if not isinstance(components, dict):
            raise ValueError("invalid build-counter components")
        for component, counter in components.items():
            if not component or type(counter) is not int or counter < 0:
                raise ValueError("invalid build counter; refusing to reuse a number")
    return state


def current_commit(root):
    try:
        return git(root, "rev-parse", "--verify", "HEAD")
    except subprocess.CalledProcessError:
        ref = git(root, "symbolic-ref", "--quiet", "HEAD")
        result = subprocess.run(
            ["git", "-C", str(root), "show-ref", "--verify", "--quiet", ref],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 1:
            return "unborn"
        raise ValueError("Git HEAD is invalid, rather than an uncommitted branch")


def identity(root):
    if (root / "FORK-BASE").exists():
        raise ValueError("fork repositories must retain their fork versioning scheme")
    epoch = (root / "APP-VERSION").read_text().strip()
    if not re.fullmatch(r"0\.(0|[1-9][0-9]*)", epoch):
        raise ValueError("APP-VERSION must contain 0.HISTORY_GENERATION, without leading zeros")
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("--root must identify the application Git repository")
    if git(root, "rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("a complete Git history is required to count commits")
    commit = current_commit(root)
    count = 0 if commit == "unborn" else int(git(root, "rev-list", "--count", commit))
    return {
        "schema": 1,
        "epoch": epoch,
        "history_generation": int(epoch.split(".")[1]),
        "commit": commit,
        "commit_count": count,
        "dirty": bool(git(root, "status", "--porcelain=v1", "--untracked-files=normal")),
    }


def state_directory(root):
    config = root / "build-support/app-version.json"
    if not config.is_file():
        return root / ".app-version"
    configuration = json.loads(config.read_text())
    if not isinstance(configuration, dict):
        raise ValueError("app-version.json must contain a configuration object")
    directory = configuration.get("state_directory", ".app-version")
    if not isinstance(directory, str):
        raise ValueError("state_directory must be a relative directory")
    relative = Path(directory)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0] == ".git"
    ):
        raise ValueError("state_directory must remain inside the application checkout")
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("state_directory resolves outside the application checkout")
    return result


def version_record(root, component, allocate=False):
    if not component or "\n" in component or "\r" in component:
        raise ValueError("a nonempty single-line build component is required")
    record = identity(root)
    directory = state_directory(root)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "counter.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = directory / "state.json"
        state = read_state(path)
        counters = state["commits"].setdefault(record["commit"], {})
        build = counters.get(component, 0)
        if allocate:
            if current_commit(root) != record["commit"]:
                raise ValueError("Git HEAD changed while allocating the build version")
            counters[component] = build + 1
            atomic_write(path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return {
        **record,
        "component": component,
        "build": build,
        "version": f"{record['epoch']}.{record['commit_count']}-{build}",
    }


def cargo_inputs(root, manifest):
    paths = {root / "APP-VERSION", Path(__file__).resolve(), manifest, manifest.parent / "src"}
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for name in names.split("\0"):
        if name:
            path = root / name
            if path.is_file():
                paths.add(path)
    for name in ("HEAD", "index", "refs", "packed-refs"):
        path = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", name))
        if path.exists():
            paths.add(path)
    # Local path dependencies can relink this executable without editing its own source.
    import tomllib

    pending = [manifest, root / "Cargo.toml"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        data = tomllib.loads(path.read_text())
        paths.add(path)
        paths.add(path.parent / "src")
        for section in ("dependencies", "build-dependencies"):
            dependencies = data.get(section, {})
            if section == "dependencies":
                dependencies = {**dependencies, **data.get("workspace", {}).get(section, {})}
            for value in dependencies.values():
                if isinstance(value, dict) and "path" in value:
                    pending.append((path.parent / value["path"] / "Cargo.toml").resolve())
    return sorted(path for path in paths if path.exists())


def emit_cargo(root, manifest, record):
    for path in cargo_inputs(root, manifest):
        if "\n" in str(path) or "\r" in str(path):
            raise ValueError("Cargo build input contains a newline")
        print(f"cargo:rerun-if-changed={path}")
    print("cargo:rerun-if-env-changed=APP_VERSION_PYTHON")
    values = {
        "APP_VERSION": record["version"],
        "APP_VERSION_COMMIT": record["commit"],
        "CARGO_PKG_VERSION": record["version"],
        "CARGO_PKG_VERSION_MAJOR": "0",
        "CARGO_PKG_VERSION_MINOR": str(record["history_generation"]),
        "CARGO_PKG_VERSION_PATCH": str(record["commit_count"]),
        "CARGO_PKG_VERSION_PRE": str(record["build"]),
    }
    for key, value in values.items():
        print(f"cargo:rustc-env={key}={value}")


def write_header(directory, record):
    header = """#pragma once
namespace app_version {
const char *version();
void handle_cli(int argc, char *const *argv);
}
"""
    atomic_write(Path(directory) / "app_version.h", header)
    atomic_write(
        Path(directory) / "app_version_value.h",
        '#pragma once\n#define APP_VERSION "' + record["version"] + '"\n',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--allocate", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--cargo", type=Path, metavar="CARGO_TOML")
    parser.add_argument("--version-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    record = version_record(root, args.component, args.allocate)
    if args.output:
        atomic_write(args.output, json.dumps(record, indent=2, sort_keys=True) + "\n")
    if args.header:
        write_header(args.header, record)
    if args.cargo:
        emit_cargo(root, args.cargo.resolve(), record)
    elif args.version_only:
        print(record["version"])
    else:
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"app-version: {error}", file=sys.stderr)
        sys.exit(1)

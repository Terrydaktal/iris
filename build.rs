use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn command_output(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain=v1"])
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false)
}

fn rust_string(value: &str) -> String {
    format!("{:?}", value)
}

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo");
    let output_path = Path::new(&out_dir).join("iris_build_identity.rs");
    let revision = command_output("git", &["rev-parse", "HEAD"]);
    let rustc = command_output("rustc", &["--version"]);
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    let dirty = git_dirty();

    let source = format!(
        "pub const GIT_REVISION: &str = {};\n\
         pub const GIT_DIRTY: bool = {};\n\
         pub const RUSTC_VERSION: &str = {};\n\
         pub const TARGET: &str = {};\n\
         pub const PROFILE: &str = {};\n",
        rust_string(&revision),
        dirty,
        rust_string(&rustc),
        rust_string(&target),
        rust_string(&profile),
    );
    fs::write(output_path, source).expect("write Iris build identity");
}

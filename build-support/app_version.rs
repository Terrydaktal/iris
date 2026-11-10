use std::path::Path;
use std::process::Command;

pub fn emit(root: &str, component: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(root);
    let python = std::env::var_os("APP_VERSION_PYTHON").unwrap_or_else(|| "python3".into());
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(python)
        .arg(root.join("build-support/app_version.py"))
        .arg("--root")
        .arg(&root)
        .args(["--component", component, "--allocate", "--cargo"])
        .arg(&manifest)
        .arg("--output")
        .arg(
            Path::new(&std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"))
                .join("app-version.json"),
        )
        .output()
        .expect("run Python 3.11+ to generate the application version");
    assert!(
        output.status.success(),
        "application version generation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    print!(
        "{}",
        String::from_utf8(output.stdout).expect("UTF-8 Cargo version instructions")
    );
}

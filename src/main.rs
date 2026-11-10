mod app;

fn main() -> eframe::Result {
    mod app_version_runtime {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/build-support/app_version_runtime.rs"
        ));
    }
    app_version_runtime::handle_cli();
    app::run()
}

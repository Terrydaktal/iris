pub fn handle_cli() {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        println!("{} {}", env!("CARGO_BIN_NAME"), env!("APP_VERSION"));
        std::process::exit(0);
    }
}

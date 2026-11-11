fn main() {
    std::process::exit(cliff::cli::main(std::env::args().skip(1).collect()));
}

use std::num::NonZero;

fn main() {
    let mut cfg = ramsey::search::SearchConfig::default();
    let mut validate_only = false;
    let mut target_n = 39;
    let mut target_k = 11;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--test" | "--validate" => {
                validate_only = true;
                i += 1;
            }
            "--case" => {
                target_n = args
                    .get(i + 1)
                    .unwrap_or_else(|| usage_and_exit(2))
                    .parse()
                    .unwrap_or_else(|_| usage_and_exit(2));
                target_k = args
                    .get(i + 2)
                    .unwrap_or_else(|| usage_and_exit(2))
                    .parse()
                    .unwrap_or_else(|_| usage_and_exit(2));
                i += 3;
            }
            "--chains" | "--workers" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2));
                cfg.chains = v.parse().unwrap_or_else(|_| usage_and_exit(2));
                i += 2;
            }
            "--p" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2));
                cfg.edge_probability = v.parse().unwrap_or_else(|_| usage_and_exit(2));
                i += 2;
            }
            "--seed" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2));
                cfg.seed = Some(v.parse().unwrap_or_else(|_| usage_and_exit(2)));
                i += 2;
            }
            "--help" | "-h" => usage_and_exit(0),
            _ => usage_and_exit(2),
        }
    }

    if validate_only {
        match ramsey::validate::validate_known_graphs() {
            Ok(()) => {
                println!("Validation OK: bundled witness graphs are correct.");
                return;
            }
            Err(e) => {
                eprintln!("Validation FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    // Macro-based dispatcher for const generic N.
    // This allows absolute maximum performance for any N <= 64.
    macro_rules! dispatch {
        ($($n:literal),*) => {
            match target_n {
                $($n => ramsey::search::run_search::<$n>(&cfg, target_k),)*
                _ => {
                    eprintln!("Error: N={target_n} is not supported (max 64).");
                    std::process::exit(2);
                }
            }
        }
    }

    dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    );
}

fn usage_and_exit(code: i32) -> ! {
    let default_chains = std::thread::available_parallelism()
        .map(NonZero::get)
        .map(|n| n * 2)
        .unwrap_or(200);

    eprintln!(
        "Ultra-Optimized Ramsey Search Engine\n\
         Usage: ramsey [OPTIONS]\n\n\
         Options:\n  \
         --case N K              Target graph order and independent set size (default: 39 11)\n  \
         --workers/--chains N     Number of parallel search chains (default: {default_chains})\n  \
         --p P                    Initial edge probability (default: 0.18)\n  \
         --seed SEED              Deterministic base seed for reproducibility\n  \
         --test/--validate         Validate bundled witness graphs (fast, deterministic)\n  \
         --help                   Show this help message\n"
    );
    std::process::exit(code)
}

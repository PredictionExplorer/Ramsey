use std::num::NonZero;
use ramsey::search::{SearchConfig, ForbiddenType};

fn main() {
    let mut cfg = SearchConfig::default();
    let mut validate_only = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--test" | "--validate" => {
                validate_only = true;
                i += 1;
            }
            "-n" => {
                cfg.n_target = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2)).parse().unwrap_or_else(|_| usage_and_exit(2));
                i += 2;
            }
            "-k" => {
                cfg.k_target = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2)).parse().unwrap_or_else(|_| usage_and_exit(2));
                i += 2;
            }
            "-c" => {
                cfg.c_target = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2)).parse().unwrap_or_else(|_| usage_and_exit(2));
                i += 2;
            }
            "--clique" => {
                cfg.forbidden_type = ForbiddenType::Clique;
                i += 1;
            }
            "--cycle" => {
                cfg.forbidden_type = ForbiddenType::Cycle;
                i += 1;
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
            "--resume" => {
                cfg.resume_path = Some(args.get(i + 1).unwrap_or_else(|| usage_and_exit(2)).clone());
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
    macro_rules! dispatch {
        ($($n:literal),*) => {
            match cfg.n_target {
                $($n => ramsey::search::run_search::<$n>(&cfg),)*
                _ => {
                    eprintln!("Error: N={} is not supported (max 64).", cfg.n_target);
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
         -n N                    Target graph order (default: 39)\n  \
         -k K                    Target independent set size to avoid (default: 11)\n  \
         -c C                    Target cycle/clique size to avoid (default: 4)\n  \
         --clique                Search for R(K_c, K_k) instead of R(C_c, K_k)\n  \
         --cycle                 Search for R(C_c, K_k) (default)\n  \
         --workers/--chains N     Number of parallel search chains (default: {default_chains})\n  \
         --p P                    Initial edge probability (default: 0.18)\n  \
         --seed SEED              Deterministic base seed for reproducibility\n  \
         --test/--validate         Validate bundled witness graphs (fast, deterministic)\n  \
         --help                   Show this help message\n"
    );
    std::process::exit(code)
}

fn main() {
    let mut cfg = ramsey::search::SearchConfig::default();
    let mut validate_only = false;
    let mut case: Option<(usize, usize)> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--test" | "--validate" => {
                validate_only = true;
                i += 1;
            }
            "--case" => {
                let n = args.get(i + 1).unwrap_or_else(|| usage_and_exit(2));
                let k = args.get(i + 2).unwrap_or_else(|| usage_and_exit(2));
                case = Some((
                    n.parse().unwrap_or_else(|_| usage_and_exit(2)),
                    k.parse().unwrap_or_else(|_| usage_and_exit(2)),
                ));
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

    match case {
        None | Some((39, 11)) => ramsey::search::run_record_search(&cfg),
        Some((6, 3)) => ramsey::search::run_search::<6, 3>(&cfg),
        Some((9, 4)) => ramsey::search::run_search::<9, 4>(&cfg),
        Some((13, 5)) => ramsey::search::run_search::<13, 5>(&cfg),
        Some((n, k)) => {
            eprintln!("Unsupported --case {n} {k}. Supported cases: 6 3, 9 4, 13 5, 39 11.");
            std::process::exit(2);
        }
    }
}

fn usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage:\n  ramsey [--case N K] [--workers N] [--p P] [--seed SEED]\n  ramsey --test\n\nOptions:\n  --case N K              Run a compiled-in case (supported: 6 3, 9 4, 13 5, 39 11)\n  --workers/--chains N     Number of independent search chains (default: auto-detect)\n  --p P                    Initial edge probability (default: 0.18)\n  --seed SEED              Deterministic base seed (optional)\n  --test/--validate         Validate bundled witness graphs (fast, deterministic)\n"
    );
    std::process::exit(code)
}

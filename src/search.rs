//! Parallel stochastic search driver (LAHC + SA hybrid).

use crate::graph::RamseyState;
use crate::iset::IndependentSetOracle;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ============================================================================
// Configuration
// ============================================================================

/// Maximum LAHC history size (compile-time to use stack allocation).
const MAX_LAHC_SIZE: usize = 2048;

/// Type of forbidden subgraph to avoid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForbiddenType {
    /// Avoid cycles of length C.
    Cycle,
    /// Avoid cliques of size C.
    Clique,
}

/// Search configuration parameters.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of independent search chains to run.
    pub chains: usize,
    /// Target graph order.
    pub n_target: usize,
    /// Target independent set size to avoid.
    pub k_target: usize,
    /// Target cycle/clique size to avoid.
    pub c_target: usize,
    /// Type of forbidden subgraph.
    pub forbidden_type: ForbiddenType,
    /// Initial Erdős–Rényi edge probability.
    pub edge_probability: f64,
    /// Weight for the C4 penalty.
    pub c4_weight: usize,
    /// Penalty added for IS violation.
    pub independent_violation_penalty: usize,
    /// LAHC history buffer length.
    pub lahc_size: usize,
    /// Starting temperature.
    pub temp_start: f64,
    /// Cooling rate.
    pub cooling_rate: f64,
    /// Temperature update period.
    pub temp_update_every: u64,
    /// If temperature drops below this threshold, it is reset to `reheat_temp`.
    pub reheat_threshold: f64,
    /// Temperature value used when reheating.
    pub reheat_temp: f64,
    /// Probability of guided move.
    pub c4_guided_probability: f64,
    /// Attempts when sampling a C4 witness.
    pub c4_probe_tries: usize,
    /// Number of candidate pairs to try.
    pub indep_pair_samples: usize,
    /// Progress report period.
    pub report_every: u64,
    /// Optional deterministic base seed.
    pub seed: Option<u64>,
    /// Optional path to a starting graph.
    pub resume_path: Option<String>,
    /// Number of iterations without improvement before performing a "Kick".
    pub kick_threshold: u64,
    /// Number of edges to flip during a "Kick".
    pub kick_strength: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        let chains = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .map(|n| n * 2)
            .unwrap_or(200);

        Self {
            chains,
            n_target: 39,
            k_target: 11,
            c_target: 4,
            forbidden_type: ForbiddenType::Cycle,
            edge_probability: 0.18,
            c4_weight: 10,
            independent_violation_penalty: 1,
            lahc_size: 500,
            temp_start: 4.0,
            cooling_rate: 0.999_999_5,
            temp_update_every: 1_000,
            reheat_threshold: 0.005,
            reheat_temp: 1.0,
            c4_guided_probability: 0.85,
            c4_probe_tries: 64,
            indep_pair_samples: 24,
            report_every: 1_000_000,
            seed: None,
            resume_path: None,
            kick_threshold: 5_000_000,
            kick_strength: 5,
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Runs the record search for `N=39, K=11`.
pub fn run_record_search(cfg: &SearchConfig) {
    run_search::<39>(cfg);
}

/// Runs a parallel search for a graph on `N` vertices.
pub fn run_search<const N: usize>(cfg: &SearchConfig) {
    let mode_str = match cfg.forbidden_type {
        ForbiddenType::Cycle => format!("C{}", cfg.c_target),
        ForbiddenType::Clique => format!("K{}", cfg.c_target),
    };
    println!("--------------------------------------------------");
    println!("Ramsey Search: target = no {} and no IS(K={}) on N={}", mode_str, cfg.k_target, N);
    println!(
        "Chains: {} (Logical Cores: {}) | p={:.3}",
        cfg.chains,
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(0),
        cfg.edge_probability
    );
    println!(
        "Metaheuristic: LAHC(size={}) + SA(T0={:.2})",
        cfg.lahc_size.min(MAX_LAHC_SIZE),
        cfg.temp_start
    );
    println!("--------------------------------------------------");

    let base_seed = cfg.seed.unwrap_or_else(random_u64);
    let found_flag = AtomicBool::new(false);
    let global_best_energy = AtomicUsize::new(usize::MAX);

    let log_filename = format!("search_log_n{N}_k{}.csv", cfg.k_target);
    if let Ok(mut log_file) = std::fs::File::create(&log_filename) {
        let _ = writeln!(log_file, "timestamp_ms,iterations,temperature,best_energy");
    }

    (0..cfg.chains).into_par_iter().for_each(|worker_id| {
        solve_worker::<N>(worker_id, base_seed, cfg, &found_flag, &global_best_energy);
    });
}

// ============================================================================
// Worker
// ============================================================================

fn solve_worker<const N: usize>(
    worker_id: usize,
    base_seed: u64,
    cfg: &SearchConfig,
    found_flag: &AtomicBool,
    global_best_energy: &AtomicUsize,
) {
    let mut rng = SmallRng::seed_from_u64(splitmix64(base_seed ^ (worker_id as u64)));
    let mut state = initialize_state::<N>(worker_id, &mut rng, cfg);
    let mut oracle = IndependentSetOracle::<N>::new();
    let mut scratch_set = Vec::<usize>::with_capacity(cfg.k_target.max(16));

    let mut temp = cfg.temp_start;
    let mut iterations: u64 = 0;
    let mut iters_since_best: u64 = 0;
    let mut last_csv_log_best = usize::MAX;
    let start_time = std::time::Instant::now();

    let lahc_len = cfg.lahc_size.clamp(1, MAX_LAHC_SIZE);
    let mut lahc_history = [0usize; MAX_LAHC_SIZE];
    let initial_energy = evaluate::<N>(&state, &mut oracle, &mut scratch_set, cfg).energy;
    global_best_energy.fetch_min(initial_energy, Ordering::Relaxed);

    for slot in lahc_history.iter_mut().take(lahc_len) {
        *slot = initial_energy;
    }
    let mut current_energy = initial_energy;
    let mut local_best_energy = initial_energy;
    let mut lahc_idx = 0usize;

    while !found_flag.load(Ordering::Relaxed) {
        iterations += 1;
        iters_since_best += 1;

        if iters_since_best >= cfg.kick_threshold {
            for _ in 0..cfg.kick_strength {
                let (u, v) = random_pair::<N, _>(&mut rng);
                state.flip_edge(u, v);
            }
            current_energy = evaluate::<N>(&state, &mut oracle, &mut scratch_set, cfg).energy;
            iters_since_best = 0;
            continue;
        }

        let (u, v) = propose_move::<N, _>(&mut state, &mut oracle, &mut scratch_set, &mut rng, cfg);
        state.flip_edge(u, v);
        let eval = evaluate::<N>(&state, &mut oracle, &mut scratch_set, cfg);

        if eval.is_solution {
            handle_success::<N>(worker_id, &state, cfg, found_flag);
            return;
        }

        if eval.energy < local_best_energy {
            local_best_energy = eval.energy;
            iters_since_best = 0;
            let old = global_best_energy.fetch_min(eval.energy, Ordering::Relaxed);
            if eval.energy < old {
                let _ = state.save_to_file(format!("best_checkpoint_n{N}_k{}.txt", cfg.k_target));
            }
        }

        if accept_move(eval.energy, lahc_history[lahc_idx], current_energy, temp, &mut rng) {
            current_energy = eval.energy;
        } else {
            state.flip_edge(u, v);
        }

        lahc_history[lahc_idx] = current_energy;
        lahc_idx = (lahc_idx + 1) % lahc_len;

        if iterations.is_multiple_of(cfg.temp_update_every) {
            temp *= cfg.cooling_rate;
            if temp < cfg.reheat_threshold { temp = cfg.reheat_temp; }
            if worker_id == 0 && iterations.is_multiple_of(cfg.report_every) {
                last_csv_log_best = report_progress::<N>(iterations, temp, current_energy, &state, global_best_energy, last_csv_log_best, start_time, cfg);
            }
        }
    }
}

fn initialize_state<const N: usize>(worker_id: usize, rng: &mut SmallRng, cfg: &SearchConfig) -> RamseyState<N> {
    if let Some(path) = &cfg.resume_path {
        match RamseyState::<N>::load_from_file(path) {
            Ok(s) => s,
            Err(e) => {
                if worker_id == 0 { eprintln!("Warning: Failed to load resume file: {e}. Starting fresh."); }
                RamseyState::<N>::new_random(rng, cfg.edge_probability)
            }
        }
    } else {
        RamseyState::<N>::new_random(rng, cfg.edge_probability)
    }
}

fn handle_success<const N: usize>(worker_id: usize, state: &RamseyState<N>, cfg: &SearchConfig, found_flag: &AtomicBool) {
    if found_flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
        let mode = if cfg.forbidden_type == ForbiddenType::Cycle { format!("c{}", cfg.c_target) } else { format!("k{}", cfg.c_target) };
        println!("\n[Worker {worker_id}] SUCCESS: found valid graph (N={N}, {mode} IS={}). Saving...", cfg.k_target);
        let filename = format!("graph_n{N}_{mode}_k{}.txt", cfg.k_target);
        let _ = state.save_to_file(filename);
    }
}

#[inline]
fn accept_move(new_e: usize, threshold: usize, current_e: usize, temp: f64, rng: &mut SmallRng) -> bool {
    if new_e <= threshold || new_e <= current_e {
        true
    } else {
        let prob = (-(new_e as f64 - current_e as f64) / temp).exp();
        rng.random_bool(prob.min(1.0))
    }
}

#[allow(clippy::too_many_arguments)]
fn report_progress<const N: usize>(
    iters: u64, temp: f64, current_e: usize, state: &RamseyState<N>,
    global_best: &AtomicUsize, last_csv: usize, start: std::time::Instant, cfg: &SearchConfig
) -> usize {
    let best = global_best.load(Ordering::Relaxed);
    print!("\rIter: {iters} | T: {temp:.4} | Best E: {best} | Worker0 E: {current_e} | C4: {}    ", state.c4_count());
    let _ = std::io::stdout().flush();

    let heartbeat = iters.is_multiple_of(cfg.report_every * 100);
    if best < last_csv || heartbeat {
        let log_filename = format!("search_log_n{N}_k{}.csv", cfg.k_target);
        if let Ok(mut log_file) = std::fs::OpenOptions::new().append(true).open(&log_filename) {
            let _ = writeln!(log_file, "{},{iters},{temp:.6},{best}", start.elapsed().as_millis());
        }
        return best;
    }
    last_csv
}

// ============================================================================
// Evaluation
// ============================================================================

#[derive(Clone, Debug)]
struct Eval {
    energy: usize,
    is_solution: bool,
}

/// Staged evaluation:
/// - Uses the incremental C4 score if target is C4.
/// - Otherwise uses the oracle for general cliques/cycles.
#[inline]
fn evaluate<const N: usize>(
    state: &RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    scratch_set: &mut Vec<usize>,
    cfg: &SearchConfig,
) -> Eval {
    // 1. Calculate Forbidden Subgraph Energy
    let forbidden_energy = match (cfg.forbidden_type, cfg.c_target) {
        (ForbiddenType::Cycle, 4) => state.c4_score_twice() * cfg.c4_weight,
        (ForbiddenType::Clique, c) => oracle.count_cliques_of_size(state.adj(), c, 100) * cfg.c4_weight,
        (ForbiddenType::Cycle, c) => {
            if c == 3 {
                oracle.count_cliques_of_size(state.adj(), 3, 100) * cfg.c4_weight
            } else {
                panic!("Unsupported cycle length {c}. Only C3 and C4 are currently implemented.");
            }
        }
    };

    if forbidden_energy != 0 {
        let violates = state.greedy_find_independent_set_of_size(cfg.k_target, scratch_set);
        return Eval {
            energy: forbidden_energy + if violates { cfg.independent_violation_penalty } else { 0 },
            is_solution: false,
        };
    }

    // 2. Calculate Independent Set Energy (Gradient)
    let is_count = oracle.count_independent_sets_of_size(state.adj(), cfg.k_target, 100);
    
    Eval {
        energy: is_count,
        is_solution: is_count == 0,
    }
}

// ============================================================================
// Move proposal
// ============================================================================

#[inline]
fn propose_move<const N: usize, R: Rng>(
    state: &mut RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    scratch_set: &mut Vec<usize>,
    rng: &mut R,
    cfg: &SearchConfig,
) -> (usize, usize) {
    let current_eval = evaluate::<N>(state, oracle, scratch_set, cfg);

    if current_eval.energy >= cfg.c4_weight {
        // We have forbidden subgraphs. Try to remove one.
        if cfg.forbidden_type == ForbiddenType::Cycle && cfg.c_target == 4 {
            if rng.random_bool(cfg.c4_guided_probability)
                && let Some(edge) = state.sample_c4_edge_to_remove(rng, cfg.c4_probe_tries)
            {
                return edge;
            }
        } else if cfg.forbidden_type == ForbiddenType::Clique {
            // Heuristic: remove an edge from a found clique
            let mut clique = Vec::new();
            if oracle.find_clique_of_size(state.adj(), cfg.c_target, &mut clique) {
                let i = rng.random_range(0..clique.len());
                let mut j = rng.random_range(0..clique.len());
                while i == j { j = rng.random_range(0..clique.len()); }
                return (clique[i], clique[j]);
            }
        }
        return random_pair::<N, _>(rng);
    }

    // Graph is clean of forbidden subgraphs. Try to break independent sets.
    if state.greedy_find_independent_set_of_size(cfg.k_target, scratch_set) {
        return best_edge_to_add_within_set(state, scratch_set, rng, cfg.indep_pair_samples);
    }

    scratch_set.clear();
    if oracle.find_independent_set_of_size(state.adj(), cfg.k_target, scratch_set) {
        return best_edge_to_add_within_set(state, scratch_set, rng, cfg.indep_pair_samples);
    }

    random_pair::<N, _>(rng)
}

#[inline]
fn best_edge_to_add_within_set<const N: usize, R: Rng>(
    state: &mut RamseyState<N>,
    set: &[usize],
    rng: &mut R,
    samples: usize,
) -> (usize, usize) {
    debug_assert!(state.c4_score_twice() == 0, "caller expects C4-free state");

    if set.len() < 2 {
        return random_pair::<N, _>(rng);
    }

    let mut best = (set[0], set[1]);
    let mut best_c4 = usize::MAX;

    let tries = samples.clamp(1, 256);
    for _ in 0..tries {
        let i = rng.random_range(0..set.len());
        let mut j = rng.random_range(0..set.len());
        while j == i {
            j = rng.random_range(0..set.len());
        }
        let u = set[i];
        let v = set[j];

        debug_assert!(!state.has_edge(u, v), "set is expected to be independent");

        state.flip_edge(u, v); // add
        let c4 = state.c4_score_twice();
        state.flip_edge(u, v); // revert

        if c4 < best_c4 {
            best_c4 = c4;
            best = (u, v);
            if c4 == 0 {
                break;
            }
        }
    }

    best
}

#[inline]
fn random_pair<const N: usize, R: Rng>(rng: &mut R) -> (usize, usize) {
    let u = rng.random_range(0..N);
    let mut v = rng.random_range(0..N);
    while v == u {
        v = rng.random_range(0..N);
    }
    (u, v)
}

fn random_u64() -> u64 {
    rand::random::<u64>()
}

/// SplitMix64 mixer for deriving per-worker seeds from a base seed.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic() {
        assert_eq!(splitmix64(0), splitmix64(0));
        assert_eq!(splitmix64(12345), splitmix64(12345));
        assert_ne!(splitmix64(0), splitmix64(1));
    }

    #[test]
    fn search_config_default_is_valid() {
        let cfg = SearchConfig::default();
        assert!(cfg.chains > 0);
        assert!((0.0..=1.0).contains(&cfg.edge_probability));
        assert!(cfg.lahc_size > 0);
        assert!(cfg.temp_start > 0.0);
        assert!((0.0..1.0).contains(&cfg.cooling_rate));
        assert!(cfg.temp_update_every > 0);
    }

    #[test]
    fn random_pair_is_valid() {
        let mut rng = SmallRng::seed_from_u64(0x1234);
        for _ in 0..1000 {
            let (u, v) = random_pair::<20, _>(&mut rng);
            assert!(u < 20);
            assert!(v < 20);
            assert_ne!(u, v);
        }
    }

    #[test]
    fn search_is_deterministic() {
        // Run two identical searches and verify they reach the same energy state.
        fn run_mock_search(seed: u64) -> usize {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = RamseyState::<10>::new_random(&mut rng, 0.3);
            let mut oracle = IndependentSetOracle::<10>::new();
            let mut scratch = Vec::new();
            let cfg = SearchConfig::default();
            
            for _ in 0..100 {
                let (u, v) = propose_move::<10, _>(&mut state, &mut oracle, &mut scratch, &mut rng, &cfg);
                state.flip_edge(u, v);
                let _eval = evaluate::<10>(&state, &mut oracle, &mut scratch, &cfg);
                // We don't accept/revert here to keep it simple, just checking move sequence
            }
            state.c4_score_twice()
        }

        let seed = 0xDEADC0DE;
        let res1 = run_mock_search(seed);
        let res2 = run_mock_search(seed);
        assert_eq!(res1, res2, "Search with same seed must be deterministic");
    }

    #[test]
    fn worker_seeding_is_independent() {
        let base_seed = 0x1337;
        let mut rng0 = SmallRng::seed_from_u64(splitmix64(base_seed ^ 0));
        let mut rng1 = SmallRng::seed_from_u64(splitmix64(base_seed ^ 1));
        
        let val0: u64 = rng0.random();
        let val1: u64 = rng1.random();
        assert_ne!(val0, val1, "Workers must have different RNG sequences");
    }

    #[test]
    fn global_best_tracking_works() {
        let best = AtomicUsize::new(usize::MAX);
        best.fetch_min(100, Ordering::Relaxed);
        best.fetch_min(50, Ordering::Relaxed);
        best.fetch_min(75, Ordering::Relaxed);
        assert_eq!(best.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn checkpoint_saving_is_atomic_and_valid() {
        use std::fs;
        let temp_file = "test_checkpoint_integrity.txt";
        let state = RamseyState::<8>::new_random(&mut SmallRng::seed_from_u64(42), 0.3);
        
        // Save
        state.save_to_file(temp_file).expect("Failed to save checkpoint");
        
        // Load and Verify
        let loaded = RamseyState::<8>::load_from_file(temp_file).expect("Failed to load checkpoint");
        assert_eq!(state.adj(), loaded.adj());
        assert_eq!(state.c4_score_twice(), loaded.c4_score_twice());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_accept_move_metropolis() {
        let mut rng = SmallRng::seed_from_u64(42);
        
        // Better energy should always be accepted
        assert!(accept_move(50, 100, 100, 1.0, &mut rng));
        assert!(accept_move(100, 100, 100, 1.0, &mut rng));
        
        // Worse energy at T=0 should never be accepted (approx)
        let mut accepted_worse = 0;
        for _ in 0..1000 {
            if accept_move(110, 100, 100, 0.0001, &mut rng) {
                accepted_worse += 1;
            }
        }
        assert!(accepted_worse < 5); // Allow for extreme RNG luck, but should be near 0
        
        // Worse energy at high T should be accepted sometimes
        let mut accepted_high_t = 0;
        for _ in 0..1000 {
            if accept_move(101, 100, 100, 100.0, &mut rng) {
                accepted_high_t += 1;
            }
        }
        assert!(accepted_high_t > 900); // Should be very likely
    }

    #[test]
    fn test_ils_kick_behavior() {
        let mut rng = SmallRng::seed_from_u64(123);
        let mut state = RamseyState::<10>::new_random(&mut rng, 0.2);
        let orig_adj = *state.adj();
        
        // Perform a mock kick
        let strength = 5;
        for _ in 0..strength {
            let (u, v) = random_pair::<10, _>(&mut rng);
            state.flip_edge(u, v);
        }
        
        assert_ne!(orig_adj, *state.adj(), "Kick must modify the graph state");
    }
}

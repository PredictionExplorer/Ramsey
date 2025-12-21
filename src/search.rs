//! Parallel stochastic search driver (LAHC + SA hybrid).

use crate::graph::RamseyState;
use crate::iset::IndependentSetOracle;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Configuration
// ============================================================================

/// Maximum LAHC history size (compile-time to use stack allocation).
const MAX_LAHC_SIZE: usize = 2048;

/// Search configuration parameters.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of independent search chains to run (Rayon will schedule them over its thread pool).
    pub chains: usize,
    /// Initial Erdős–Rényi edge probability.
    pub edge_probability: f64,
    /// Weight for the C4 penalty (uses the "twice-count" score).
    pub c4_weight: usize,
    /// Penalty added when we have a certified independent-set violation (heuristic or exact).
    pub independent_violation_penalty: usize,
    /// LAHC history buffer length (capped at `MAX_LAHC_SIZE`).
    pub lahc_size: usize,
    /// Starting temperature for the SA component.
    pub temp_start: f64,
    /// Multiplicative cooling rate applied every `temp_update_every` iterations.
    pub cooling_rate: f64,
    /// Temperature update period.
    pub temp_update_every: u64,
    /// If temperature drops below this threshold, it is reset to `reheat_temp`.
    pub reheat_threshold: f64,
    /// Temperature value used when reheating.
    pub reheat_temp: f64,
    /// Probability of using a C4-guided move when C4s exist.
    pub c4_guided_probability: f64,
    /// Attempts when sampling a C4 witness.
    pub c4_probe_tries: usize,
    /// Number of candidate pairs to try when adding an edge inside a violating independent set.
    pub indep_pair_samples: usize,
    /// Progress report period (worker 0 only).
    pub report_every: u64,
    /// Optional deterministic base seed. If `None`, a random seed is used.
    pub seed: Option<u64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        // Automatically detect available logical cores.
        // We default to cores * 2 to keep the CPU saturated and explore more paths.
        let chains = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .map(|n| n * 2)
            .unwrap_or(200);

        Self {
            chains,
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
            report_every: 20_000,
            seed: None,
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Runs the record search for `N=39, K=11`.
pub fn run_record_search(cfg: &SearchConfig) {
    run_search::<39, 11>(cfg);
}

/// Runs a parallel search for a graph on `N` vertices with:
/// - no `C4` cycles
/// - no independent set of size `K`
pub fn run_search<const N: usize, const K: usize>(cfg: &SearchConfig) {
    println!("--------------------------------------------------");
    println!("Ramsey Search: target = no C4 and no IS(K={K}) on N={N}");
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

    (0..cfg.chains).into_par_iter().for_each(|worker_id| {
        solve_worker::<N, K>(worker_id, base_seed, cfg, &found_flag);
    });
}

// ============================================================================
// Worker
// ============================================================================

fn solve_worker<const N: usize, const K: usize>(
    worker_id: usize,
    base_seed: u64,
    cfg: &SearchConfig,
    found_flag: &AtomicBool,
) {
    let mut rng = XorShiftRng::seed_from_u64(splitmix64(base_seed ^ (worker_id as u64)));
    let mut state = RamseyState::<N>::new_random(&mut rng, cfg.edge_probability);
    let mut oracle = IndependentSetOracle::<N>::new();
    let mut scratch_set = Vec::<usize>::with_capacity(K.max(16));

    let mut temp = cfg.temp_start;
    let mut iterations: u64 = 0;

    let lahc_len = cfg.lahc_size.clamp(1, MAX_LAHC_SIZE);
    let mut lahc_history = [0usize; MAX_LAHC_SIZE];
    let initial_energy = evaluate::<N, K>(&state, &mut oracle, &mut scratch_set, cfg).energy;
    for slot in lahc_history.iter_mut().take(lahc_len) {
        *slot = initial_energy;
    }
    let mut current_energy = initial_energy;
    let mut lahc_idx = 0usize;

    while !found_flag.load(Ordering::Relaxed) {
        iterations += 1;

        let (u, v) =
            propose_move::<N, K, _>(&mut state, &mut oracle, &mut scratch_set, &mut rng, cfg);

        state.flip_edge(u, v);
        let eval = evaluate::<N, K>(&state, &mut oracle, &mut scratch_set, cfg);

        if eval.is_solution {
            if found_flag
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                println!(
                    "\n[Worker {worker_id}] SUCCESS: found valid graph (N={N}, K={K}). Saving..."
                );
                let filename = format!("graph_n{N}_k{K}.txt");
                if let Err(e) = state.save_to_file(&filename) {
                    eprintln!("[Worker {worker_id}] ERROR: failed to save {filename}: {e}");
                } else {
                    println!("[Worker {worker_id}] Wrote {filename}");
                }
            }
            return;
        }

        let new_energy = eval.energy;
        let lahc_threshold = lahc_history[lahc_idx];
        let delta = (new_energy as f64) - (current_energy as f64);

        let accept = if new_energy <= lahc_threshold || delta <= 0.0 {
            true
        } else {
            let prob = (-delta / temp).exp().min(1.0);
            rng.random_bool(prob)
        };

        if accept {
            current_energy = new_energy;
        } else {
            state.flip_edge(u, v); // revert
        }

        lahc_history[lahc_idx] = current_energy;
        lahc_idx += 1;
        if lahc_idx >= lahc_len {
            lahc_idx = 0;
        }

        if iterations % cfg.temp_update_every == 0 {
            temp *= cfg.cooling_rate;
            if temp < cfg.reheat_threshold {
                temp = cfg.reheat_temp;
            }

            if worker_id == 0 && iterations % cfg.report_every == 0 {
                print!(
                    "\rIter: {iterations} | T: {temp:.4} | E: {current_energy} | C4: {}    ",
                    state.c4_count()
                );
                let _ = std::io::stdout().flush();
            }
        }
    }
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
/// - Always uses the exact incremental C4 score.
/// - Uses a greedy independent-set lower bound as a cheap certificate of violation.
/// - Uses an exact oracle **only when C4==0 and the greedy check didn't find a violation**.
#[inline]
fn evaluate<const N: usize, const K: usize>(
    state: &RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    scratch_set: &mut Vec<usize>,
    cfg: &SearchConfig,
) -> Eval {
    let c4_twice = state.c4_score_twice();

    if c4_twice != 0 {
        // Cheap guidance only: we avoid exact IS checks unless we are C4-free.
        let violates = state.greedy_find_independent_set_of_size(K, scratch_set);
        let energy = c4_twice.saturating_mul(cfg.c4_weight)
            + if violates {
                cfg.independent_violation_penalty
            } else {
                0
            };
        return Eval {
            energy,
            is_solution: false,
        };
    }

    // C4-free: now enforce the independent-set constraint.
    if state.greedy_find_independent_set_of_size(K, scratch_set) {
        return Eval {
            energy: cfg.independent_violation_penalty,
            is_solution: false,
        };
    }

    if oracle.has_independent_set_of_size(state.adj(), K) {
        return Eval {
            energy: cfg.independent_violation_penalty,
            is_solution: false,
        };
    }

    Eval {
        energy: 0,
        is_solution: true,
    }
}

// ============================================================================
// Move proposal
// ============================================================================

#[inline]
fn propose_move<const N: usize, const K: usize, R: Rng>(
    state: &mut RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    scratch_set: &mut Vec<usize>,
    rng: &mut R,
    cfg: &SearchConfig,
) -> (usize, usize) {
    let c4_twice = state.c4_score_twice();

    if c4_twice != 0 {
        if rng.random_bool(cfg.c4_guided_probability) {
            if let Some(edge) = state.sample_c4_edge_to_remove(rng, cfg.c4_probe_tries) {
                return edge;
            }
        }
        return random_pair::<N, _>(rng);
    }

    // C4-free: try to break independent sets by adding an edge inside a violating set.
    if state.greedy_find_independent_set_of_size(K, scratch_set) {
        return best_edge_to_add_within_set(state, scratch_set, rng, cfg.indep_pair_samples);
    }

    scratch_set.clear();
    if oracle.find_independent_set_of_size(state.adj(), K, scratch_set) {
        return best_edge_to_add_within_set(state, scratch_set, rng, cfg.indep_pair_samples);
    }

    // Looks feasible; evaluation will confirm. Keep exploring with a random perturbation.
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
        let mut rng = XorShiftRng::seed_from_u64(0x1234);
        for _ in 0..1000 {
            let (u, v) = random_pair::<20, _>(&mut rng);
            assert!(u < 20);
            assert!(v < 20);
            assert_ne!(u, v);
        }
    }
}

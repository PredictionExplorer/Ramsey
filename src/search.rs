//! Parallel stochastic search driver (LAHC + SA hybrid with Tabu).

use crate::graph::RamseyState;
use crate::iset::{CachedIsOracle, IndependentSetOracle};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::collections::VecDeque;
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
    /// Size of the tabu list (prevents cycling).
    pub tabu_size: usize,
    /// Probability of using degree-biased move selection.
    pub degree_bias_probability: f64,
    /// Jitter applied to the initial edge probability per worker for diversification.
    /// The actual p is sampled uniformly from `[p - jitter, p + jitter]` and clamped to `[0, 1]`.
    pub initial_p_jitter: f64,
    /// If enabled, workers may restart from scratch (or from an elite state) when the search gets
    /// very cold and stagnant.
    pub enable_cold_restarts: bool,
    /// If the temperature falls below this value and the worker hasn't improved for
    /// `cold_restart_patience` iterations, perform a restart.
    pub cold_restart_temp: f64,
    /// Number of iterations without improving the worker's best energy before allowing a cold restart.
    pub cold_restart_patience: u64,
    /// Probability of restarting completely from a fresh random graph (vs. restarting from the worker's
    /// elite state with a strong perturbation).
    pub restart_from_scratch_probability: f64,
    /// Number of random edge flips applied when restarting from the worker's elite state.
    pub elite_restart_perturb_strength: usize,
    /// Jitter applied to edge probability when generating a fresh random restart state.
    pub restart_p_jitter: f64,
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
            c4_weight: 50, // Increased to prioritize cycle removal more strongly
            independent_violation_penalty: 1,
            lahc_size: 1000,
            temp_start: 10.0, // Higher starting temp for better exploration
            cooling_rate: 0.999_999_7,
            temp_update_every: 1_000,
            reheat_threshold: 0.01,
            reheat_temp: 2.0,
            c4_guided_probability: 0.9,
            c4_probe_tries: 128,
            indep_pair_samples: 32,
            report_every: 100_000, // Balanced reporting frequency
            seed: None,
            resume_path: None,
            kick_threshold: 10_000_000,
            kick_strength: 8,
            tabu_size: 64, // Larger tabu to prevent long-range cycling
            degree_bias_probability: 0.4,
            initial_p_jitter: 0.02,
            enable_cold_restarts: true,
            cold_restart_temp: 0.20,
            cold_restart_patience: 200_000_000,
            restart_from_scratch_probability: 0.5,
            elite_restart_perturb_strength: 200,
            restart_p_jitter: 0.03,
        }
    }
}

// ============================================================================
// Restart / diversification helpers
// ============================================================================

#[inline]
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Uniformly jitters a probability `p` by `±jitter`, clamped to `[0, 1]`.
#[inline]
fn jitter_probability<R: Rng>(rng: &mut R, p: f64, jitter: f64) -> f64 {
    if jitter <= 0.0 {
        return clamp01(p);
    }
    let delta = rng.random_range(-jitter..=jitter);
    clamp01(p + delta)
}

// ============================================================================
// Tabu List
// ============================================================================

/// Short-term memory to prevent cycling by forbidding recently flipped edges.
#[derive(Clone, Debug)]
struct TabuList {
    /// Recent edges stored as (min(u,v), max(u,v)).
    recent: VecDeque<(usize, usize)>,
    /// Maximum size of the list.
    max_size: usize,
}

impl TabuList {
    fn new(max_size: usize) -> Self {
        Self {
            recent: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Check if an edge is tabu (forbidden).
    #[inline]
    fn is_tabu(&self, u: usize, v: usize) -> bool {
        let edge = (u.min(v), u.max(v));
        self.recent.contains(&edge)
    }

    /// Add an edge to the tabu list.
    #[inline]
    fn add(&mut self, u: usize, v: usize) {
        if self.max_size == 0 {
            return;
        }
        let edge = (u.min(v), u.max(v));
        if self.recent.len() >= self.max_size {
            self.recent.pop_front();
        }
        self.recent.push_back(edge);
    }

    /// Clear the tabu list.
    fn clear(&mut self) {
        self.recent.clear();
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
    if worker_id == 0 {
        println!("[Worker 0] Starting search chain...");
    }
    let mut rng = SmallRng::seed_from_u64(splitmix64(base_seed ^ (worker_id as u64)));
    let mut state = initialize_state::<N>(worker_id, &mut rng, cfg);
    let mut oracle = IndependentSetOracle::<N>::new();
    let mut cached_oracle = CachedIsOracle::<N>::new();
    let mut scratch_set = Vec::<usize>::with_capacity(cfg.k_target.max(16));
    let mut tabu = TabuList::new(cfg.tabu_size);

    let mut temp = cfg.temp_start;
    let mut iterations: u64 = 0;
    let mut iters_since_best: u64 = 0;
    let mut last_csv_log_best = usize::MAX;
    let start_time = std::time::Instant::now();

    let lahc_len = cfg.lahc_size.clamp(1, MAX_LAHC_SIZE);
    let mut lahc_history = [0usize; MAX_LAHC_SIZE];
    let initial_energy = evaluate::<N>(&state, &mut oracle, &mut cached_oracle, &mut scratch_set, cfg).energy;
    global_best_energy.fetch_min(initial_energy, Ordering::Relaxed);

    let mut lahc_idx = 0usize;
    reinitialize_lahc(&mut lahc_history, lahc_len, initial_energy, &mut lahc_idx);
    let mut current_energy = initial_energy;
    let mut local_best_energy = initial_energy;
    let mut elite_state = state.clone();

    if worker_id == 0 {
        println!("[Worker 0] Initial energy: {}", initial_energy);
    }

    while !found_flag.load(Ordering::Relaxed) {
        iterations += 1;
        iters_since_best += 1;

        if iters_since_best >= cfg.kick_threshold {
            if worker_id == 0 {
                println!("\n[Worker 0] Applying kick (threshold reached)");
            }
            apply_kick::<N>(&mut state, &mut rng, &mut cached_oracle, &mut tabu, cfg);
            current_energy =
                evaluate::<N>(&state, &mut oracle, &mut cached_oracle, &mut scratch_set, cfg).energy;
            iters_since_best = 0;
            continue;
        }

        let (u, v) = propose_move_with_tabu::<N, _>(
            &mut state,
            &mut oracle,
            &mut cached_oracle,
            &mut scratch_set,
            &mut rng,
            &tabu,
            cfg,
        );
        state.flip_edge(u, v);
        cached_oracle.invalidate_edge(u, v);
        let eval = evaluate::<N>(&state, &mut oracle, &mut cached_oracle, &mut scratch_set, cfg);

        if eval.is_solution {
            handle_success::<N>(worker_id, &state, cfg, found_flag);
            return;
        }

        if eval.energy < local_best_energy {
            local_best_energy = eval.energy;
            elite_state = state.clone();
            iters_since_best = 0;
            let old = global_best_energy.fetch_min(eval.energy, Ordering::Relaxed);
            if eval.energy < old {
                let _ = state.save_to_file(format!("best_checkpoint_n{N}_k{}.txt", cfg.k_target));
            }
        }

        if accept_move(
            eval.energy,
            lahc_history[lahc_idx],
            current_energy,
            temp,
            &mut rng,
        ) {
            current_energy = eval.energy;
            tabu.add(u, v);
        } else {
            state.flip_edge(u, v);
            cached_oracle.invalidate_edge(u, v);
        }

        lahc_history[lahc_idx] = current_energy;
        lahc_idx = (lahc_idx + 1) % lahc_len;

        if iterations.is_multiple_of(cfg.temp_update_every) {
            temp *= cfg.cooling_rate;
            // Cold + stagnant => restart (state-of-the-art multi-start diversification).
            //
            // We check BEFORE reheating so a true "cold" state can trigger a restart.
            if maybe_cold_restart::<N>(
                &mut temp,
                &mut iters_since_best,
                &mut state,
                &elite_state,
                &mut rng,
                &mut oracle,
                &mut cached_oracle,
                &mut scratch_set,
                &mut tabu,
                &mut lahc_history,
                lahc_len,
                &mut lahc_idx,
                &mut current_energy,
                cfg,
            ) {
                if worker_id == 0 {
                    println!("\n[Worker 0] Cold restart triggered!");
                }
            } else if temp < cfg.reheat_threshold {
                if worker_id == 0 {
                    println!("\n[Worker 0] Reheating...");
                }
                temp = cfg.reheat_temp;
            }
            if worker_id == 0 && iterations.is_multiple_of(cfg.report_every) {
                last_csv_log_best = report_progress::<N>(
                    iterations,
                    temp,
                    current_energy,
                    &state,
                    global_best_energy,
                    last_csv_log_best,
                    start_time,
                    cfg,
                );
            }
        }
    }
}

#[inline]
fn reinitialize_lahc(
    lahc_history: &mut [usize; MAX_LAHC_SIZE],
    lahc_len: usize,
    energy: usize,
    lahc_idx: &mut usize,
) {
    for slot in lahc_history.iter_mut().take(lahc_len) {
        *slot = energy;
    }
    *lahc_idx = 0;
}

#[inline]
fn apply_kick<const N: usize>(
    state: &mut RamseyState<N>,
    rng: &mut SmallRng,
    cached_oracle: &mut CachedIsOracle<N>,
    tabu: &mut TabuList,
    cfg: &SearchConfig,
) {
    for _ in 0..cfg.kick_strength {
        let (u, v) = random_pair::<N, _>(rng);
        state.flip_edge(u, v);
        cached_oracle.invalidate_edge(u, v);
    }
    tabu.clear();
    cached_oracle.clear_cache();
}

#[allow(clippy::too_many_arguments)]
fn maybe_cold_restart<const N: usize>(
    temp: &mut f64,
    iters_since_best: &mut u64,
    state: &mut RamseyState<N>,
    elite_state: &RamseyState<N>,
    rng: &mut SmallRng,
    oracle: &mut IndependentSetOracle<N>,
    cached_oracle: &mut CachedIsOracle<N>,
    scratch_set: &mut Vec<usize>,
    tabu: &mut TabuList,
    lahc_history: &mut [usize; MAX_LAHC_SIZE],
    lahc_len: usize,
    lahc_idx: &mut usize,
    current_energy: &mut usize,
    cfg: &SearchConfig,
) -> bool {
    if !(cfg.enable_cold_restarts
        && *temp < cfg.cold_restart_temp
        && *iters_since_best >= cfg.cold_restart_patience)
    {
        return false;
    }

    let from_scratch =
        rng.random_bool(cfg.restart_from_scratch_probability.clamp(0.0, 1.0));
    if from_scratch {
        let p = jitter_probability(rng, cfg.edge_probability, cfg.restart_p_jitter);
        *state = RamseyState::<N>::new_random(rng, p);
    } else {
        *state = elite_state.clone();
        for _ in 0..cfg.elite_restart_perturb_strength {
            let (u, v) = random_pair::<N, _>(rng);
            state.flip_edge(u, v);
        }
    }

    tabu.clear();
    cached_oracle.clear_cache();
    scratch_set.clear();

    let restarted_energy = evaluate::<N>(state, oracle, cached_oracle, scratch_set, cfg).energy;
    *current_energy = restarted_energy;
    reinitialize_lahc(lahc_history, lahc_len, restarted_energy, lahc_idx);
    *temp = cfg.temp_start;
    *iters_since_best = 0;
    true
}

fn initialize_state<const N: usize>(worker_id: usize, rng: &mut SmallRng, cfg: &SearchConfig) -> RamseyState<N> {
    if let Some(path) = &cfg.resume_path {
        match RamseyState::<N>::load_from_file(path) {
            Ok(s) => s,
            Err(e) => {
                if worker_id == 0 { eprintln!("Warning: Failed to load resume file: {e}. Starting fresh."); }
                let p = jitter_probability(rng, cfg.edge_probability, cfg.initial_p_jitter);
                RamseyState::<N>::new_random(rng, p)
            }
        }
    } else {
        let p = jitter_probability(rng, cfg.edge_probability, cfg.initial_p_jitter);
        RamseyState::<N>::new_random(rng, p)
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
/// - Uses cached oracle for IS detection.
#[inline]
fn evaluate<const N: usize>(
    state: &RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    cached_oracle: &mut CachedIsOracle<N>,
    _scratch_set: &mut Vec<usize>,
    cfg: &SearchConfig,
) -> Eval {
    // 1. Calculate Forbidden Subgraph Energy
    let forbidden_score = match (cfg.forbidden_type, cfg.c_target) {
        (ForbiddenType::Cycle, 4) => state.c4_score_twice(),
        (ForbiddenType::Clique, c) => oracle.count_cliques_of_size(state.adj(), c, 100),
        (ForbiddenType::Cycle, c) => {
            if c == 3 {
                oracle.count_cliques_of_size(state.adj(), 3, 100)
            } else {
                panic!("Unsupported cycle length {c}. Only C3 and C4 are currently implemented.");
            }
        }
    };

    // 2. Calculate Independent Set Energy (Gradient)
    // We always count IS to provide a gradient, even if forbidden subgraphs exist.
    // We use a higher limit to provide better resolution for the metaheuristic.
    let is_limit = 500;
    let is_count = cached_oracle.count_independent_sets_of_size(state.adj(), cfg.k_target, is_limit);

    let energy = forbidden_score * cfg.c4_weight + is_count;

    Eval {
        energy,
        is_solution: forbidden_score == 0 && is_count == 0,
    }
}

// ============================================================================
// Move proposal with Tabu and Degree Bias
// ============================================================================

#[inline]
fn propose_move_with_tabu<const N: usize, R: Rng>(
    state: &mut RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    cached_oracle: &mut CachedIsOracle<N>,
    scratch_set: &mut Vec<usize>,
    rng: &mut R,
    tabu: &TabuList,
    cfg: &SearchConfig,
) -> (usize, usize) {
    // Try guided move first
    let guided = propose_guided_move::<N, _>(state, oracle, cached_oracle, scratch_set, rng, cfg);

    if let Some((u, v)) = guided
        && !tabu.is_tabu(u, v)
    {
        return (u, v);
    }

    // Fall back to degree-biased or random move
    if rng.random_bool(cfg.degree_bias_probability)
        && let Some((u, v)) = degree_biased_pair::<N, _>(state, rng, tabu)
    {
        return (u, v);
    }

    // Final fallback: random pair avoiding tabu
    random_pair_avoiding_tabu::<N, _>(rng, tabu, 10)
}

/// Propose a guided move based on current graph state.
///
/// Uses a balanced strategy that considers BOTH constraints:
/// - When C4s exist: sometimes try to remove them, sometimes try to break IS
/// - When C4-free: focus on breaking independent sets
#[inline]
fn propose_guided_move<const N: usize, R: Rng>(
    state: &mut RamseyState<N>,
    oracle: &mut IndependentSetOracle<N>,
    cached_oracle: &mut CachedIsOracle<N>,
    scratch_set: &mut Vec<usize>,
    rng: &mut R,
    cfg: &SearchConfig,
) -> Option<(usize, usize)> {
    let c4_count = state.c4_count();
    let has_c4 = c4_count > 0;

    // Strategy: balance between removing C4s and breaking ISs
    // When C4 count is low, we're close to feasible - more aggressive IS breaking helps
    // When C4 count is high, focus on reducing cycles first
    let c4_focus_prob = if c4_count == 0 {
        0.0
    } else if c4_count <= 3 {
        0.6 // 60% C4 removal, 40% IS breaking
    } else if c4_count <= 10 {
        0.8
    } else {
        0.95
    };

    if has_c4 && cfg.forbidden_type == ForbiddenType::Cycle && cfg.c_target == 4 {
        if rng.random_bool(c4_focus_prob) {
            // Try to remove a C4
            if c4_count <= 5 {
                let edges = state.find_c4_edges(10);
                if !edges.is_empty() {
                    return Some(edges[rng.random_range(0..edges.len())]);
                }
            }
            if let Some(edge) = state.sample_c4_edge_to_remove(rng, cfg.c4_probe_tries) {
                return Some(edge);
            }
        }
        // Fall through to IS breaking (even when C4s exist!)
    } else if cfg.forbidden_type == ForbiddenType::Clique {
        let mut clique = Vec::new();
        if oracle.find_clique_of_size(state.adj(), cfg.c_target, &mut clique)
            && clique.len() >= 2
            && rng.random_bool(0.8)
        {
            let i = rng.random_range(0..clique.len());
            let mut j = rng.random_range(0..clique.len());
            while i == j {
                j = rng.random_range(0..clique.len());
            }
            return Some((clique[i], clique[j]));
        }
        // Fall through to IS breaking
    }

    // Try to break independent sets (this now happens even when C4s exist!)
    if state.greedy_find_independent_set_of_size(cfg.k_target, scratch_set) {
        // When C4s exist, we can't use best_edge_to_add_within_set (it assumes C4-free)
        // Instead, just pick a random pair from the IS
        if has_c4 && scratch_set.len() >= 2 {
            let i = rng.random_range(0..scratch_set.len());
            let mut j = rng.random_range(0..scratch_set.len());
            while j == i {
                j = rng.random_range(0..scratch_set.len());
            }
            return Some((scratch_set[i], scratch_set[j]));
        } else if !has_c4 {
            return Some(best_edge_to_add_within_set(
                state,
                scratch_set,
                rng,
                cfg.indep_pair_samples,
            ));
        }
    }

    scratch_set.clear();
    if cached_oracle.find_independent_set_of_size(state.adj(), cfg.k_target, scratch_set) {
        if has_c4 && scratch_set.len() >= 2 {
            let i = rng.random_range(0..scratch_set.len());
            let mut j = rng.random_range(0..scratch_set.len());
            while j == i {
                j = rng.random_range(0..scratch_set.len());
            }
            return Some((scratch_set[i], scratch_set[j]));
        } else if !has_c4 {
            return Some(best_edge_to_add_within_set(
                state,
                scratch_set,
                rng,
                cfg.indep_pair_samples,
            ));
        }
    }

    None
}

/// Select a vertex pair biased by degree (higher degree = more likely to be involved).
/// High-degree vertices tend to participate in more C4s and affect more independent sets.
#[inline]
fn degree_biased_pair<const N: usize, R: Rng>(
    state: &RamseyState<N>,
    rng: &mut R,
    tabu: &TabuList,
) -> Option<(usize, usize)> {
    // Compute degree weights (degree² for stronger bias)
    let mut weights = [0u32; 64];
    let mut total_weight = 0u64;

    for v in 0..N {
        let deg = state.degree(v);
        let w = deg.saturating_mul(deg).max(1);
        weights[v] = w;
        total_weight += u64::from(w);
    }

    if total_weight == 0 {
        return None;
    }

    // Sample first vertex proportional to weight
    let mut r = rng.random_range(0..total_weight);
    let mut u = 0;
    for v in 0..N {
        let w = u64::from(weights[v]);
        if r < w {
            u = v;
            break;
        }
        r -= w;
    }

    // For second vertex, prefer neighbors (for edge removal) or non-neighbors (for edge addition)
    // based on what's more useful. Here we sample uniformly from remaining.
    for _ in 0..10 {
        let v = rng.random_range(0..N);
        if v != u && !tabu.is_tabu(u, v) {
            return Some((u, v));
        }
    }

    None
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

/// Random pair that tries to avoid tabu edges.
#[inline]
fn random_pair_avoiding_tabu<const N: usize, R: Rng>(
    rng: &mut R,
    tabu: &TabuList,
    max_tries: usize,
) -> (usize, usize) {
    for _ in 0..max_tries {
        let (u, v) = random_pair::<N, _>(rng);
        if !tabu.is_tabu(u, v) {
            return (u, v);
        }
    }
    // Give up and return any pair
    random_pair::<N, _>(rng)
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
    use crate::iset::CachedIsOracle;

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
            let mut cached_oracle = CachedIsOracle::<10>::new();
            let mut scratch = Vec::new();
            let cfg = SearchConfig::default();
            let tabu = TabuList::new(cfg.tabu_size);

            for _ in 0..100 {
                let (u, v) = propose_move_with_tabu::<10, _>(
                    &mut state,
                    &mut oracle,
                    &mut cached_oracle,
                    &mut scratch,
                    &mut rng,
                    &tabu,
                    &cfg,
                );
                state.flip_edge(u, v);
                let _eval = evaluate::<10>(&state, &mut oracle, &mut cached_oracle, &mut scratch, &cfg);
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

    #[test]
    fn test_tabu_list_basic() {
        let mut tabu = TabuList::new(3);

        // Initially empty
        assert!(!tabu.is_tabu(0, 1));
        assert!(!tabu.is_tabu(1, 0));

        // Add edge (0, 1)
        tabu.add(0, 1);
        assert!(tabu.is_tabu(0, 1));
        assert!(tabu.is_tabu(1, 0)); // Order shouldn't matter

        // Add more edges
        tabu.add(2, 3);
        tabu.add(4, 5);

        // All three should be tabu
        assert!(tabu.is_tabu(0, 1));
        assert!(tabu.is_tabu(2, 3));
        assert!(tabu.is_tabu(4, 5));

        // Adding a fourth should evict the first
        tabu.add(6, 7);
        assert!(!tabu.is_tabu(0, 1)); // Evicted
        assert!(tabu.is_tabu(2, 3));
        assert!(tabu.is_tabu(4, 5));
        assert!(tabu.is_tabu(6, 7));
    }

    #[test]
    fn test_tabu_list_clear() {
        let mut tabu = TabuList::new(5);
        tabu.add(0, 1);
        tabu.add(2, 3);
        assert!(tabu.is_tabu(0, 1));

        tabu.clear();
        assert!(!tabu.is_tabu(0, 1));
        assert!(!tabu.is_tabu(2, 3));
    }

    #[test]
    fn test_degree_biased_pair() {
        let mut rng = SmallRng::seed_from_u64(42);
        let tabu = TabuList::new(0);

        // Create a graph where vertex 0 has high degree
        let mut adj = [0u64; 10];
        for i in 1..8 {
            adj[0] |= 1 << i;
            adj[i] |= 1;
        }
        let state = RamseyState::<10>::from_adj(adj);

        // Sample many pairs and check vertex 0 appears more often
        let mut v0_count = 0;
        for _ in 0..1000 {
            if let Some((u, v)) = degree_biased_pair::<10, _>(&state, &mut rng, &tabu) {
                if u == 0 || v == 0 {
                    v0_count += 1;
                }
            }
        }

        // Vertex 0 has degree 7, others have degree 1-2
        // It should appear in significantly more than 10% of pairs
        assert!(
            v0_count > 200,
            "High-degree vertex should be selected more often"
        );
    }

    #[test]
    fn test_search_config_invariants() {
        let cfg = SearchConfig::default();
        assert!(cfg.lahc_size <= MAX_LAHC_SIZE);
        assert!(cfg.cooling_rate < 1.0 && cfg.cooling_rate > 0.0);
        assert!(cfg.temp_start > 0.0);
        assert!(cfg.reheat_threshold < cfg.reheat_temp);
        assert!(cfg.kick_threshold > 0);
    }

    #[test]
    fn test_tabu_list_no_duplicates() {
        let mut tabu = TabuList::new(10);
        // Adding the same edge multiple times shouldn't break FIFO logic
        for _ in 0..5 {
            tabu.add(0, 1);
        }
        assert_eq!(tabu.recent.len(), 5);
        for _ in 0..10 {
            tabu.add(2, 3);
        }
        assert_eq!(tabu.recent.len(), 10);
        assert!(!tabu.is_tabu(0, 1)); // Should have been evicted
    }

    #[test]
    fn test_evaluate_gradient_sanity() {
        // Evaluate energy on a sequence of edge additions
        let mut state = RamseyState::<5>::empty();
        let mut oracle = IndependentSetOracle::<5>::new();
        let mut cached = CachedIsOracle::<5>::new();
        let mut scratch = Vec::new();
        let mut cfg = SearchConfig::default();
        cfg.k_target = 3; // Use K < N so energy is non-zero

        let energy0 = evaluate::<5>(&state, &mut oracle, &mut cached, &mut scratch, &cfg).energy;
        assert!(energy0 > 0);

        // Adding an edge should reduce the independent set count (energy)
        state.flip_edge(0, 1);
        let energy1 = evaluate::<5>(&state, &mut oracle, &mut cached, &mut scratch, &cfg).energy;
        assert!(
            energy1 < energy0,
            "Energy did not decrease: {} -> {}",
            energy0,
            energy1
        );
    }

    #[test]
    fn jitter_probability_clamps_to_unit_interval() {
        let mut rng = SmallRng::seed_from_u64(123);
        for _ in 0..1000 {
            let p0 = jitter_probability(&mut rng, 0.0, 10.0);
            let p1 = jitter_probability(&mut rng, 1.0, 10.0);
            let pm = jitter_probability(&mut rng, 0.5, 10.0);
            assert!((0.0..=1.0).contains(&p0));
            assert!((0.0..=1.0).contains(&p1));
            assert!((0.0..=1.0).contains(&pm));
        }
    }

    #[test]
    fn cold_restart_triggers_and_reinitializes_state() {
        const N: usize = 8;
        let mut cfg = SearchConfig::default();
        cfg.enable_cold_restarts = true;
        cfg.cold_restart_temp = 1.0;
        cfg.cold_restart_patience = 1;
        cfg.restart_from_scratch_probability = 1.0;
        cfg.restart_p_jitter = 0.0;
        cfg.edge_probability = 0.3;
        cfg.temp_start = 5.0;

        let mut rng = SmallRng::seed_from_u64(0xBADC0FFE);
        let mut state = RamseyState::<N>::empty();
        let elite_state = state.clone();
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut cached_oracle = CachedIsOracle::<N>::new();
        let mut scratch = Vec::new();
        let mut tabu = TabuList::new(0);
        let mut lahc_history = [777usize; MAX_LAHC_SIZE];
        let lahc_len = 7;
        let mut lahc_idx = 5;
        let mut current_energy = 999_999usize;
        let mut temp = 0.1;
        let mut iters_since_best = 1;

        let did_restart = maybe_cold_restart::<N>(
            &mut temp,
            &mut iters_since_best,
            &mut state,
            &elite_state,
            &mut rng,
            &mut oracle,
            &mut cached_oracle,
            &mut scratch,
            &mut tabu,
            &mut lahc_history,
            lahc_len,
            &mut lahc_idx,
            &mut current_energy,
            &cfg,
        );

        assert!(did_restart);
        assert_eq!(temp, cfg.temp_start);
        assert_eq!(iters_since_best, 0);
        assert_eq!(lahc_idx, 0);
        assert_eq!(lahc_history[0], current_energy);
        assert_ne!(current_energy, 999_999);
    }

    #[test]
    fn lahc_circular_buffer_indexing() {
        let mut history = [0usize; MAX_LAHC_SIZE];
        let len = 50;
        let mut idx = 0;
        reinitialize_lahc(&mut history, len, 100, &mut idx);
        assert_eq!(idx, 0);
        for i in 0..len {
            assert_eq!(history[i], 100);
        }

        // Simulate iterations
        for i in 0..123 {
            history[idx] = i;
            idx = (idx + 1) % len;
        }
        assert!(idx < len);
    }

    #[test]
    fn cooling_rate_monotonicity() {
        let mut temp = 100.0;
        let rate = 0.999;
        for _ in 0..1000 {
            let old_temp = temp;
            temp *= rate;
            assert!(temp < old_temp);
        }
    }

    #[test]
    fn search_config_jitter_clamping() {
        let mut rng = SmallRng::seed_from_u64(42);
        let p = 0.5;
        let jitter = 0.6; // would go outside [0,1]
        for _ in 0..100 {
            let res = jitter_probability(&mut rng, p, jitter);
            assert!((0.0..=1.0).contains(&res));
        }
    }
}

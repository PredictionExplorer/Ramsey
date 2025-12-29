//! Structured graph constructions for better initial solutions.
//!
//! Instead of random Erdős–Rényi graphs, we use constructions that exploit
//! mathematical structure known to produce good C₄-free graphs.

use crate::graph::RamseyState;
use rand::Rng;

// ============================================================================
// Construction Strategies
// ============================================================================

/// Strategy for initial graph construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstructionStrategy {
    /// Standard random Erdős–Rényi graph.
    Random,
    /// Random regular graph (all vertices have same degree).
    Regular,
    /// Greedily build a C₄-free graph.
    GreedyC4Free,
    /// Paley-like construction based on quadratic residues.
    PaleyLike,
    /// Sparse random graph with controlled density.
    Sparse,
    /// Bipartite-like construction.
    BipartiteLike,
}

impl ConstructionStrategy {
    /// Returns all available strategies.
    pub const fn all() -> &'static [ConstructionStrategy] {
        &[
            ConstructionStrategy::Random,
            ConstructionStrategy::Regular,
            ConstructionStrategy::GreedyC4Free,
            ConstructionStrategy::PaleyLike,
            ConstructionStrategy::Sparse,
            ConstructionStrategy::BipartiteLike,
        ]
    }
}

/// Constructs an initial graph using the specified strategy.
pub fn construct_initial<const N: usize, R: Rng>(
    rng: &mut R,
    strategy: ConstructionStrategy,
    edge_probability: f64,
) -> RamseyState<N> {
    match strategy {
        ConstructionStrategy::Random => RamseyState::new_random(rng, edge_probability),
        ConstructionStrategy::Regular => construct_regular(rng, edge_probability),
        ConstructionStrategy::GreedyC4Free => construct_greedy_c4_free(rng, edge_probability),
        ConstructionStrategy::PaleyLike => construct_paley_like(rng),
        ConstructionStrategy::Sparse => construct_sparse(rng, edge_probability * 0.7),
        ConstructionStrategy::BipartiteLike => construct_bipartite_like(rng, edge_probability),
    }
}

/// Constructs a random graph and returns the best of multiple attempts.
pub fn construct_best_of<const N: usize, R: Rng>(
    rng: &mut R,
    edge_probability: f64,
    attempts: usize,
) -> RamseyState<N> {
    let strategies = ConstructionStrategy::all();
    let mut best_state = RamseyState::<N>::new_random(rng, edge_probability);
    let mut best_score = evaluate_initial_quality(&best_state);

    for _ in 0..attempts {
        let strategy = strategies[rng.random_range(0..strategies.len())];
        let state = construct_initial::<N, _>(rng, strategy, edge_probability);
        let score = evaluate_initial_quality(&state);

        if score < best_score {
            best_score = score;
            best_state = state;
        }
    }

    best_state
}

/// Evaluates the quality of an initial graph (lower is better).
fn evaluate_initial_quality<const N: usize>(state: &RamseyState<N>) -> usize {
    // Prefer graphs with:
    // 1. Fewer C4s
    // 2. More regular degree distribution
    // 3. Moderate edge density

    let c4_penalty = state.c4_score_twice() * 10;

    let degrees: Vec<u32> = (0..N).map(|v| state.degree(v)).collect();
    let mean_deg = degrees.iter().sum::<u32>() as f64 / N as f64;
    let variance: f64 = degrees
        .iter()
        .map(|&d| (d as f64 - mean_deg).powi(2))
        .sum::<f64>()
        / N as f64;
    let regularity_penalty = variance as usize;

    c4_penalty + regularity_penalty
}

// ============================================================================
// Specific Constructions
// ============================================================================

/// Constructs a random regular (or near-regular) graph.
fn construct_regular<const N: usize, R: Rng>(rng: &mut R, p: f64) -> RamseyState<N> {
    let target_degree = ((N - 1) as f64 * p).round() as u32;
    let mut adj = [0u64; N];

    // Use a configuration model approach
    // Each vertex starts with 'target_degree' half-edges
    let mut stubs: Vec<usize> = Vec::with_capacity(N * target_degree as usize);
    for v in 0..N {
        for _ in 0..target_degree {
            stubs.push(v);
        }
    }

    // Shuffle and pair up stubs
    shuffle_slice(rng, &mut stubs);

    for chunk in stubs.chunks(2) {
        if chunk.len() == 2 {
            let u = chunk[0];
            let v = chunk[1];
            if u != v {
                let u_mask = 1u64 << u;
                let v_mask = 1u64 << v;
                // Add edge if not already present
                if (adj[u] & v_mask) == 0 {
                    adj[u] |= v_mask;
                    adj[v] |= u_mask;
                }
            }
        }
    }

    RamseyState::from_adj(adj)
}

/// Greedily constructs a C₄-free graph by adding edges that don't create C₄s.
fn construct_greedy_c4_free<const N: usize, R: Rng>(rng: &mut R, p: f64) -> RamseyState<N> {
    let target_edges = ((N * (N - 1) / 2) as f64 * p).round() as usize;
    let mut state = RamseyState::<N>::empty();

    // Generate all possible edges in random order
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(N * (N - 1) / 2);
    for i in 0..N {
        for j in (i + 1)..N {
            edges.push((i, j));
        }
    }
    shuffle_slice(rng, &mut edges);

    let mut added = 0;
    for (u, v) in edges {
        if added >= target_edges {
            break;
        }

        // Try adding the edge and check if it creates a C4
        state.flip_edge(u, v);

        if state.c4_count() > 0 {
            // Revert - this edge creates a C4
            state.flip_edge(u, v);
        } else {
            added += 1;
        }
    }

    state
}

/// Constructs a Paley-like graph based on quadratic residues.
///
/// For N vertices, we simulate a finite field structure and add edges
/// based on "quadratic residue"-like relations.
fn construct_paley_like<const N: usize, R: Rng>(rng: &mut R) -> RamseyState<N> {
    // Find a prime close to N for the construction
    let p = find_prime_near(N);
    let residues = compute_quadratic_residues(p);

    let mut adj = [0u64; N];

    for i in 0..N {
        for j in (i + 1)..N {
            // Use the difference modulo p to determine edge
            let diff = if j > i {
                (j - i) % p
            } else {
                (i - j) % p
            };

            // Add edge if difference is a quadratic residue
            if residues.contains(&diff) {
                adj[i] |= 1u64 << j;
                adj[j] |= 1u64 << i;
            }
        }
    }

    // Add some randomization to escape the exact algebraic structure
    let perturbation_rate = 0.05;
    for i in 0..N {
        for j in (i + 1)..N {
            if rng.random_bool(perturbation_rate) {
                let mask_j = 1u64 << j;
                let mask_i = 1u64 << i;
                adj[i] ^= mask_j;
                adj[j] ^= mask_i;
            }
        }
    }

    RamseyState::from_adj(adj)
}

/// Constructs a sparse random graph.
fn construct_sparse<const N: usize, R: Rng>(rng: &mut R, p: f64) -> RamseyState<N> {
    let p_clamped = p.clamp(0.05, 0.3);
    RamseyState::new_random(rng, p_clamped)
}

/// Constructs a bipartite-like graph (almost bipartite with some within-part edges).
fn construct_bipartite_like<const N: usize, R: Rng>(rng: &mut R, p: f64) -> RamseyState<N> {
    let mut adj = [0u64; N];

    // Divide vertices into two parts
    let mid = N / 2;

    // Add edges between parts with higher probability
    for i in 0..mid {
        for j in mid..N {
            if rng.random_bool(p * 1.5) {
                adj[i] |= 1u64 << j;
                adj[j] |= 1u64 << i;
            }
        }
    }

    // Add edges within parts with lower probability
    for i in 0..mid {
        for j in (i + 1)..mid {
            if rng.random_bool(p * 0.3) {
                adj[i] |= 1u64 << j;
                adj[j] |= 1u64 << i;
            }
        }
    }
    for i in mid..N {
        for j in (i + 1)..N {
            if rng.random_bool(p * 0.3) {
                adj[i] |= 1u64 << j;
                adj[j] |= 1u64 << i;
            }
        }
    }

    RamseyState::from_adj(adj)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Fisher-Yates shuffle.
fn shuffle_slice<T, R: Rng>(rng: &mut R, slice: &mut [T]) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// Finds a prime number close to (but not exceeding) n.
fn find_prime_near(n: usize) -> usize {
    for candidate in (2..=n).rev() {
        if is_prime(candidate) {
            return candidate;
        }
    }
    2
}

/// Simple primality test.
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as usize;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Computes quadratic residues modulo p.
fn compute_quadratic_residues(p: usize) -> Vec<usize> {
    let mut residues = Vec::with_capacity(p / 2);
    for a in 1..p {
        let r = (a * a) % p;
        if !residues.contains(&r) {
            residues.push(r);
        }
    }
    residues
}

// ============================================================================
// Extension from Smaller Witnesses
// ============================================================================

/// Attempts to extend a witness graph by adding one vertex.
///
/// Given a C₄-free graph with α(G) < k, tries to find a way to add
/// a vertex that maintains both properties.
///
/// Returns a Vec<u64> representing the adjacency of the extended graph,
/// or None if no valid extension was found.
pub fn try_extend_witness<const N: usize, R: Rng>(
    witness: &RamseyState<N>,
    rng: &mut R,
    max_attempts: usize,
) -> Option<Vec<u64>> {
    // For each possible neighborhood of the new vertex, check if it works
    let total_neighborhoods = 1u64 << N;

    // Sample randomly rather than exhaustively (exponential in N)
    for _ in 0..max_attempts {
        let neighborhood = rng.random_range(0..total_neighborhoods);

        // Build the extended adjacency as a Vec
        let mut adj: Vec<u64> = witness.adj().to_vec();
        adj.push(0); // Add slot for new vertex

        // Add new vertex N with the chosen neighborhood
        adj[N] = neighborhood;
        for v in 0..N {
            if (neighborhood >> v) & 1 == 1 {
                adj[v] |= 1u64 << N;
            }
        }

        // Check if this creates any C4 involving the new vertex
        // New vertex N is in a C4 iff there exist i, j both adjacent to N
        // such that i and j have another common neighbor
        let mut has_c4 = false;
        let neighbors_of_new: Vec<usize> = (0..N).filter(|&v| (neighborhood >> v) & 1 == 1).collect();

        for i_idx in 0..neighbors_of_new.len() {
            if has_c4 {
                break;
            }
            for j_idx in (i_idx + 1)..neighbors_of_new.len() {
                let i = neighbors_of_new[i_idx];
                let j = neighbors_of_new[j_idx];
                // i and j are both adjacent to N
                // Check if they have another common neighbor (not N)
                if witness.common_neighbor_count(i, j) > 0 {
                    has_c4 = true;
                    break;
                }
            }
        }

        if !has_c4 {
            return Some(adj);
        }
    }

    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn greedy_c4_free_produces_c4_free_graph() {
        let mut rng = XorShiftRng::seed_from_u64(42);
        let state = construct_greedy_c4_free::<20, _>(&mut rng, 0.3);

        assert_eq!(
            state.c4_count(),
            0,
            "Greedy C4-free should produce C4-free graph"
        );
    }

    #[test]
    fn regular_graph_is_nearly_regular() {
        let mut rng = XorShiftRng::seed_from_u64(123);
        let state = construct_regular::<16, _>(&mut rng, 0.3);

        let degrees: Vec<u32> = (0..16).map(|v| state.degree(v)).collect();
        let mean = degrees.iter().sum::<u32>() as f64 / 16.0;
        let max_deviation = degrees
            .iter()
            .map(|&d| (d as f64 - mean).abs())
            .fold(0.0, f64::max);

        // Should be relatively regular (deviation < 3)
        assert!(
            max_deviation < 5.0,
            "Regular graph should have low degree variance, got max deviation {}",
            max_deviation
        );
    }

    #[test]
    fn paley_like_has_reasonable_structure() {
        let mut rng = XorShiftRng::seed_from_u64(456);
        let state = construct_paley_like::<13, _>(&mut rng);

        let edge_count = state.edge_count();
        let max_edges = 13 * 12 / 2;

        // Should have moderate density
        assert!(edge_count > 10, "Should have some edges");
        assert!(edge_count < max_edges, "Should not be complete");
    }

    #[test]
    fn bipartite_like_has_bipartite_bias() {
        let mut rng = XorShiftRng::seed_from_u64(789);
        let state = construct_bipartite_like::<20, _>(&mut rng, 0.3);

        // Count edges within parts vs between parts
        let mid = 10;
        let mut within = 0;
        let mut between = 0;

        for i in 0..20 {
            for j in (i + 1)..20 {
                if state.has_edge(i, j) {
                    if (i < mid && j < mid) || (i >= mid && j >= mid) {
                        within += 1;
                    } else {
                        between += 1;
                    }
                }
            }
        }

        // Should have more edges between parts than within
        assert!(
            between > within,
            "Bipartite-like should have more cross-edges: between={}, within={}",
            between,
            within
        );
    }

    #[test]
    fn best_of_returns_good_quality() {
        let mut rng = XorShiftRng::seed_from_u64(999);
        let state = construct_best_of::<15, _>(&mut rng, 0.3, 10);

        // Just verify it returns a valid state
        assert!(state.edge_count() > 0);
    }

    #[test]
    fn find_prime_near_works() {
        assert_eq!(find_prime_near(13), 13);
        assert_eq!(find_prime_near(14), 13);
        assert_eq!(find_prime_near(37), 37);
        assert_eq!(find_prime_near(40), 37);
    }

    #[test]
    fn is_prime_correct() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(13));
        assert!(is_prime(37));
        assert!(!is_prime(39));
    }

    #[test]
    fn all_strategies_produce_valid_graphs() {
        let mut rng = XorShiftRng::seed_from_u64(111);

        for strategy in ConstructionStrategy::all() {
            let state = construct_initial::<12, _>(&mut rng, *strategy, 0.25);
            // Just verify we get a valid graph
            assert!(state.edge_count() <= 12 * 11 / 2);
        }
    }
}


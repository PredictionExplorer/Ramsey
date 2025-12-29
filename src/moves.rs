//! Advanced move operators for the Ramsey search.
//!
//! Beyond single edge flips, this module provides compound moves that can
//! escape local minima and explore the search space more effectively.

use crate::graph::RamseyState;
use crate::iset::IndependentSetOracle;
use rand::Rng;

// ============================================================================
// Move Types
// ============================================================================

/// Different types of moves that can be applied to a graph.
#[derive(Clone, Debug)]
pub enum MoveType {
    /// Flip a single edge.
    Single(usize, usize),
    /// Flip two edges simultaneously.
    Double((usize, usize), (usize, usize)),
    /// Flip a sequence of edges.
    Multi(Vec<(usize, usize)>),
    /// Flip a set of edges (typically incident to a single vertex).
    ///
    /// This is intentionally represented as a list of *edge flips* so it is self-inverse:
    /// applying the same move again reverts it, which the search driver relies on.
    VertexRewire(Vec<(usize, usize)>),
    /// Swap edges along a path (2-opt style).
    PathSwap(Vec<(usize, usize)>),
}

impl MoveType {
    /// Returns the edges involved in this move.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        match self {
            MoveType::Single(u, v) => vec![(*u, *v)],
            MoveType::Double(e1, e2) => vec![*e1, *e2],
            MoveType::Multi(edges) => edges.clone(),
            MoveType::VertexRewire(edges) => edges.clone(),
            MoveType::PathSwap(edges) => edges.clone(),
        }
    }

    /// Calls `f(u, v)` for each flipped edge in this move.
    ///
    /// This avoids allocating a `Vec` in the hot loop (unlike `edges()`).
    #[inline]
    pub fn for_each_edge<F: FnMut(usize, usize)>(&self, mut f: F) {
        match self {
            MoveType::Single(u, v) => f(*u, *v),
            MoveType::Double((u1, v1), (u2, v2)) => {
                f(*u1, *v1);
                f(*u2, *v2);
            }
            MoveType::Multi(edges) | MoveType::PathSwap(edges) => {
                for &(u, v) in edges {
                    f(u, v);
                }
            }
            MoveType::VertexRewire(edges) => {
                for &(u, v) in edges {
                    f(u, v);
                }
            }
        }
    }
}

// ============================================================================
// Compound Move Generator
// ============================================================================

/// Generator for compound moves.
pub struct CompoundMoveGenerator<const N: usize> {
    /// Probability of trying a compound move vs single move.
    pub compound_probability: f64,
    /// Maximum depth for compound moves.
    pub max_depth: usize,
}

impl<const N: usize> Default for CompoundMoveGenerator<N> {
    fn default() -> Self {
        Self {
            compound_probability: 0.15,
            max_depth: 3,
        }
    }
}

impl<const N: usize> CompoundMoveGenerator<N> {
    /// Creates a new compound move generator.
    pub fn new(compound_probability: f64, max_depth: usize) -> Self {
        Self {
            compound_probability,
            max_depth,
        }
    }

    /// Generates a move, potentially compound.
    pub fn generate<R: Rng>(
        &self,
        state: &RamseyState<N>,
        oracle: &mut IndependentSetOracle<N>,
        rng: &mut R,
        k_target: usize,
    ) -> MoveType {
        if rng.random_bool(self.compound_probability) {
            self.generate_compound(state, oracle, rng, k_target)
        } else {
            let (u, v) = random_pair::<N, R>(rng);
            MoveType::Single(u, v)
        }
    }

    /// Generates a compound move unconditionally (caller controls probability).
    #[inline]
    pub fn generate_compound_only<R: Rng>(
        &self,
        state: &RamseyState<N>,
        oracle: &mut IndependentSetOracle<N>,
        rng: &mut R,
        k_target: usize,
    ) -> MoveType {
        self.generate_compound(state, oracle, rng, k_target)
    }

    /// Generates a compound move.
    fn generate_compound<R: Rng>(
        &self,
        state: &RamseyState<N>,
        oracle: &mut IndependentSetOracle<N>,
        rng: &mut R,
        k_target: usize,
    ) -> MoveType {
        let choice = rng.random_range(0..4);
        match choice {
            0 => self.generate_double(state, rng),
            1 => self.generate_c4_break(state, rng),
            2 => self.generate_is_break(state, oracle, rng, k_target),
            _ => self.generate_vertex_rewire(state, rng),
        }
    }

    /// Generates a double flip move.
    fn generate_double<R: Rng>(&self, _state: &RamseyState<N>, rng: &mut R) -> MoveType {
        let (u1, v1) = random_pair::<N, R>(rng);
        let (u2, v2) = random_pair::<N, R>(rng);
        MoveType::Double((u1, v1), (u2, v2))
    }

    /// Generates a move that breaks a C4 by removing multiple edges.
    fn generate_c4_break<R: Rng>(&self, state: &RamseyState<N>, rng: &mut R) -> MoveType {
        if state.c4_count() == 0 {
            let (u, v) = random_pair::<N, R>(rng);
            return MoveType::Single(u, v);
        }

        // Find edges that participate in C4s
        let c4_edges = state.find_c4_edges(20);
        if c4_edges.is_empty() {
            let (u, v) = random_pair::<N, R>(rng);
            return MoveType::Single(u, v);
        }

        // Remove 1-3 edges from C4s
        let num_edges = rng.random_range(1..=3.min(c4_edges.len()));
        let mut selected = Vec::with_capacity(num_edges);

        for _ in 0..num_edges {
            let idx = rng.random_range(0..c4_edges.len());
            if !selected.contains(&c4_edges[idx]) {
                selected.push(c4_edges[idx]);
            }
        }

        if selected.len() == 1 {
            MoveType::Single(selected[0].0, selected[0].1)
        } else {
            MoveType::Multi(selected)
        }
    }

    /// Generates a move that breaks independent sets by adding edges within them.
    fn generate_is_break<R: Rng>(
        &self,
        state: &RamseyState<N>,
        oracle: &mut IndependentSetOracle<N>,
        rng: &mut R,
        k_target: usize,
    ) -> MoveType {
        let mut is_vertices = Vec::new();

        if !oracle.find_independent_set_of_size(state.adj(), k_target, &mut is_vertices) {
            let (u, v) = random_pair::<N, R>(rng);
            return MoveType::Single(u, v);
        }

        if is_vertices.len() < 2 {
            let (u, v) = random_pair::<N, R>(rng);
            return MoveType::Single(u, v);
        }

        // Add 1-2 edges within the IS
        let mut edges = Vec::new();
        let num_to_add = rng.random_range(1..=2.min(is_vertices.len() / 2));

        for _ in 0..num_to_add {
            let i = rng.random_range(0..is_vertices.len());
            let mut j = rng.random_range(0..is_vertices.len());
            while j == i {
                j = rng.random_range(0..is_vertices.len());
            }

            let u = is_vertices[i];
            let v = is_vertices[j];

            if !state.has_edge(u, v) && !edges.contains(&(u.min(v), u.max(v))) {
                edges.push((u.min(v), u.max(v)));
            }
        }

        if edges.is_empty() {
            let (u, v) = random_pair::<N, R>(rng);
            MoveType::Single(u, v)
        } else if edges.len() == 1 {
            MoveType::Single(edges[0].0, edges[0].1)
        } else {
            MoveType::Multi(edges)
        }
    }

    /// Generates a vertex rewire move.
    fn generate_vertex_rewire<R: Rng>(&self, state: &RamseyState<N>, rng: &mut R) -> MoveType {
        // Pick a vertex (prefer high-degree vertices as they're more impactful)
        let degrees: Vec<(usize, u32)> = (0..N).map(|v| (v, state.degree(v))).collect();

        let v = if rng.random_bool(0.7) {
            // Biased toward high degree
            let mut best_v = 0;
            let mut best_deg = 0;
            for &(vertex, deg) in &degrees {
                if deg > best_deg {
                    best_deg = deg;
                    best_v = vertex;
                }
            }
            best_v
        } else {
            rng.random_range(0..N)
        };

        // Compute new neighborhood
        let current_degree = state.degree(v);
        let target_degree = current_degree; // Keep same degree

        // Build new neighborhood avoiding C4 creation
        let mut new_neighbors = Vec::new();
        let mut candidates: Vec<usize> = (0..N).filter(|&u| u != v).collect();
        shuffle_slice(rng, &mut candidates);

        let adj = state.adj();

        // Record current neighbors of v so we can produce a self-inverse "flip list".
        let current_neighbors_mask = adj[v] & !(1u64 << v);

        for u in candidates {
            if new_neighbors.len() >= target_degree as usize {
                break;
            }

            // Check if adding edge (v, u) would create C4 with existing new neighbors
            // (u and any new neighbor should not share a common neighbor)
            let mut creates_c4 = false;
            for &w in &new_neighbors {
                // Check if u and w have a common neighbor (other than v)
                let common: u64 = adj[u] & adj[w];
                if common.count_ones() > 0 {
                    creates_c4 = true;
                    break;
                }
            }

            if !creates_c4 {
                new_neighbors.push(u);
            }
        }

        // Compute the symmetric difference between current neighbors and desired neighbors.
        let mut desired_mask = 0u64;
        for &u in &new_neighbors {
            desired_mask |= 1u64 << u;
        }
        let flips = current_neighbors_mask ^ desired_mask;

        let mut edges_to_flip = Vec::new();
        let mut t = flips;
        while t != 0 {
            let u = t.trailing_zeros() as usize;
            t &= t - 1;
            edges_to_flip.push((v, u));
        }

        if edges_to_flip.is_empty() {
            let (u, w) = random_pair::<N, R>(rng);
            MoveType::Single(u, w)
        } else if edges_to_flip.len() == 1 {
            MoveType::Single(edges_to_flip[0].0, edges_to_flip[0].1)
        } else {
            MoveType::VertexRewire(edges_to_flip)
        }
    }
}

// ============================================================================
// Move Application
// ============================================================================

/// Applies a move to a graph state.
pub fn apply_move<const N: usize>(state: &mut RamseyState<N>, mv: &MoveType) {
    match mv {
        MoveType::Single(u, v) => {
            state.flip_edge(*u, *v);
        }
        MoveType::Double((u1, v1), (u2, v2)) => {
            state.flip_edge(*u1, *v1);
            state.flip_edge(*u2, *v2);
        }
        MoveType::Multi(edges) => {
            for &(u, v) in edges {
                state.flip_edge(u, v);
            }
        }
        MoveType::VertexRewire(edges) => {
            // Apply as a batch of edge flips (self-inverse).
            for &(u, v) in edges {
                state.flip_edge(u, v);
            }
        }
        MoveType::PathSwap(edges) => {
            for &(u, v) in edges {
                state.flip_edge(u, v);
            }
        }
    }
}

/// Reverts a move by applying it again (all moves are self-inverse for edge flips).
pub fn revert_move<const N: usize>(state: &mut RamseyState<N>, mv: &MoveType) {
    // For simple flips, the inverse is the same operation
    // For vertex rewire, we need to store the old state
    apply_move(state, mv);
}

// ============================================================================
// C4-Aware Edge Selection
// ============================================================================

/// Finds the edge in a C4-free graph whose addition would create the fewest C4s.
pub fn best_edge_to_add<const N: usize, R: Rng>(
    state: &RamseyState<N>,
    candidates: &[(usize, usize)],
    rng: &mut R,
    sample_size: usize,
) -> (usize, usize) {
    if candidates.is_empty() {
        return random_pair::<N, R>(rng);
    }

    let samples = sample_size.min(candidates.len());
    let mut best_edge = candidates[0];
    let mut best_c4_count = usize::MAX;

    for _ in 0..samples {
        let idx = rng.random_range(0..candidates.len());
        let (u, v) = candidates[idx];

        if state.has_edge(u, v) {
            continue;
        }

        // Count C4s that would be created by adding (u, v)
        // A C4 is created for each pair of common neighbors of u and v
        let common = state.common_neighbor_count(u, v) as usize;
        let c4_created = if common >= 2 {
            common * (common - 1) / 2
        } else {
            0
        };

        if c4_created < best_c4_count {
            best_c4_count = c4_created;
            best_edge = (u, v);

            if c4_created == 0 {
                break; // Can't do better
            }
        }
    }

    best_edge
}

/// Finds the edge whose removal would destroy the most C4s.
///
/// Uses trial removal to accurately count C4 reduction.
pub fn best_edge_to_remove<const N: usize>(state: &RamseyState<N>) -> Option<(usize, usize)> {
    if state.c4_count() == 0 {
        return None;
    }

    let current_c4 = state.c4_count();
    let mut best_edge = None;
    let mut best_c4_destroyed = 0;

    // Use find_c4_edges to get candidate edges that are in C4s
    let c4_edges = state.find_c4_edges(100);

    if !c4_edges.is_empty() {
        // Try each C4 edge and see which removal destroys the most C4s
        let mut test_state = state.clone();

        for &(u, v) in &c4_edges {
            test_state.flip_edge(u, v); // Remove
            let new_c4 = test_state.c4_count();
            let destroyed = current_c4.saturating_sub(new_c4);
            test_state.flip_edge(u, v); // Restore

            if destroyed > best_c4_destroyed {
                best_c4_destroyed = destroyed;
                best_edge = Some((u, v));
            }
        }
    }

    // If no edges found via find_c4_edges, try all edges
    if best_edge.is_none() {
        let mut test_state = state.clone();

        for u in 0..N {
            for v in (u + 1)..N {
                if !state.has_edge(u, v) {
                    continue;
                }

                test_state.flip_edge(u, v); // Remove
                let new_c4 = test_state.c4_count();
                let destroyed = current_c4.saturating_sub(new_c4);
                test_state.flip_edge(u, v); // Restore

                if destroyed > best_c4_destroyed {
                    best_c4_destroyed = destroyed;
                    best_edge = Some((u, v));
                }
            }
        }
    }

    best_edge
}

// ============================================================================
// Look-Ahead Evaluation
// ============================================================================

/// Evaluates multiple candidate moves and returns the best one.
pub fn look_ahead_best_move<const N: usize, R: Rng, F>(
    state: &mut RamseyState<N>,
    candidates: &[(usize, usize)],
    rng: &mut R,
    sample_size: usize,
    mut eval_fn: F,
) -> (usize, usize)
where
    F: FnMut(&RamseyState<N>) -> usize,
{
    if candidates.is_empty() {
        return random_pair::<N, R>(rng);
    }

    let samples = sample_size.min(candidates.len());
    let mut best_edge = candidates[0];
    let mut best_energy = usize::MAX;

    for _ in 0..samples {
        let idx = rng.random_range(0..candidates.len());
        let (u, v) = candidates[idx];

        // Apply move
        state.flip_edge(u, v);

        // Evaluate
        let energy = eval_fn(state);

        // Revert
        state.flip_edge(u, v);

        if energy < best_energy {
            best_energy = energy;
            best_edge = (u, v);
        }
    }

    best_edge
}

/// Two-step look-ahead: evaluate (move1, move2) pairs.
pub fn two_step_look_ahead<const N: usize, R: Rng, F>(
    state: &mut RamseyState<N>,
    rng: &mut R,
    samples: usize,
    mut eval_fn: F,
) -> Vec<(usize, usize)>
where
    F: FnMut(&RamseyState<N>) -> usize,
{
    let mut best_sequence = Vec::new();
    let mut best_energy = eval_fn(state);

    for _ in 0..samples {
        let (u1, v1) = random_pair::<N, R>(rng);
        state.flip_edge(u1, v1);

        let (u2, v2) = random_pair::<N, R>(rng);
        state.flip_edge(u2, v2);

        let energy = eval_fn(state);

        // Revert
        state.flip_edge(u2, v2);
        state.flip_edge(u1, v1);

        if energy < best_energy {
            best_energy = energy;
            best_sequence = vec![(u1, v1), (u2, v2)];
        }
    }

    best_sequence
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generates a random vertex pair.
#[inline]
fn random_pair<const N: usize, R: Rng>(rng: &mut R) -> (usize, usize) {
    let u = rng.random_range(0..N);
    let mut v = rng.random_range(0..N);
    while v == u {
        v = rng.random_range(0..N);
    }
    (u, v)
}

/// Fisher-Yates shuffle.
fn shuffle_slice<T, R: Rng>(rng: &mut R, slice: &mut [T]) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
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
    fn single_move_is_reversible() {
        let mut rng = XorShiftRng::seed_from_u64(42);
        let mut state = RamseyState::<10>::new_random(&mut rng, 0.3);
        let original = state.clone();

        let mv = MoveType::Single(2, 5);
        apply_move(&mut state, &mv);
        apply_move(&mut state, &mv); // Revert

        assert_eq!(state.adj(), original.adj());
    }

    #[test]
    fn double_move_is_reversible() {
        let mut rng = XorShiftRng::seed_from_u64(123);
        let mut state = RamseyState::<10>::new_random(&mut rng, 0.3);
        let original = state.clone();

        let mv = MoveType::Double((1, 3), (5, 7));
        apply_move(&mut state, &mv);
        apply_move(&mut state, &mv);

        assert_eq!(state.adj(), original.adj());
    }

    #[test]
    fn multi_move_is_reversible() {
        let mut rng = XorShiftRng::seed_from_u64(456);
        let mut state = RamseyState::<10>::new_random(&mut rng, 0.3);
        let original = state.clone();

        let mv = MoveType::Multi(vec![(0, 1), (2, 3), (4, 5)]);
        apply_move(&mut state, &mv);
        apply_move(&mut state, &mv);

        assert_eq!(state.adj(), original.adj());
    }

    #[test]
    fn compound_generator_produces_valid_moves() {
        let mut rng = XorShiftRng::seed_from_u64(789);
        let state = RamseyState::<12>::new_random(&mut rng, 0.3);
        let mut oracle = IndependentSetOracle::<12>::new();

        let generator = CompoundMoveGenerator::<12>::new(0.5, 3);

        for _ in 0..100 {
            let mv = generator.generate(&state, &mut oracle, &mut rng, 4);
            let edges = mv.edges();

            // All edges should be valid
            for (u, v) in edges {
                assert!(u < 12);
                assert!(v < 12);
                assert_ne!(u, v);
            }
        }
    }

    #[test]
    fn best_edge_to_add_prefers_c4_free() {
        let mut rng = XorShiftRng::seed_from_u64(111);
        let state = RamseyState::<8>::empty();

        // Add some edges
        let mut state = state;
        state.flip_edge(0, 1);
        state.flip_edge(1, 2);
        state.flip_edge(2, 3);

        // Best edge should not create C4
        let candidates: Vec<_> = (0..8)
            .flat_map(|i| ((i + 1)..8).map(move |j| (i, j)))
            .filter(|&(u, v)| !state.has_edge(u, v))
            .collect();

        let (u, v) = best_edge_to_add(&state, &candidates, &mut rng, 50);

        // Verify adding this edge doesn't create C4
        let mut test_state = state.clone();
        test_state.flip_edge(u, v);
        assert_eq!(
            test_state.c4_count(),
            0,
            "Best edge should not create C4"
        );
    }

    #[test]
    fn best_edge_to_remove_targets_c4_edges() {
        // Create a graph with a C4: 0-1-2-3-0
        let mut adj = [0u64; 8];
        adj[0] = 0b1010; // 1, 3
        adj[1] = 0b0101; // 0, 2
        adj[2] = 0b1010; // 1, 3
        adj[3] = 0b0101; // 0, 2
        let state = RamseyState::<8>::from_adj(adj);

        assert!(state.c4_count() > 0);

        let best = best_edge_to_remove(&state);
        assert!(best.is_some());

        // Removing the best edge should reduce C4 count
        let (u, v) = best.unwrap();
        let mut new_state = state.clone();
        new_state.flip_edge(u, v);
        assert!(
            new_state.c4_count() < state.c4_count(),
            "Removing best edge should reduce C4 count"
        );
    }

    #[test]
    fn look_ahead_improves_decision() {
        let mut rng = XorShiftRng::seed_from_u64(222);
        let mut state = RamseyState::<10>::new_random(&mut rng, 0.3);

        let candidates: Vec<_> = (0..10)
            .flat_map(|i| ((i + 1)..10).map(move |j| (i, j)))
            .collect();

        let (u, v) = look_ahead_best_move(
            &mut state,
            &candidates,
            &mut rng,
            20,
            |s| s.c4_score_twice(),
        );

        // Just verify it returns a valid edge
        assert!(u < 10);
        assert!(v < 10);
        assert_ne!(u, v);
    }

    #[test]
    fn move_type_edges_returns_all_edges() {
        let mv = MoveType::Multi(vec![(0, 1), (2, 3), (4, 5)]);
        let edges = mv.edges();
        assert_eq!(edges.len(), 3);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(2, 3)));
        assert!(edges.contains(&(4, 5)));
    }
}


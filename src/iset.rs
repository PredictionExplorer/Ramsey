//! Exact independent-set detection for small graphs (currently \(N \le 64\)).
//!
//! We detect an independent set of size `k` in `G` by searching for a clique of size `k`
//! in the complement graph `\bar{G}`. The clique search uses a standard branch-and-bound
//! approach with greedy coloring to derive an upper bound for pruning (Tomita-style).

use crate::graph::all_bits;

#[inline(always)]
const fn bit(v: usize) -> u64 {
    1u64 << v
}

// ============================================================================
// IndependentSetOracle
// ============================================================================

/// Exact oracle for independent-set existence queries.
///
/// Internally reuses a stack buffer to avoid repeated allocations.
/// The oracle builds the complement graph on each query; for repeated queries
/// on the same graph, consider caching the complement externally.
#[derive(Clone, Debug)]
pub struct IndependentSetOracle<const N: usize> {
    stack: Vec<usize>,
    /// Cached complement adjacency (rebuilt on each query).
    comp: [u64; N],
}

impl<const N: usize> Default for IndependentSetOracle<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> IndependentSetOracle<N> {
    /// Creates a new oracle with preallocated scratch space.
    pub fn new() -> Self {
        debug_assert!(N <= 64, "This oracle assumes N <= 64");
        Self {
            stack: Vec::with_capacity(N),
            comp: [0u64; N],
        }
    }

    /// Builds the complement of `adj` into `self.comp`.
    #[inline]
    fn build_complement(&mut self, adj: &[u64; N]) {
        let mask = all_bits::<N>();
        for v in 0..N {
            // Complement: non-neighbors excluding self
            self.comp[v] = (!adj[v]) & mask & !bit(v);
        }
    }

    /// Returns `true` iff the graph contains an independent set of size `k`.
    #[inline]
    pub fn has_independent_set_of_size(&mut self, adj: &[u64; N], k: usize) -> bool {
        if k == 0 {
            return true;
        }
        if k > N {
            return false;
        }

        self.build_complement(adj);
        self.stack.clear();
        self.search_clique_k_exists(k, 0, all_bits::<N>())
    }

    /// If the graph contains an independent set of size `k`, writes one witness to `out`
    /// and returns `true`. Otherwise returns `false`.
    pub fn find_independent_set_of_size(
        &mut self,
        adj: &[u64; N],
        k: usize,
        out: &mut Vec<usize>,
    ) -> bool {
        out.clear();
        if k == 0 {
            return true;
        }
        if k > N {
            return false;
        }

        self.build_complement(adj);
        self.stack.clear();
        self.search_clique_k_find(k, 0, all_bits::<N>(), out)
    }

    /// Returns the independence number α(G) of the graph.
    ///
    /// This is expensive for large graphs; use sparingly.
    pub fn independence_number(&mut self, adj: &[u64; N]) -> usize {
        self.build_complement(adj);
        self.stack.clear();
        self.max_clique_size(0, all_bits::<N>())
    }

    fn search_clique_k_exists(&mut self, k: usize, size: usize, mut candidates: u64) -> bool {
        if size >= k {
            return true;
        }

        let remaining = candidates.count_ones() as usize;
        if size + remaining < k {
            return false;
        }

        // Fast path: if we need exactly as many as we have, just check they form a clique
        if size + remaining == k {
            return self.is_clique(candidates);
        }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(&self.comp, candidates, &mut order, &mut colors);

        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound < k {
                return false;
            }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & self.comp[v];
            if self.search_clique_k_exists(k, size + 1, next_candidates) {
                return true;
            }
            self.stack.pop();
            candidates &= !bit(v);
        }
        false
    }

    fn search_clique_k_find(
        &mut self,
        k: usize,
        size: usize,
        mut candidates: u64,
        out: &mut Vec<usize>,
    ) -> bool {
        if size >= k {
            out.clear();
            out.extend_from_slice(&self.stack);
            return true;
        }

        let remaining = candidates.count_ones() as usize;
        if size + remaining < k {
            return false;
        }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(&self.comp, candidates, &mut order, &mut colors);

        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound < k {
                return false;
            }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & self.comp[v];
            if self.search_clique_k_find(k, size + 1, next_candidates, out) {
                return true;
            }
            self.stack.pop();
            candidates &= !bit(v);
        }

        false
    }

    fn max_clique_size(&mut self, size: usize, mut candidates: u64) -> usize {
        if candidates == 0 {
            return size;
        }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(&self.comp, candidates, &mut order, &mut colors);

        let mut best = size;

        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound <= best {
                break; // Can't improve
            }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & self.comp[v];
            let found = self.max_clique_size(size + 1, next_candidates);
            if found > best {
                best = found;
            }
            self.stack.pop();
            candidates &= !bit(v);
        }

        best
    }

    /// Checks if all vertices in `mask` form a clique in the complement graph.
    #[inline]
    fn is_clique(&self, mask: u64) -> bool {
        let mut t = mask;
        while t != 0 {
            let v = t.trailing_zeros() as usize;
            t &= t - 1;
            // All other vertices in mask must be in v's neighborhood
            if (self.comp[v] & mask) != (mask & !bit(v)) {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Greedy coloring for clique bound
// ============================================================================

/// Greedy coloring used to bound the maximum clique size in `candidates`.
///
/// Produces `order` + `colors` such that `colors[i]` is a (greedy) coloring number, and thus
/// an upper bound on the size of a clique reachable from the suffix `order[..=i]`.
#[inline]
fn color_sort<const N: usize>(
    adj: &[u64; N],
    mut candidates: u64,
    order: &mut [usize; N],
    colors: &mut [u8; N],
) -> usize {
    let mut len = 0usize;
    let mut color: u8 = 0;

    while candidates != 0 {
        color = color.wrapping_add(1);

        let mut available = candidates;
        while available != 0 {
            let v = available.trailing_zeros() as usize;
            let v_mask = bit(v);

            order[len] = v;
            colors[len] = color;
            len += 1;

            candidates &= !v_mask;
            available &= !v_mask;
            // Build one color class as an independent set (in the candidate subgraph).
            available &= !adj[v];
        }
    }

    len
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    fn brute_alpha<const N: usize>(adj: &[u64; N]) -> usize {
        let mut best = 0usize;
        let total = 1u64 << N;

        for subset in 0..total {
            let sz = subset.count_ones() as usize;
            if sz <= best {
                continue;
            }
            if is_independent(adj, subset) {
                best = sz;
            }
        }
        best
    }

    fn is_independent<const N: usize>(adj: &[u64; N], subset: u64) -> bool {
        let mut t = subset;
        while t != 0 {
            let v = t.trailing_zeros() as usize;
            t &= t - 1;
            if (adj[v] & subset) != 0 {
                return false;
            }
        }
        true
    }

    #[test]
    fn oracle_matches_bruteforce_small_graphs() {
        const N: usize = 14;
        let mut rng = XorShiftRng::seed_from_u64(0xDEADBEEF);
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut witness = Vec::new();

        for _case in 0..40 {
            // Random symmetric graph.
            let mut adj = [0u64; N];
            for i in 0..N {
                for j in (i + 1)..N {
                    if rng.random_bool(0.45) {
                        adj[i] |= bit(j);
                        adj[j] |= bit(i);
                    }
                }
            }

            let alpha = brute_alpha(&adj);
            for k in 0..=N {
                let expect = alpha >= k;
                let got = oracle.find_independent_set_of_size(&adj, k, &mut witness);
                assert_eq!(expect, got, "mismatch for k={k} alpha={alpha}");
                if got {
                    assert_eq!(witness.len(), k);
                    // sanity: witness is independent
                    let mut mask = 0u64;
                    for &v in &witness {
                        mask |= bit(v);
                    }
                    assert!(is_independent(&adj, mask));
                }
            }
        }
    }

    #[test]
    fn independence_number_matches_brute_force() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xABCD);
        let mut oracle = IndependentSetOracle::<N>::new();

        for _ in 0..30 {
            let mut adj = [0u64; N];
            for i in 0..N {
                for j in (i + 1)..N {
                    if rng.random_bool(0.4) {
                        adj[i] |= bit(j);
                        adj[j] |= bit(i);
                    }
                }
            }

            let expected = brute_alpha(&adj);
            let got = oracle.independence_number(&adj);
            assert_eq!(expected, got, "independence_number mismatch");
        }
    }

    #[test]
    fn empty_graph_alpha_equals_n() {
        let adj = [0u64; 8];
        let mut oracle = IndependentSetOracle::<8>::new();
        assert_eq!(oracle.independence_number(&adj), 8);
        assert!(oracle.has_independent_set_of_size(&adj, 8));
    }

    #[test]
    fn complete_graph_alpha_equals_1() {
        // K4
        let adj: [u64; 4] = [0b1110, 0b1101, 0b1011, 0b0111];
        let mut oracle = IndependentSetOracle::<4>::new();
        assert_eq!(oracle.independence_number(&adj), 1);
        assert!(oracle.has_independent_set_of_size(&adj, 1));
        assert!(!oracle.has_independent_set_of_size(&adj, 2));
    }

    #[test]
    fn path_graph_alpha() {
        // Path: 0-1-2-3-4 (length 4, 5 vertices)
        // Alpha should be ceil(5/2) = 3 (take vertices 0, 2, 4)
        let adj: [u64; 5] = [
            0b00010, // 0 -- 1
            0b00101, // 1 -- 0, 2
            0b01010, // 2 -- 1, 3
            0b10100, // 3 -- 2, 4
            0b01000, // 4 -- 3
        ];
        let mut oracle = IndependentSetOracle::<5>::new();
        assert_eq!(oracle.independence_number(&adj), 3);
    }

    #[test]
    fn cycle_graph_alpha() {
        // C5: 0-1-2-3-4-0
        let adj: [u64; 5] = [
            0b10010, // 0 -- 1, 4
            0b00101, // 1 -- 0, 2
            0b01010, // 2 -- 1, 3
            0b10100, // 3 -- 2, 4
            0b01001, // 4 -- 3, 0
        ];
        let mut oracle = IndependentSetOracle::<5>::new();
        // C5 has alpha = 2
        assert_eq!(oracle.independence_number(&adj), 2);
    }

    #[test]
    fn bipartite_graph_alpha() {
        // K_{3,3} bipartite: {0,1,2} fully connected to {3,4,5}, no edges within parts.
        // Alpha = 3 (either part).
        let k33: [u64; 6] = [
            0b111000, // 0 -- 3,4,5
            0b111000, // 1 -- 3,4,5
            0b111000, // 2 -- 3,4,5
            0b000111, // 3 -- 0,1,2
            0b000111, // 4 -- 0,1,2
            0b000111, // 5 -- 0,1,2
        ];
        let mut oracle = IndependentSetOracle::<6>::new();
        assert_eq!(oracle.independence_number(&k33), 3);
        
        // Verify witness
        let mut witness = Vec::new();
        assert!(oracle.find_independent_set_of_size(&k33, 3, &mut witness));
        assert_eq!(witness.len(), 3);
        // All vertices should be from the same part (all < 3 or all >= 3)
        let all_left = witness.iter().all(|&v| v < 3);
        let all_right = witness.iter().all(|&v| v >= 3);
        assert!(all_left || all_right, "witness should be from one partition");
    }

    #[test]
    fn witness_is_valid_independent_set() {
        const N: usize = 10;
        let mut rng = XorShiftRng::seed_from_u64(0x1111);
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut witness = Vec::new();

        for _ in 0..50 {
            let mut adj = [0u64; N];
            for i in 0..N {
                for j in (i + 1)..N {
                    if rng.random_bool(0.35) {
                        adj[i] |= bit(j);
                        adj[j] |= bit(i);
                    }
                }
            }

            for k in 1..=N {
                if oracle.find_independent_set_of_size(&adj, k, &mut witness) {
                    assert_eq!(witness.len(), k, "witness has wrong size");
                    // Verify independence
                    let mut mask = 0u64;
                    for &v in &witness {
                        assert!(v < N, "witness contains invalid vertex");
                        mask |= bit(v);
                    }
                    assert!(is_independent(&adj, mask), "witness is not independent");
                }
            }
        }
    }

    #[test]
    fn has_is_matches_find_is() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0x2222);
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut witness = Vec::new();

        for _ in 0..30 {
            let mut adj = [0u64; N];
            for i in 0..N {
                for j in (i + 1)..N {
                    if rng.random_bool(0.4) {
                        adj[i] |= bit(j);
                        adj[j] |= bit(i);
                    }
                }
            }

            for k in 0..=N {
                let has = oracle.has_independent_set_of_size(&adj, k);
                let find = oracle.find_independent_set_of_size(&adj, k, &mut witness);
                assert_eq!(has, find, "has_is and find_is disagree for k={k}");
            }
        }
    }

    #[test]
    fn k_zero_always_succeeds() {
        let adj = [0b111u64, 0b111, 0b111]; // K3
        let mut oracle = IndependentSetOracle::<3>::new();
        assert!(oracle.has_independent_set_of_size(&adj, 0));
    }

    #[test]
    fn k_greater_than_n_fails() {
        let adj = [0u64; 5]; // Empty 5-vertex graph
        let mut oracle = IndependentSetOracle::<5>::new();
        assert!(!oracle.has_independent_set_of_size(&adj, 6));
        assert!(!oracle.has_independent_set_of_size(&adj, 100));
    }
}

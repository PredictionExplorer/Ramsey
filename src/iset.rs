//! Exact independent-set and clique detection for small graphs (currently \(N \le 64\)).
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

/// Exact oracle for independent-set and clique existence queries.
///
/// Internally reuses a stack buffer to avoid repeated allocations.
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
        if k == 0 { return true; }
        if k > N { return false; }

        self.build_complement(adj);
        let comp = self.comp;
        self.stack.clear();
        self.search_clique_k_exists(&comp, k, 0, all_bits::<N>())
    }

    /// Returns `true` iff the graph contains a clique of size `k`.
    #[inline]
    pub fn has_clique_of_size(&mut self, adj: &[u64; N], k: usize) -> bool {
        if k == 0 { return true; }
        if k > N { return false; }

        let adj_copy = *adj;
        self.stack.clear();
        self.search_clique_k_exists(&adj_copy, k, 0, all_bits::<N>())
    }

    /// Writes one independent set of size `k` to `out` if it exists.
    pub fn find_independent_set_of_size(
        &mut self,
        adj: &[u64; N],
        k: usize,
        out: &mut Vec<usize>,
    ) -> bool {
        out.clear();
        if k == 0 { return true; }
        if k > N { return false; }

        self.build_complement(adj);
        let comp = self.comp;
        self.stack.clear();
        self.search_clique_k_find(&comp, k, 0, all_bits::<N>(), out)
    }

    /// Writes one clique of size `k` to `out` if it exists.
    pub fn find_clique_of_size(
        &mut self,
        adj: &[u64; N],
        k: usize,
        out: &mut Vec<usize>,
    ) -> bool {
        out.clear();
        if k == 0 { return true; }
        if k > N { return false; }

        let adj_copy = *adj;
        self.stack.clear();
        self.search_clique_k_find(&adj_copy, k, 0, all_bits::<N>(), out)
    }

    /// Counts how many independent sets of size `k` exist, up to `limit`.
    pub fn count_independent_sets_of_size(
        &mut self,
        adj: &[u64; N],
        k: usize,
        limit: usize,
    ) -> usize {
        if k == 0 { return 1; }
        if k > N { return 0; }

        self.build_complement(adj);
        let comp = self.comp;
        self.stack.clear();
        let mut count = 0;
        self.search_clique_k_count(&comp, k, 0, all_bits::<N>(), &mut count, limit);
        count
    }

    /// Counts how many cliques of size `k` exist, up to `limit`.
    pub fn count_cliques_of_size(
        &mut self,
        adj: &[u64; N],
        k: usize,
        limit: usize,
    ) -> usize {
        if k == 0 { return 1; }
        if k > N { return 0; }

        let adj_copy = *adj;
        self.stack.clear();
        let mut count = 0;
        self.search_clique_k_count(&adj_copy, k, 0, all_bits::<N>(), &mut count, limit);
        count
    }

    fn search_clique_k_exists(&mut self, adj: &[u64; N], k: usize, size: usize, mut candidates: u64) -> bool {
        if size >= k { return true; }

        let remaining = candidates.count_ones() as usize;
        if size + remaining < k { return false; }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(adj, candidates, &mut order, &mut colors);

        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound < k { return false; }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & adj[v];
            if self.search_clique_k_exists(adj, k, size + 1, next_candidates) {
                return true;
            }
            self.stack.pop();
            candidates &= !bit(v);
        }
        false
    }

    fn search_clique_k_find(
        &mut self,
        adj: &[u64; N],
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
        if size + remaining < k { return false; }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(adj, candidates, &mut order, &mut colors);

        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound < k { return false; }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & adj[v];
            if self.search_clique_k_find(adj, k, size + 1, next_candidates, out) {
                return true;
            }
            self.stack.pop();
            candidates &= !bit(v);
        }

        false
    }

    fn search_clique_k_count(
        &mut self,
        adj: &[u64; N],
        k: usize,
        size: usize,
        mut candidates: u64,
        count: &mut usize,
        limit: usize,
    ) {
        if size >= k {
            *count += 1;
            return;
        }

        if *count >= limit { return; }

        let remaining = candidates.count_ones() as usize;
        if size + remaining < k { return; }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(adj, candidates, &mut order, &mut colors);

        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound < k { return; }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & adj[v];
            self.search_clique_k_count(adj, k, size + 1, next_candidates, count, limit);
            self.stack.pop();

            if *count >= limit { return; }
            candidates &= !bit(v);
        }
    }

    /// Returns the independence number α(G) of the graph.
    pub fn independence_number(&mut self, adj: &[u64; N]) -> usize {
        self.build_complement(adj);
        let comp = self.comp;
        self.stack.clear();
        self.max_clique_size(&comp, 0, all_bits::<N>())
    }

    /// Returns the clique number ω(G) of the graph.
    pub fn clique_number(&mut self, adj: &[u64; N]) -> usize {
        let adj_copy = *adj;
        self.stack.clear();
        self.max_clique_size(&adj_copy, 0, all_bits::<N>())
    }

    fn max_clique_size(&mut self, adj: &[u64; N], size: usize, mut candidates: u64) -> usize {
        if candidates == 0 { return size; }

        let mut order = [0usize; N];
        let mut colors = [0u8; N];
        let len = color_sort(adj, candidates, &mut order, &mut colors);

        let mut best = size;
        for idx in (0..len).rev() {
            let color_bound = colors[idx] as usize;
            if size + color_bound <= best { break; }

            let v = order[idx];
            self.stack.push(v);
            let next_candidates = candidates & adj[v];
            let found = self.max_clique_size(adj, size + 1, next_candidates);
            if found > best { best = found; }
            self.stack.pop();
            candidates &= !bit(v);
        }
        best
    }
}

// ============================================================================
// Greedy coloring for clique bound
// ============================================================================

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
            if sz <= best { continue; }
            if is_independent(adj, subset) { best = sz; }
        }
        best
    }

    fn brute_omega<const N: usize>(adj: &[u64; N]) -> usize {
        let mut best = 0usize;
        let total = 1u64 << N;
        for subset in 0..total {
            let sz = subset.count_ones() as usize;
            if sz <= best { continue; }
            if is_clique(adj, subset) { best = sz; }
        }
        best
    }

    fn is_independent<const N: usize>(adj: &[u64; N], subset: u64) -> bool {
        let mut t = subset;
        while t != 0 {
            let v = t.trailing_zeros() as usize;
            t &= t - 1;
            if (adj[v] & subset) != 0 { return false; }
        }
        true
    }

    fn is_clique<const N: usize>(adj: &[u64; N], subset: u64) -> bool {
        let mut t = subset;
        while t != 0 {
            let v = t.trailing_zeros() as usize;
            t &= t - 1;
            if (adj[v] & subset) != (subset & !bit(v)) { return false; }
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
                    let mut mask = 0u64;
                    for &v in &witness { mask |= bit(v); }
                    assert!(is_independent(&adj, mask));
                }
            }
        }
    }

    #[test]
    fn clique_matches_independent_set_on_complement() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xAAAA);
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut adj = [0u64; N];
        let mut comp = [0u64; N];

        for i in 0..N {
            for j in (i + 1)..N {
                if rng.random_bool(0.5) {
                    adj[i] |= bit(j);
                    adj[j] |= bit(i);
                } else {
                    comp[i] |= bit(j);
                    comp[j] |= bit(i);
                }
            }
        }

        for k in 1..=N {
            let has_clique = oracle.has_clique_of_size(&adj, k);
            let has_iset = oracle.has_independent_set_of_size(&comp, k);
            assert_eq!(has_clique, has_iset, "Clique/IS duality failed for k={k}");
        }
    }

    #[test]
    fn count_independent_sets_matches_brute_force() {
        const N: usize = 8;
        let mut rng = XorShiftRng::seed_from_u64(0x12345);
        let mut oracle = IndependentSetOracle::<N>::new();

        for _ in 0..20 {
            let mut adj = [0u64; N];
            for i in 0..N {
                for j in (i + 1)..N {
                    if rng.random_bool(0.4) {
                        adj[i] |= bit(j);
                        adj[j] |= bit(i);
                    }
                }
            }

            for k in 1..=4 {
                let mut brute_count = 0;
                let total = 1u64 << N;
                for mask in 0..total {
                    if mask.count_ones() as usize == k && is_independent(&adj, mask) {
                        brute_count += 1;
                    }
                }
                let oracle_count = oracle.count_independent_sets_of_size(&adj, k, 1000);
                assert_eq!(brute_count, oracle_count, "IS count mismatch for k={k}");
            }
        }
    }

    #[test]
    fn find_clique_returns_valid_clique() {
        const N: usize = 16;
        let mut rng = XorShiftRng::seed_from_u64(0xBCDE);
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut witness = Vec::new();

        for _ in 0..50 {
            let mut adj = [0u64; N];
            for i in 0..N {
                for j in (i + 1)..N {
                    if rng.random_bool(0.6) {
                        adj[i] |= bit(j);
                        adj[j] |= bit(i);
                    }
                }
            }

            let omega = brute_omega(&adj);
            if oracle.find_clique_of_size(&adj, omega, &mut witness) {
                assert_eq!(witness.len(), omega);
                let mut mask = 0u64;
                for &v in &witness { mask |= bit(v); }
                assert!(is_clique(&adj, mask), "Witness is not a clique");
            }
        }
    }

    #[test]
    fn oracle_base_cases() {
        let adj = [0u64; 10]; // Empty graph
        let mut oracle = IndependentSetOracle::<10>::new();
        
        // k=1 always true for any non-empty set of vertices
        assert!(oracle.has_independent_set_of_size(&adj, 1));
        assert!(oracle.has_clique_of_size(&[!0u64; 10], 1));
        
        // k=0 always true
        assert!(oracle.has_independent_set_of_size(&adj, 0));
        
        // k > N always false
        assert!(!oracle.has_independent_set_of_size(&adj, 11));
    }

    #[test]
    fn oracle_complete_graph() {
        let mut adj = [0u64; 8];
        let mask = all_bits::<8>();
        for i in 0..8 { adj[i] = mask & !bit(i); } // K8
        
        let mut oracle = IndependentSetOracle::<8>::new();
        assert_eq!(oracle.clique_number(&adj), 8);
        assert_eq!(oracle.independence_number(&adj), 1);
    }

    #[test]
    fn oracle_ramsey_r33_limit() {
        // R(3,3) = 6. This means any graph on 6 vertices MUST have a K3 or an IS3.
        // We test that for N=5, there exists a graph with no K3 and no IS3 (the C5 cycle).
        let mut adj = [0u64; 5];
        let edges = [(0,1), (1,2), (2,3), (3,4), (4,0)];
        for (u,v) in edges {
            adj[u] |= 1 << v;
            adj[v] |= 1 << u;
        }
        
        let mut oracle = IndependentSetOracle::<5>::new();
        assert!(!oracle.has_clique_of_size(&adj, 3));
        assert!(!oracle.has_independent_set_of_size(&adj, 3));
    }

    #[test]
    fn count_cliques_gradient_test() {
        const N: usize = 10;
        let mut oracle = IndependentSetOracle::<N>::new();
        let mut adj = [0u64; N];
        
        // Empty graph has 0 cliques of size 3
        assert_eq!(oracle.count_cliques_of_size(&adj, 3, 100), 0);
        
        // Adding one triangle (0,1,2)
        adj[0] |= 0b110; adj[1] |= 0b101; adj[2] |= 0b011;
        assert_eq!(oracle.count_cliques_of_size(&adj, 3, 100), 1);
        
        // Adding another triangle (3,4,5)
        adj[3] |= 0b110000; adj[4] |= 0b101000; adj[5] |= 0b011000;
        assert_eq!(oracle.count_cliques_of_size(&adj, 3, 100), 2);
    }

    #[test]
    fn test_ramsey_r34_limit() {
        // R(3,4) = 9. Any graph on 9 vertices must have K3 or IS4.
        // We verify that for any random 9-vertex graph, the oracle finds one or the other.
        let mut rng = XorShiftRng::seed_from_u64(42);
        let mut oracle = IndependentSetOracle::<9>::new();
        
        for _ in 0..100 {
            let mut adj = [0u64; 9];
            for i in 0..9 {
                for j in i+1..9 {
                    if rng.random_bool(0.5) {
                        adj[i] |= 1 << j;
                        adj[j] |= 1 << i;
                    }
                }
            }
            
            let has_k3 = oracle.has_clique_of_size(&adj, 3);
            let has_is4 = oracle.has_independent_set_of_size(&adj, 4);
            assert!(has_k3 || has_is4, "R(3,4) violation! Any 9-vertex graph must have K3 or IS4.");
        }
    }
}

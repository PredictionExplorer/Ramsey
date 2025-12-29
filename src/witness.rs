//! Witness Pool: Lazy constraint learning for independent set detection.
//!
//! Instead of re-solving the NP-hard independent set detection every iteration,
//! we maintain a pool of known IS witnesses. A witness is valid until an edge
//! is added that breaks it. This converts O(expensive) exact search into
//! O(pool_size × k²) cheap validity checks.
//!
//! Key insight: When we find an IS of size k, we learn a constraint:
//! "at least one edge must exist among these k vertices". We can reuse
//! this constraint until it's satisfied.

use rand::Rng;
use std::sync::Arc;

use crate::lockfree::LockFreeWitnessPool;

// ============================================================================
// Witness Type
// ============================================================================

/// A witness is an independent set represented as a bitset.
/// If this bitset is still independent in the current graph, the constraint is violated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Witness {
    /// Bitset of vertices in the independent set.
    pub vertices: u64,
    /// Size of the independent set.
    pub size: u8,
}

impl Witness {
    /// Creates a new witness from a list of vertices.
    #[inline]
    pub fn from_vertices(verts: &[usize]) -> Self {
        let mut vertices = 0u64;
        for &v in verts {
            debug_assert!(v < 64, "Vertex index out of range");
            vertices |= 1u64 << v;
        }
        Self {
            vertices,
            size: verts.len() as u8,
        }
    }

    /// Creates a witness directly from a bitset.
    #[inline]
    pub const fn from_bitset(vertices: u64) -> Self {
        Self {
            vertices,
            size: vertices.count_ones() as u8,
        }
    }

    /// Checks if this witness is still valid (still an independent set) in the graph.
    ///
    /// A witness is valid iff no two vertices in it are adjacent.
    /// Complexity: O(k) where k is the witness size.
    #[inline]
    pub fn is_valid<const N: usize>(&self, adj: &[u64; N]) -> bool {
        let mut remaining = self.vertices;
        while remaining != 0 {
            let v = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            // Check if v has any neighbors in the witness
            if (adj[v] & self.vertices) != 0 {
                return false;
            }
        }
        true
    }

    /// Returns whether adding edge (u,v) would break this witness.
    /// An edge breaks a witness iff both endpoints are in the witness.
    #[inline]
    pub fn would_be_broken_by(&self, u: usize, v: usize) -> bool {
        let mask = (1u64 << u) | (1u64 << v);
        (self.vertices & mask) == mask
    }

    /// Returns whether this witness contains vertex v.
    #[inline]
    pub fn contains(&self, v: usize) -> bool {
        (self.vertices & (1u64 << v)) != 0
    }

    /// Returns the vertices as a Vec.
    pub fn to_vec(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.size as usize);
        let mut remaining = self.vertices;
        while remaining != 0 {
            let v = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            result.push(v);
        }
        result
    }

    /// Computes the Hamming distance to another witness.
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.vertices ^ other.vertices).count_ones()
    }
}

// ============================================================================
// Local Witness Pool (per-worker)
// ============================================================================

/// Pool of IS witnesses with coverage-based move selection.
///
/// This is the core data structure for lazy constraint learning.
/// Each worker maintains its own pool and periodically syncs with the shared pool.
#[derive(Clone, Debug)]
pub struct WitnessPool {
    /// Active witnesses.
    witnesses: Vec<Witness>,
    /// Maximum pool size.
    max_size: usize,
    /// Minimum witness size to track (typically k_target).
    min_size: usize,
    /// Number of witnesses added since last prune.
    adds_since_prune: usize,
    /// Prune interval.
    prune_interval: usize,
}

impl WitnessPool {
    /// Creates a new witness pool.
    pub fn new(max_size: usize, min_size: usize) -> Self {
        Self {
            witnesses: Vec::with_capacity(max_size),
            max_size,
            min_size,
            adds_since_prune: 0,
            prune_interval: 100,
        }
    }

    /// Adds a witness to the pool if not duplicate and large enough.
    pub fn add(&mut self, witness: Witness) -> bool {
        if (witness.size as usize) < self.min_size {
            return false;
        }

        // Check for duplicates or subsets
        for existing in &self.witnesses {
            if existing.vertices == witness.vertices {
                return false; // Duplicate
            }
            // If new witness is a superset of existing, skip (existing is stronger)
            if (existing.vertices & witness.vertices) == existing.vertices
                && existing.size < witness.size
            {
                return false;
            }
        }

        // Remove witnesses that are supersets of the new one (new is stronger)
        self.witnesses
            .retain(|w| (w.vertices & witness.vertices) != witness.vertices || w.size <= witness.size);

        if self.witnesses.len() >= self.max_size {
            // Evict oldest (FIFO)
            self.witnesses.remove(0);
        }

        self.witnesses.push(witness);
        self.adds_since_prune += 1;
        true
    }

    /// Adds a witness from vertex list.
    pub fn add_from_vertices(&mut self, verts: &[usize]) -> bool {
        self.add(Witness::from_vertices(verts))
    }

    /// Removes witnesses that are no longer valid (broken by added edges).
    pub fn prune_invalid<const N: usize>(&mut self, adj: &[u64; N]) {
        self.witnesses.retain(|w| w.is_valid(adj));
        self.adds_since_prune = 0;
    }

    /// Conditionally prunes if enough operations have occurred.
    pub fn maybe_prune<const N: usize>(&mut self, adj: &[u64; N]) {
        if self.adds_since_prune >= self.prune_interval {
            self.prune_invalid(adj);
        }
    }

    /// Returns number of witnesses currently valid.
    pub fn count_valid<const N: usize>(&self, adj: &[u64; N]) -> usize {
        self.witnesses.iter().filter(|w| w.is_valid(adj)).count()
    }

    /// Returns total witness count.
    #[inline]
    pub fn len(&self) -> usize {
        self.witnesses.len()
    }

    /// Returns true if pool is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.witnesses.is_empty()
    }

    /// Clears all witnesses.
    pub fn clear(&mut self) {
        self.witnesses.clear();
        self.adds_since_prune = 0;
    }

    /// Returns the edge (u,v) that would break the most valid witnesses.
    ///
    /// Only considers non-edges that don't create too many C4s.
    /// This is the "hitting set" heuristic for IS elimination.
    ///
    /// Returns (edge, hits_count) or None if no good edge found.
    pub fn best_edge_by_coverage<const N: usize>(
        &self,
        adj: &[u64; N],
        max_c4_created: usize,
    ) -> Option<((usize, usize), usize)> {
        let best = self
            .edges_by_coverage::<N>(adj, max_c4_created, 1)
            .into_iter()
            .next();
        best.map(|(e, hits, _c4)| (e, hits))
    }

    /// Returns edges sorted by coverage (most hits first), filtered by C4 constraint.
    ///
    /// Useful for look-ahead or k-best selection.
    pub fn edges_by_coverage<const N: usize>(
        &self,
        adj: &[u64; N],
        max_c4_created: usize,
        limit: usize,
    ) -> Vec<((usize, usize), usize, usize)> {
        if self.witnesses.is_empty() || limit == 0 {
            return Vec::new();
        }

        // ---------------------------------------------------------------------
        // Fast coverage computation: O(sum_w choose2(|w|)) instead of O(N^2 * |W|)
        //
        // For each valid witness W, every pair (u,v) in W is a non-edge that would
        // break W if added. We count how many witnesses include each pair.
        // ---------------------------------------------------------------------

        let mut pair_hits = [[0u16; N]; N];
        let mut any_valid = false;

        // Reusable vertex buffer to avoid per-witness Vec allocations.
        let mut verts = [0usize; 64];

        for w in &self.witnesses {
            if !w.is_valid(adj) {
                continue;
            }
            any_valid = true;

            // Extract vertices in ascending order.
            let mut len = 0usize;
            let mut t = w.vertices;
            while t != 0 {
                let v = t.trailing_zeros() as usize;
                t &= t - 1;
                verts[len] = v;
                len += 1;
            }

            // Count all pairs inside this witness (all are non-edges in a valid witness).
            for i in 0..len {
                for j in (i + 1)..len {
                    let u = verts[i];
                    let v = verts[j];
                    // Saturating add to avoid u16 overflow in pathological cases.
                    pair_hits[u][v] = pair_hits[u][v].saturating_add(1);
                }
            }
        }

        if !any_valid {
            return Vec::new();
        }

        let mut candidates: Vec<((usize, usize), usize, usize)> = Vec::new();

        for u in 0..N {
            for v in (u + 1)..N {
                let hits = pair_hits[u][v] as usize;
                if hits == 0 {
                    continue;
                }
                // Safety check: this should hold because we only counted pairs inside
                // valid witnesses, but keep it to avoid surprises with malformed witnesses.
                if (adj[u] & (1u64 << v)) != 0 {
                    continue;
                }

                // IMPORTANT: A non-edge (u,v) can create a C4 even when u and v have <= 1 common
                // neighbor (e.g., adding a chord to a length-3 path). For strict C4-free moves
                // (`max_c4_created == 0`) we must enforce the correct incremental safety condition.
                let c4_created = if max_c4_created == 0 {
                    if !is_c4_safe_to_add_from_adj::<N>(adj, u, v) {
                        continue;
                    }
                    0
                } else {
                    // Heuristic estimate used only when callers explicitly allow C4 creation.
                    let common = (adj[u] & adj[v]).count_ones() as usize;
                    if common >= 2 {
                        common * (common - 1) / 2
                    } else {
                        0
                    }
                };
                if c4_created > max_c4_created {
                    continue;
                }

                candidates.push(((u, v), hits, c4_created));
            }
        }

        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
        candidates.truncate(limit);
        candidates
    }

    /// Returns all C4-safe non-edges (edges that create 0 C4s if added).
    pub fn c4_safe_non_edges<const N: usize>(&self, adj: &[u64; N]) -> Vec<(usize, usize)> {
        let mut result = Vec::new();

        for u in 0..N {
            for v in (u + 1)..N {
                if (adj[u] & (1u64 << v)) != 0 {
                    continue;
                }

                if is_c4_safe_to_add_from_adj::<N>(adj, u, v) {
                    result.push((u, v));
                }
            }
        }

        result
    }

    /// Samples a random valid witness.
    pub fn sample_valid<const N: usize, R: Rng>(
        &self,
        adj: &[u64; N],
        rng: &mut R,
    ) -> Option<Witness> {
        let valid: Vec<_> = self.witnesses.iter().filter(|w| w.is_valid(adj)).collect();
        if valid.is_empty() {
            None
        } else {
            Some(*valid[rng.random_range(0..valid.len())])
        }
    }

    /// Returns an iterator over all witnesses.
    pub fn iter(&self) -> impl Iterator<Item = &Witness> {
        self.witnesses.iter()
    }

    /// Returns witnesses as a Vec (for snapshotting).
    pub fn snapshot(&self) -> Vec<Witness> {
        self.witnesses.clone()
    }

    /// Merges witnesses from another pool (deduplicating).
    pub fn merge(&mut self, other: &[Witness]) {
        for w in other {
            self.add(*w);
        }
    }
}

/// Returns true iff adding the non-edge (u,v) would create **zero** C4s, using only adjacency
/// bitsets.
///
/// This matches `RamseyState::is_c4_safe_to_add` but is available in this module where we only
/// have `adj`.
#[inline]
fn is_c4_safe_to_add_from_adj<const N: usize>(adj: &[u64; N], u: usize, v: usize) -> bool {
    debug_assert!(u < N && v < N && u != v);
    debug_assert!((adj[u] & (1u64 << v)) == 0, "caller must pass a non-edge");

    // Safe iff:
    // - For every w in N(v): N(u) ∩ N(w) is empty
    // - For every w in N(u): N(v) ∩ N(w) is empty
    let mut t = adj[v];
    while t != 0 {
        let w = t.trailing_zeros() as usize;
        t &= t - 1;
        if (adj[u] & adj[w]) != 0 {
            return false;
        }
    }
    let mut t = adj[u];
    while t != 0 {
        let w = t.trailing_zeros() as usize;
        t &= t - 1;
        if (adj[v] & adj[w]) != 0 {
            return false;
        }
    }
    true
}

// ============================================================================
// Shared Witness Pool (cross-worker learning)
// ============================================================================

/// Thread-safe shared witness pool for cross-worker constraint learning.
///
/// Workers periodically:
/// 1. Submit new witnesses they discover
/// 2. Fetch a snapshot to augment their local pool
///
/// This enables "constraint parallelism": when one worker discovers a witness,
/// all workers can benefit from it.
#[derive(Clone)]
pub struct SharedWitnessPool {
    inner: Arc<LockFreeWitnessPool>,
}

impl SharedWitnessPool {
    /// Creates a new shared witness pool.
    pub fn new(max_size: usize, min_size: usize) -> Self {
        Self {
            inner: LockFreeWitnessPool::new(max_size, min_size),
        }
    }

    /// Adds a witness to the shared pool.
    pub fn add(&self, witness: Witness) -> bool {
        self.inner.try_add(witness.vertices, witness.size)
    }

    /// Adds witnesses from vertex list.
    pub fn add_from_vertices(&self, verts: &[usize]) -> bool {
        self.add(Witness::from_vertices(verts))
    }

    /// Returns the number of witnesses.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a snapshot of current witnesses for local use.
    pub fn snapshot(&self) -> Vec<Witness> {
        self.inner
            .snapshot()
            .into_iter()
            .map(|e| Witness {
                vertices: e.vertices,
                size: e.size,
            })
            .collect()
    }

    /// Bulk-adds witnesses (more efficient than individual adds).
    pub fn add_batch(&self, witnesses: &[Witness]) {
        for w in witnesses {
            let _ = self.add(*w);
        }
    }

    /// Prunes invalid witnesses.
    pub fn prune_invalid<const N: usize>(&self, adj: &[u64; N]) {
        self.inner.prune_invalid(adj);
    }
}

impl Default for SharedWitnessPool {
    fn default() -> Self {
        Self::new(1000, 1)
    }
}

// ============================================================================
// Coverage Statistics
// ============================================================================

/// Statistics about witness coverage for monitoring.
#[derive(Clone, Debug, Default)]
pub struct CoverageStats {
    /// Total witnesses in pool.
    pub total_witnesses: usize,
    /// Witnesses currently valid.
    pub valid_witnesses: usize,
    /// Best single-edge hit count.
    pub best_single_hit: usize,
    /// Number of C4-safe non-edges.
    pub c4_safe_edges: usize,
}

impl CoverageStats {
    /// Computes coverage statistics for the current state.
    pub fn compute<const N: usize>(pool: &WitnessPool, adj: &[u64; N]) -> Self {
        let total_witnesses = pool.len();
        let valid_witnesses = pool.count_valid(adj);

        let best_single_hit = pool
            .best_edge_by_coverage(adj, 0)
            .map(|(_, hits)| hits)
            .unwrap_or(0);

        let c4_safe_edges = pool.c4_safe_non_edges(adj).len();

        Self {
            total_witnesses,
            valid_witnesses,
            best_single_hit,
            c4_safe_edges,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_from_vertices_basic() {
        let w = Witness::from_vertices(&[0, 2, 5]);
        assert_eq!(w.size, 3);
        assert!(w.contains(0));
        assert!(w.contains(2));
        assert!(w.contains(5));
        assert!(!w.contains(1));
        assert!(!w.contains(3));
    }

    #[test]
    fn witness_to_vec_roundtrip() {
        let verts = vec![1, 3, 7, 10];
        let w = Witness::from_vertices(&verts);
        let mut recovered = w.to_vec();
        recovered.sort();
        assert_eq!(recovered, verts);
    }

    #[test]
    fn witness_is_valid_empty_graph() {
        let adj = [0u64; 10];
        let w = Witness::from_vertices(&[0, 1, 2, 3]);
        assert!(w.is_valid(&adj), "Any set is independent in empty graph");
    }

    #[test]
    fn witness_is_valid_detects_edge() {
        let mut adj = [0u64; 10];
        // Add edge 1-2
        adj[1] |= 1 << 2;
        adj[2] |= 1 << 1;

        let w1 = Witness::from_vertices(&[0, 3, 4]); // No edge among these
        let w2 = Witness::from_vertices(&[1, 2, 4]); // Has edge 1-2

        assert!(w1.is_valid(&adj));
        assert!(!w2.is_valid(&adj));
    }

    #[test]
    fn witness_would_be_broken_by() {
        let w = Witness::from_vertices(&[1, 3, 5, 7]);

        // Edge between two vertices in witness
        assert!(w.would_be_broken_by(1, 3));
        assert!(w.would_be_broken_by(3, 7));
        assert!(w.would_be_broken_by(5, 1));

        // Edge with only one vertex in witness
        assert!(!w.would_be_broken_by(1, 2));
        assert!(!w.would_be_broken_by(0, 3));

        // Edge with no vertices in witness
        assert!(!w.would_be_broken_by(0, 2));
    }

    #[test]
    fn pool_add_deduplicates() {
        let mut pool = WitnessPool::new(100, 2);

        let w = Witness::from_vertices(&[0, 1, 2]);
        assert!(pool.add(w));
        assert!(!pool.add(w)); // Duplicate
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pool_add_respects_min_size() {
        let mut pool = WitnessPool::new(100, 4);

        let small = Witness::from_vertices(&[0, 1, 2]); // Size 3 < min 4
        let large = Witness::from_vertices(&[0, 1, 2, 3]); // Size 4 = min

        assert!(!pool.add(small));
        assert!(pool.add(large));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pool_prune_removes_invalid() {
        let mut pool = WitnessPool::new(100, 2);
        pool.add(Witness::from_vertices(&[0, 1, 2]));
        pool.add(Witness::from_vertices(&[3, 4, 5]));
        pool.add(Witness::from_vertices(&[0, 3, 6]));

        assert_eq!(pool.len(), 3);

        // Add edge 0-1, invalidating first witness
        let mut adj = [0u64; 10];
        adj[0] |= 1 << 1;
        adj[1] |= 1 << 0;

        pool.prune_invalid(&adj);

        assert_eq!(pool.len(), 2);
        assert_eq!(pool.count_valid(&adj), 2);
    }

    #[test]
    fn pool_best_edge_by_coverage_basic() {
        let mut pool = WitnessPool::new(100, 2);

        // Three witnesses all containing vertices 0 and 1
        pool.add(Witness::from_vertices(&[0, 1, 2]));
        pool.add(Witness::from_vertices(&[0, 1, 3]));
        pool.add(Witness::from_vertices(&[0, 1, 4]));

        let adj = [0u64; 10]; // Empty graph

        let result = pool.best_edge_by_coverage::<10>(&adj, 100);
        assert!(result.is_some());

        let ((u, v), hits) = result.unwrap();
        // Edge (0,1) should hit all 3 witnesses
        assert!((u == 0 && v == 1) || (u == 1 && v == 0));
        assert_eq!(hits, 3);
    }

    #[test]
    fn pool_best_edge_respects_c4_constraint() {
        use crate::graph::RamseyState;

        // Build a C4-free tree where adding edge (0,3) would create a 4-cycle:
        // 0-1-2-3 plus leaves attached to 1 and 2.
        //
        // Edges: (0,1), (1,2), (2,3), (1,4), (1,5), (2,6), (2,7)
        let mut adj = [0u64; 8];
        let edges = [(0, 1), (1, 2), (2, 3), (1, 4), (1, 5), (2, 6), (2, 7)];
        for (u, v) in edges {
            adj[u] |= 1u64 << v;
            adj[v] |= 1u64 << u;
        }
        let state = RamseyState::<8>::from_adj(adj);
        assert_eq!(state.c4_score_twice(), 0, "graph must start C4-free");
        assert!(!state.is_c4_safe_to_add(0, 3), "0-3 chord should create a C4");

        // Create multiple distinct witnesses that all contain {0,3} plus different leaves.
        // Edge (0,3) would hit all of them, but must be rejected as not C4-safe.
        let mut pool = WitnessPool::new(100, 2);
        pool.add(Witness::from_vertices(&[0, 3, 4]));
        pool.add(Witness::from_vertices(&[0, 3, 5]));
        pool.add(Witness::from_vertices(&[0, 3, 6]));
        pool.add(Witness::from_vertices(&[0, 3, 7]));

        let result = pool.best_edge_by_coverage::<8>(&adj, 0);
        assert!(result.is_some(), "should find a safe witness-breaking edge");
        let ((u, v), hits) = result.unwrap();
        assert!(hits >= 1);
        assert_ne!((u, v), (0, 3), "must reject non C4-safe chord (0,3)");
        assert!(state.is_c4_safe_to_add(u, v), "returned edge must be C4-safe");
    }

    #[test]
    fn pool_c4_safe_non_edges() {
        use crate::graph::RamseyState;

        let mut adj = [0u64; 5];
        // Path: 0-1-2-3-4
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
        for (u, v) in edges {
            adj[u] |= 1u64 << v;
            adj[v] |= 1u64 << u;
        }

        let state = RamseyState::<5>::from_adj(adj);
        assert_eq!(state.c4_score_twice(), 0);

        // Brute-force compute the true C4-safe non-edges using the incremental criterion.
        let mut expected = Vec::new();
        for u in 0..5 {
            for v in (u + 1)..5 {
                if !state.has_edge(u, v) && state.is_c4_safe_to_add(u, v) {
                    expected.push((u, v));
                }
            }
        }

        let pool = WitnessPool::new(100, 2);
        let mut safe = pool.c4_safe_non_edges(&adj);
        safe.sort();
        expected.sort();
        assert_eq!(safe, expected);
    }

    #[test]
    fn pool_eviction_fifo() {
        let mut pool = WitnessPool::new(3, 2);

        pool.add(Witness::from_vertices(&[0, 1]));
        pool.add(Witness::from_vertices(&[2, 3]));
        pool.add(Witness::from_vertices(&[4, 5]));
        assert_eq!(pool.len(), 3);

        // Adding fourth should evict first
        pool.add(Witness::from_vertices(&[6, 7]));
        assert_eq!(pool.len(), 3);

        // First witness should be gone
        let has_first = pool.iter().any(|w| w.vertices == Witness::from_vertices(&[0, 1]).vertices);
        assert!(!has_first);
    }

    #[test]
    fn shared_pool_thread_safety() {
        use std::thread;

        let pool = SharedWitnessPool::new(100, 2);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let p = pool.clone();
                thread::spawn(move || {
                    for j in 0..25 {
                        p.add(Witness::from_vertices(&[i * 10 + j, i * 10 + j + 1]));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should have some witnesses (exact count depends on timing)
        assert!(pool.len() > 0);
        assert!(pool.len() <= 100);
    }

    #[test]
    fn shared_pool_snapshot() {
        let pool = SharedWitnessPool::new(100, 2);

        pool.add(Witness::from_vertices(&[0, 1, 2]));
        pool.add(Witness::from_vertices(&[3, 4, 5]));

        let snap = pool.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn witness_hamming_distance() {
        let w1 = Witness::from_vertices(&[0, 1, 2]);
        let w2 = Witness::from_vertices(&[0, 1, 3]);

        // Differ by vertices 2 and 3
        assert_eq!(w1.hamming_distance(&w2), 2);

        // Same witness has distance 0
        assert_eq!(w1.hamming_distance(&w1), 0);
    }

    #[test]
    fn coverage_stats_compute() {
        let mut pool = WitnessPool::new(100, 2);
        pool.add(Witness::from_vertices(&[0, 1, 2]));
        pool.add(Witness::from_vertices(&[0, 1, 3]));

        let adj = [0u64; 10];
        let stats = CoverageStats::compute(&pool, &adj);

        assert_eq!(stats.total_witnesses, 2);
        assert_eq!(stats.valid_witnesses, 2);
        assert!(stats.best_single_hit >= 2); // Edge (0,1) hits both
    }

    #[test]
    fn pool_edges_by_coverage_sorted() {
        let mut pool = WitnessPool::new(100, 2);

        // Witness that only (0,1) hits
        pool.add(Witness::from_vertices(&[0, 1, 9]));
        // Two witnesses that (2,3) hits
        pool.add(Witness::from_vertices(&[2, 3, 8]));
        pool.add(Witness::from_vertices(&[2, 3, 7]));

        let adj = [0u64; 10];
        let edges = pool.edges_by_coverage::<10>(&adj, 100, 10);

        assert!(edges.len() >= 2);
        // First should be (2,3) with 2 hits
        assert_eq!(edges[0].0, (2, 3));
        assert_eq!(edges[0].1, 2);
    }

    #[test]
    fn pool_sample_valid_returns_valid() {
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;

        let mut pool = WitnessPool::new(100, 2);
        pool.add(Witness::from_vertices(&[0, 1, 2]));
        pool.add(Witness::from_vertices(&[3, 4, 5]));

        let mut adj = [0u64; 10];
        // Invalidate first witness
        adj[0] |= 1 << 1;
        adj[1] |= 1 << 0;

        let mut rng = XorShiftRng::seed_from_u64(42);

        // Sample should return the valid one
        for _ in 0..10 {
            if let Some(w) = pool.sample_valid(&adj, &mut rng) {
                assert!(w.is_valid(&adj));
                assert_eq!(w.vertices, Witness::from_vertices(&[3, 4, 5]).vertices);
            }
        }
    }

    #[test]
    fn pool_merge_deduplicates() {
        let mut pool1 = WitnessPool::new(100, 2);
        pool1.add(Witness::from_vertices(&[0, 1]));
        pool1.add(Witness::from_vertices(&[2, 3]));

        let mut pool2 = WitnessPool::new(100, 2);
        pool2.add(Witness::from_vertices(&[2, 3])); // Duplicate
        pool2.add(Witness::from_vertices(&[4, 5])); // New

        pool1.merge(&pool2.snapshot());

        assert_eq!(pool1.len(), 3); // No duplicate
    }
}


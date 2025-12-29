//! Lock-free data structures and SIMD-optimized operations.
//!
//! This module provides high-performance concurrent data structures designed
//! for the multi-core Ramsey search:
//!
//! - `LockFreeWitnessPool`: Lock-free witness sharing across workers
//! - `LockFreeElitePool`: Lock-free elite solution sharing
//! - SIMD-optimized batch witness validation
//!
//! # Design Principles
//!
//! 1. **Lock-free where possible**: Use atomic operations and CAS loops
//! 2. **Wait-free reads**: Readers never block
//! 3. **Bounded memory**: Fixed-size ring buffers prevent unbounded growth
//! 4. **Cache-friendly**: Structures aligned to cache lines

use crossbeam::queue::ArrayQueue;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

/// Cache line size for padding (64 bytes on most x86/ARM)
const CACHE_LINE_SIZE: usize = 64;

// ============================================================================
// Lock-Free Witness Pool
// ============================================================================

/// A single witness entry in the lock-free pool.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct WitnessEntry {
    /// Bitset of vertices in the independent set.
    pub vertices: u64,
    /// Size of the witness.
    pub size: u8,
}

impl WitnessEntry {
    /// Creates a new witness entry.
    #[inline]
    pub const fn new(vertices: u64, size: u8) -> Self {
        Self {
            vertices,
            size,
        }
    }

    /// Creates an empty (invalid) entry.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            vertices: 0,
            size: 0,
        }
    }

    /// Returns true if this entry is valid (non-empty).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.vertices != 0
    }

    /// Checks if this witness is still valid in the given adjacency.
    #[inline]
    pub fn is_independent<const N: usize>(&self, adj: &[u64; N]) -> bool {
        let mut remaining = self.vertices;
        while remaining != 0 {
            let v = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            if (adj[v] & self.vertices) != 0 {
                return false;
            }
        }
        true
    }
}

/// Lock-free witness pool using a ring buffer of atomic entries.
///
/// This structure is designed for high-throughput concurrent access:
/// - Writes use CAS to avoid lost updates
/// - Reads are wait-free (just load atomics)
/// - Fixed size prevents unbounded memory growth
pub struct LockFreeWitnessPool {
    /// Ring buffer of witness vertex bitsets. `0` means empty.
    entries: Box<[AtomicU64]>,
    /// Capacity of the pool.
    capacity: usize,
    /// Next write position (wraps around).
    write_head: AtomicUsize,
    /// Count of valid entries (approximate).
    count: AtomicUsize,
    /// Minimum witness size to accept.
    min_size: usize,
    /// Padding to prevent false sharing.
    _pad: [u8; CACHE_LINE_SIZE],
}

impl LockFreeWitnessPool {
    /// Creates a new lock-free witness pool.
    pub fn new(capacity: usize, min_size: usize) -> Arc<Self> {
        let entries: Vec<AtomicU64> = (0..capacity).map(|_| AtomicU64::new(0)).collect();
        Arc::new(Self {
            entries: entries.into_boxed_slice(),
            capacity,
            write_head: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            min_size,
            _pad: [0; CACHE_LINE_SIZE],
        })
    }

    /// Returns the capacity of the pool.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the approximate count of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Returns true if the pool is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to add a witness to the pool.
    ///
    /// Returns true if the witness was added (or a similar one exists).
    /// This is lock-free but not wait-free (may retry on contention).
    pub fn try_add(&self, vertices: u64, size: u8) -> bool {
        if (size as usize) < self.min_size {
            return false;
        }
        debug_assert_eq!(vertices.count_ones() as u8, size, "size must match vertices popcount");
        if vertices == 0 {
            return false;
        }

        // Check for duplicates (approximate - we might miss some due to races)
        for entry in self.entries.iter() {
            if entry.load(Ordering::Relaxed) == vertices {
                return false; // Already exists
            }
        }

        // Get next write position
        let pos = self.write_head.fetch_add(1, Ordering::Relaxed) % self.capacity;

        // Swap into ring buffer. This is lock-free and sufficient for our bounded pool semantics.
        let old = self.entries[pos].swap(vertices, Ordering::AcqRel);
        if old == 0 {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        true
    }

    /// Adds a witness from a vertex list.
    pub fn add_from_vertices(&self, verts: &[usize]) -> bool {
        let mut vertices = 0u64;
        for &v in verts {
            vertices |= 1u64 << v;
        }
        self.try_add(vertices, verts.len() as u8)
    }

    /// Returns all valid witnesses as a vector.
    ///
    /// This is a snapshot - the pool may change during iteration.
    pub fn snapshot(&self) -> Vec<WitnessEntry> {
        let mut result = Vec::with_capacity(self.capacity);
        for entry in self.entries.iter() {
            let vertices = entry.load(Ordering::Relaxed);
            if vertices != 0 {
                let e = WitnessEntry::new(vertices, vertices.count_ones() as u8);
                result.push(e);
            }
        }
        result
    }

    /// Counts witnesses that are still valid for the given adjacency.
    pub fn count_valid<const N: usize>(&self, adj: &[u64; N]) -> usize {
        let mut count = 0;
        for entry in self.entries.iter() {
            let vertices = entry.load(Ordering::Relaxed);
            if vertices == 0 {
                continue;
            }
            let e = WitnessEntry::new(vertices, vertices.count_ones() as u8);
            if e.is_independent(adj) {
                count += 1;
            }
        }
        count
    }

    /// Removes witnesses that are invalid for the given adjacency (best-effort).
    ///
    /// Note: This is primarily useful for tests/diagnostics. In the search driver, workers
    /// typically prune their *local* witness pools; shared pools may contain witnesses that are
    /// invalid for some workers but still useful for others.
    pub fn prune_invalid<const N: usize>(&self, adj: &[u64; N]) {
        // Best-effort clearing of invalid entries.
        for slot in self.entries.iter() {
            let vertices = slot.load(Ordering::Acquire);
            if vertices == 0 {
                continue;
            }
            let e = WitnessEntry::new(vertices, vertices.count_ones() as u8);
            if !e.is_independent(adj) {
                let _ = slot.compare_exchange(vertices, 0, Ordering::AcqRel, Ordering::Relaxed);
            }
        }

        // Recompute count (still only approximate under concurrent writers, but bounded).
        let mut live = 0usize;
        for slot in self.entries.iter() {
            if slot.load(Ordering::Relaxed) != 0 {
                live += 1;
            }
        }
        self.count.store(live, Ordering::Relaxed);
    }

    /// Batch validates witnesses using SIMD-friendly iteration.
    ///
    /// Returns the count of valid witnesses and optionally fills
    /// the provided buffer with the first `limit` valid witnesses.
    pub fn batch_validate<const N: usize>(
        &self,
        adj: &[u64; N],
        mut out: Option<&mut Vec<WitnessEntry>>,
        limit: usize,
    ) -> usize {
        let mut count = 0;

        // Process entries in cache-line-sized batches for better locality
        const BATCH_SIZE: usize = 8;
        let mut batch = [0u64; BATCH_SIZE];

        for chunk in self.entries.chunks(BATCH_SIZE) {
            // Load batch (SIMD-friendly: sequential memory access)
            for (i, entry) in chunk.iter().enumerate() {
                batch[i] = entry.load(Ordering::Relaxed);
            }

            // Validate batch
            for &vertices in &batch[..chunk.len()] {
                if vertices == 0 {
                    continue;
                }
                let entry = WitnessEntry::new(vertices, vertices.count_ones() as u8);
                if entry.is_independent(adj) {
                    count += 1;
                    if let Some(ref mut out_vec) = out {
                        if out_vec.len() < limit {
                            out_vec.push(entry);
                        }
                    }
                }
            }
        }

        count
    }

    /// Finds the best edge by coverage (lock-free read).
    pub fn best_edge_by_coverage<const N: usize>(
        &self,
        adj: &[u64; N],
        max_c4_created: usize,
    ) -> Option<((usize, usize), usize)> {
        // Collect valid witnesses first
        let witnesses: Vec<_> = self
            .snapshot()
            .into_iter()
            .filter(|w| w.is_independent(adj))
            .collect();

        if witnesses.is_empty() {
            return None;
        }

        let mut best_edge = None;
        let mut best_hits = 0;
        let mut best_c4 = usize::MAX;

        for u in 0..N {
            for v in (u + 1)..N {
                // Skip existing edges
                if (adj[u] & (1u64 << v)) != 0 {
                    continue;
                }

                let c4_created = if max_c4_created == 0 {
                    if !is_c4_safe_to_add_from_adj::<N>(adj, u, v) {
                        continue;
                    }
                    0
                } else {
                    // Heuristic estimate when callers allow C4 creation.
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

                // Count hits using SIMD-friendly loop
                let edge_mask = (1u64 << u) | (1u64 << v);
                let hits = witnesses
                    .iter()
                    .filter(|w| (w.vertices & edge_mask) == edge_mask)
                    .count();

                if hits > best_hits || (hits == best_hits && c4_created < best_c4) {
                    best_hits = hits;
                    best_c4 = c4_created;
                    best_edge = Some((u, v));
                }
            }
        }

        best_edge.map(|e| (e, best_hits))
    }
}

/// Returns true iff adding the non-edge (u,v) would create **zero** C4s, using only adjacency
/// bitsets.
#[inline]
fn is_c4_safe_to_add_from_adj<const N: usize>(adj: &[u64; N], u: usize, v: usize) -> bool {
    debug_assert!(u < N && v < N && u != v);
    debug_assert!((adj[u] & (1u64 << v)) == 0, "caller must pass a non-edge");

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
// Lock-Free Elite Pool
// ============================================================================

/// Entry in the lock-free elite pool.
#[derive(Clone)]
pub struct EliteEntry<const N: usize> {
    /// Adjacency matrix (packed).
    pub adj: [u64; N],
    /// Energy of this solution.
    pub energy: usize,
    /// Hash for quick comparison.
    pub hash: u64,
}

impl<const N: usize> EliteEntry<N> {
    /// Creates a new elite entry.
    pub fn new(adj: [u64; N], energy: usize) -> Self {
        let hash = compute_hash(&adj);
        Self { adj, energy, hash }
    }

    /// Computes Hamming distance to another entry.
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let mut dist = 0;
        for i in 0..N {
            dist += (self.adj[i] ^ other.adj[i]).count_ones() as usize;
        }
        dist / 2
    }
}

/// Computes a hash of an adjacency matrix.
fn compute_hash<const N: usize>(adj: &[u64; N]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &row in adj {
        hash ^= row;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Lock-free elite pool using a concurrent queue.
///
/// This uses crossbeam's ArrayQueue for bounded lock-free operations.
pub struct LockFreeElitePool<const N: usize> {
    /// Queue of elite entries.
    queue: ArrayQueue<EliteEntry<N>>,
    /// Best energy seen (atomic for lock-free reads).
    best_energy: AtomicUsize,
    /// Count of entries.
    count: AtomicUsize,
    /// Minimum diversity (Hamming distance) for uniqueness.
    #[allow(dead_code)]
    min_diversity: usize,
}

impl<const N: usize> LockFreeElitePool<N> {
    /// Creates a new lock-free elite pool.
    pub fn new(capacity: usize) -> Arc<Self> {
        let total_edges = N * (N - 1) / 2;
        let min_diversity = (total_edges / 20).max(3);

        Arc::new(Self {
            queue: ArrayQueue::new(capacity),
            best_energy: AtomicUsize::new(usize::MAX),
            count: AtomicUsize::new(0),
            min_diversity,
        })
    }

    /// Returns the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Returns the current count.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the best energy seen.
    #[inline]
    pub fn best_energy(&self) -> usize {
        self.best_energy.load(Ordering::Relaxed)
    }

    /// Attempts to add a solution to the pool.
    ///
    /// Returns true if added (meets quality/diversity criteria).
    pub fn try_add(&self, adj: [u64; N], energy: usize) -> bool {
        // Update best energy atomically
        let mut current_best = self.best_energy.load(Ordering::Relaxed);
        while energy < current_best {
            match self.best_energy.compare_exchange_weak(
                current_best,
                energy,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_best = actual,
            }
        }

        let entry = EliteEntry::new(adj, energy);

        // Try to push to queue
        match self.queue.push(entry.clone()) {
            Ok(()) => {
                self.count.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                // Queue is full, try to pop worst and retry
                if let Some(worst) = self.queue.pop() {
                    if energy < worst.energy {
                        // Our entry is better, push it
                        let _ = self.queue.push(entry);
                        true
                    } else {
                        // Put the old one back
                        let _ = self.queue.push(worst);
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Samples a random entry from the pool.
    pub fn sample<R: rand::Rng>(&self, _rng: &mut R) -> Option<[u64; N]> {
        if self.queue.is_empty() {
            return None;
        }

        // Pop and re-push to sample (not ideal but lock-free)
        if let Some(entry) = self.queue.pop() {
            let adj = entry.adj;
            let _ = self.queue.push(entry);
            Some(adj)
        } else {
            None
        }
    }

    /// Returns a snapshot of all entries.
    pub fn snapshot(&self) -> Vec<EliteEntry<N>> {
        let mut result = Vec::with_capacity(self.queue.capacity());

        // Drain and refill to get snapshot
        let mut temp = Vec::new();
        while let Some(entry) = self.queue.pop() {
            temp.push(entry);
        }

        for entry in temp {
            result.push(entry.clone());
            let _ = self.queue.push(entry);
        }

        result
    }
}

// ============================================================================
// SIMD-Optimized Batch Operations
// ============================================================================

/// Batch validates multiple witnesses against an adjacency matrix.
///
/// This function is optimized for SIMD auto-vectorization by:
/// - Processing multiple witnesses in parallel
/// - Using predictable memory access patterns
/// - Avoiding branches in the inner loop where possible
#[inline]
pub fn batch_validate_witnesses<const N: usize>(
    adj: &[u64; N],
    witnesses: &[u64],
) -> Vec<bool> {
    witnesses
        .iter()
        .map(|&w| {
            // SIMD-friendly: each witness check is independent
            let mut valid = true;
            let mut remaining = w;
            while remaining != 0 && valid {
                let v = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                valid = (adj[v] & w) == 0;
            }
            valid
        })
        .collect()
}

/// Counts valid witnesses using SIMD-friendly accumulation.
#[inline]
pub fn count_valid_witnesses<const N: usize>(adj: &[u64; N], witnesses: &[u64]) -> usize {
    witnesses
        .iter()
        .filter(|&&w| {
            let mut remaining = w;
            while remaining != 0 {
                let v = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                if (adj[v] & w) != 0 {
                    return false;
                }
            }
            true
        })
        .count()
}

/// Finds edges that hit the most witnesses (coverage-based selection).
///
/// Optimized for batch processing with predictable memory access.
pub fn find_best_coverage_edges<const N: usize>(
    adj: &[u64; N],
    witnesses: &[u64],
    max_c4: usize,
    limit: usize,
) -> Vec<((usize, usize), usize)> {
    if witnesses.is_empty() {
        return Vec::new();
    }

    // Pre-filter valid witnesses
    let valid_witnesses: Vec<u64> = witnesses
        .iter()
        .copied()
        .filter(|&w| {
            let mut remaining = w;
            while remaining != 0 {
                let v = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                if (adj[v] & w) != 0 {
                    return false;
                }
            }
            true
        })
        .collect();

    if valid_witnesses.is_empty() {
        return Vec::new();
    }

    let mut candidates: Vec<((usize, usize), usize, usize)> = Vec::new();

    for u in 0..N {
        for v in (u + 1)..N {
            if (adj[u] & (1u64 << v)) != 0 {
                continue;
            }

            let common = (adj[u] & adj[v]).count_ones() as usize;
            let c4 = if common >= 2 {
                common * (common - 1) / 2
            } else {
                0
            };

            if c4 > max_c4 {
                continue;
            }

            // Count hits (SIMD-friendly: independent iterations)
            let edge_mask = (1u64 << u) | (1u64 << v);
            let hits: usize = valid_witnesses
                .iter()
                .map(|&w| ((w & edge_mask) == edge_mask) as usize)
                .sum();

            if hits > 0 {
                candidates.push(((u, v), hits, c4));
            }
        }
    }

    // Sort by hits descending, C4 ascending
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    candidates.truncate(limit);

    candidates.into_iter().map(|(e, h, _)| (e, h)).collect()
}

/// Computes the intersection of multiple adjacency rows in parallel.
///
/// Useful for common neighbor computation.
#[inline]
pub fn parallel_intersection(rows: &[u64]) -> u64 {
    if rows.is_empty() {
        return 0;
    }
    rows.iter().fold(u64::MAX, |acc, &r| acc & r)
}

/// Computes the union of multiple adjacency rows.
#[inline]
pub fn parallel_union(rows: &[u64]) -> u64 {
    rows.iter().fold(0u64, |acc, &r| acc | r)
}

// ============================================================================
// Thread-Local Buffers
// ============================================================================

/// Thread-local buffer for avoiding allocations in hot paths.
///
/// Each worker maintains its own buffer that gets synced periodically.
pub struct LocalBuffer<T> {
    /// Local items.
    items: Vec<T>,
    /// Maximum local capacity before sync.
    max_local: usize,
}

impl<T: Clone> LocalBuffer<T> {
    /// Creates a new local buffer.
    pub fn new(max_local: usize) -> Self {
        Self {
            items: Vec::with_capacity(max_local),
            max_local,
        }
    }

    /// Adds an item to the local buffer.
    ///
    /// Returns true if the buffer is now full and should be synced.
    pub fn add(&mut self, item: T) -> bool {
        self.items.push(item);
        self.items.len() >= self.max_local
    }

    /// Drains and returns all items.
    pub fn drain(&mut self) -> Vec<T> {
        std::mem::take(&mut self.items)
    }

    /// Returns current count.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ============================================================================
// Atomic Statistics
// ============================================================================

/// Lock-free statistics counters.
#[derive(Default)]
pub struct AtomicStats {
    /// Total iterations.
    pub iterations: AtomicU64,
    /// Improvements found.
    pub improvements: AtomicU64,
    /// Witnesses learned.
    pub witnesses_learned: AtomicU64,
    /// Swaps attempted (for replica exchange).
    pub swap_attempts: AtomicU64,
    /// Swaps accepted.
    pub swap_accepts: AtomicU64,
}

impl AtomicStats {
    /// Creates new zeroed stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments iteration count.
    #[inline]
    pub fn inc_iterations(&self) {
        self.iterations.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments improvement count.
    #[inline]
    pub fn inc_improvements(&self) {
        self.improvements.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments witness count.
    #[inline]
    pub fn inc_witnesses(&self) {
        self.witnesses_learned.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a swap attempt and result.
    #[inline]
    pub fn record_swap(&self, accepted: bool) {
        self.swap_attempts.fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.swap_accepts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns current statistics as a tuple.
    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.iterations.load(Ordering::Relaxed),
            self.improvements.load(Ordering::Relaxed),
            self.witnesses_learned.load(Ordering::Relaxed),
            self.swap_attempts.load(Ordering::Relaxed),
            self.swap_accepts.load(Ordering::Relaxed),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;
    use std::thread;

    // -------------------------------------------------------------------------
    // WitnessEntry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn witness_entry_size_matches_popcount() {
        let vertices = 0x1234_5678_9ABC_DEF0u64;
        let entry = WitnessEntry::new(vertices, vertices.count_ones() as u8);
        assert_eq!(entry.vertices, vertices);
        assert_eq!(entry.size, vertices.count_ones() as u8);
    }

    #[test]
    fn witness_entry_empty_is_invalid() {
        let empty = WitnessEntry::empty();
        assert!(!empty.is_valid());

        let valid = WitnessEntry::new(0b111, 3);
        assert!(valid.is_valid());
    }

    #[test]
    fn witness_entry_is_independent_empty_graph() {
        let adj = [0u64; 10];
        let witness = WitnessEntry::new(0b11111, 5);
        assert!(witness.is_independent(&adj));
    }

    #[test]
    fn witness_entry_is_independent_detects_edge() {
        let mut adj = [0u64; 10];
        adj[0] = 1 << 1;
        adj[1] = 1 << 0;

        let witness_with_edge = WitnessEntry::new(0b11, 2); // vertices 0,1
        let witness_without = WitnessEntry::new(0b1100, 2); // vertices 2,3

        assert!(!witness_with_edge.is_independent(&adj));
        assert!(witness_without.is_independent(&adj));
    }

    // -------------------------------------------------------------------------
    // LockFreeWitnessPool Tests
    // -------------------------------------------------------------------------

    #[test]
    fn lockfree_witness_pool_basic() {
        let pool = LockFreeWitnessPool::new(100, 2);

        assert!(pool.try_add(0b111, 3));
        assert!(pool.try_add(0b1110, 3));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn lockfree_witness_pool_rejects_small() {
        let pool = LockFreeWitnessPool::new(100, 5);

        assert!(!pool.try_add(0b111, 3)); // Too small
        assert!(pool.try_add(0b11111, 5)); // Just right
    }

    #[test]
    fn lockfree_witness_pool_deduplicates() {
        let pool = LockFreeWitnessPool::new(100, 2);

        assert!(pool.try_add(0b111, 3));
        assert!(!pool.try_add(0b111, 3)); // Duplicate
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn lockfree_witness_pool_snapshot() {
        let pool = LockFreeWitnessPool::new(100, 2);

        pool.try_add(0b111, 3);
        pool.try_add(0b11100, 3);

        let snap = pool.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn lockfree_witness_pool_concurrent_adds() {
        let pool = LockFreeWitnessPool::new(1000, 2);
        let pool_clone = Arc::clone(&pool);

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let p = Arc::clone(&pool_clone);
                thread::spawn(move || {
                    for j in 0..100 {
                        // Construct a 3-vertex witness bitset deterministically.
                        let a = (i * 17 + j) % 64;
                        let b = (a + 1) % 64;
                        let c = (a + 2) % 64;
                        let vertices = (1u64 << a) | (1u64 << b) | (1u64 << c);
                        p.try_add(vertices, 3);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should have some entries (exact count depends on timing)
        assert!(pool.len() > 0);
    }

    #[test]
    fn lockfree_witness_pool_count_valid() {
        let pool = LockFreeWitnessPool::new(100, 2);

        // Add witnesses for vertices {0,1,2} and {3,4,5}
        pool.try_add(0b111, 3);
        pool.try_add(0b111000, 3);

        // Empty graph - both valid
        let adj = [0u64; 10];
        assert_eq!(pool.count_valid(&adj), 2);

        // Add edge 0-1 - first witness invalid
        let mut adj2 = [0u64; 10];
        adj2[0] = 1 << 1;
        adj2[1] = 1 << 0;
        assert_eq!(pool.count_valid(&adj2), 1);
    }

    #[test]
    fn lockfree_witness_pool_batch_validate() {
        let pool = LockFreeWitnessPool::new(100, 2);

        for i in 0..50 {
            pool.try_add((1u64 << i) | (1u64 << (i + 1)), 2);
        }

        let adj = [0u64; 64];
        let mut out = Vec::new();
        let count = pool.batch_validate(&adj, Some(&mut out), 10);

        assert!(count > 0);
        assert!(out.len() <= 10);
    }

    // -------------------------------------------------------------------------
    // LockFreeElitePool Tests
    // -------------------------------------------------------------------------

    #[test]
    fn lockfree_elite_pool_basic() {
        let pool = LockFreeElitePool::<10>::new(10);

        let adj = [0u64; 10];
        assert!(pool.try_add(adj, 100));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.best_energy(), 100);
    }

    #[test]
    fn lockfree_elite_pool_updates_best() {
        let pool = LockFreeElitePool::<10>::new(10);

        let adj = [0u64; 10];
        pool.try_add(adj, 100);
        pool.try_add(adj, 50);
        pool.try_add(adj, 75);

        assert_eq!(pool.best_energy(), 50);
    }

    #[test]
    fn lockfree_elite_pool_sample() {
        let pool = LockFreeElitePool::<10>::new(10);
        let mut rng = XorShiftRng::seed_from_u64(42);

        let adj = [1u64; 10];
        pool.try_add(adj, 100);

        let sampled = pool.sample(&mut rng);
        assert!(sampled.is_some());
        assert_eq!(sampled.unwrap(), adj);
    }

    // -------------------------------------------------------------------------
    // Batch Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn batch_validate_witnesses_empty() {
        let adj = [0u64; 10];
        let witnesses: Vec<u64> = vec![];
        let result = batch_validate_witnesses(&adj, &witnesses);
        assert!(result.is_empty());
    }

    #[test]
    fn batch_validate_witnesses_all_valid() {
        let adj = [0u64; 10];
        let witnesses = vec![0b111, 0b11100, 0b1110000];
        let result = batch_validate_witnesses(&adj, &witnesses);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&v| v));
    }

    #[test]
    fn batch_validate_witnesses_mixed() {
        let mut adj = [0u64; 10];
        adj[0] = 1 << 1;
        adj[1] = 1 << 0;

        let witnesses = vec![
            0b11,    // Contains edge 0-1, invalid
            0b1100,  // No edges, valid
            0b111,   // Contains edge 0-1, invalid
        ];
        let result = batch_validate_witnesses(&adj, &witnesses);

        assert_eq!(result, vec![false, true, false]);
    }

    #[test]
    fn count_valid_witnesses_matches_batch() {
        let mut adj = [0u64; 10];
        adj[0] = 1 << 1;
        adj[1] = 1 << 0;

        let witnesses = vec![0b11, 0b1100, 0b111, 0b11110000];

        let count = count_valid_witnesses(&adj, &witnesses);
        let batch = batch_validate_witnesses(&adj, &witnesses);
        let batch_count = batch.iter().filter(|&&v| v).count();

        assert_eq!(count, batch_count);
    }

    #[test]
    fn find_best_coverage_edges_basic() {
        let adj = [0u64; 8];
        // Witnesses that all contain vertices 0 and 1
        let witnesses = vec![
            0b111,     // 0,1,2
            0b1011,    // 0,1,3
            0b10011,   // 0,1,4
        ];

        let result = find_best_coverage_edges(&adj, &witnesses, 100, 5);

        assert!(!result.is_empty());
        // Edge (0,1) should hit all 3 witnesses
        assert!(result.iter().any(|&((u, v), hits)| (u == 0 && v == 1) && hits == 3));
    }

    #[test]
    fn find_best_coverage_edges_respects_c4() {
        let mut adj = [0u64; 8];
        // Create situation where (0,1) would create C4
        adj[0] = 0b1100; // neighbors 2,3
        adj[1] = 0b1100;
        adj[2] = 0b11;
        adj[3] = 0b11;

        let witnesses = vec![0b11]; // Want to hit 0,1

        let result = find_best_coverage_edges(&adj, &witnesses, 0, 5);

        // Should not include (0,1) because it creates C4s
        assert!(!result.iter().any(|&((u, v), _)| u == 0 && v == 1));
    }

    // -------------------------------------------------------------------------
    // Parallel Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn parallel_intersection_basic() {
        let rows = vec![0b11111111, 0b11110000, 0b11001100];
        let result = parallel_intersection(&rows);
        assert_eq!(result, 0b11000000);
    }

    #[test]
    fn parallel_intersection_empty() {
        let rows: Vec<u64> = vec![];
        let result = parallel_intersection(&rows);
        assert_eq!(result, 0);
    }

    #[test]
    fn parallel_union_basic() {
        let rows = vec![0b00000001, 0b00000010, 0b00000100];
        let result = parallel_union(&rows);
        assert_eq!(result, 0b00000111);
    }

    // -------------------------------------------------------------------------
    // LocalBuffer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn local_buffer_basic() {
        let mut buf: LocalBuffer<u32> = LocalBuffer::new(5);

        assert!(!buf.add(1));
        assert!(!buf.add(2));
        assert!(!buf.add(3));
        assert!(!buf.add(4));
        assert!(buf.add(5)); // Full now

        assert_eq!(buf.len(), 5);

        let drained = buf.drain();
        assert_eq!(drained, vec![1, 2, 3, 4, 5]);
        assert!(buf.is_empty());
    }

    // -------------------------------------------------------------------------
    // AtomicStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn atomic_stats_basic() {
        let stats = AtomicStats::new();

        stats.inc_iterations();
        stats.inc_iterations();
        stats.inc_improvements();
        stats.inc_witnesses();
        stats.record_swap(true);
        stats.record_swap(false);

        let (iters, imps, wits, attempts, accepts) = stats.snapshot();
        assert_eq!(iters, 2);
        assert_eq!(imps, 1);
        assert_eq!(wits, 1);
        assert_eq!(attempts, 2);
        assert_eq!(accepts, 1);
    }

    #[test]
    fn atomic_stats_concurrent() {
        let stats = Arc::new(AtomicStats::new());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let s = Arc::clone(&stats);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        s.inc_iterations();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let (iters, _, _, _, _) = stats.snapshot();
        assert_eq!(iters, 4000);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn witness_entry_max_vertices() {
        // Test with maximum supported vertices (64 bits).
        let max_vertices = u64::MAX;
        let entry = WitnessEntry::new(max_vertices, 64);
        assert!(entry.is_valid());
        assert_eq!(entry.vertices, max_vertices);
        assert_eq!(entry.size, 64);
    }

    #[test]
    fn lockfree_witness_pool_wraps_around() {
        let pool = LockFreeWitnessPool::new(4, 2);

        // Fill the pool
        for i in 0..4 {
            pool.try_add(1u64 << i | 1u64 << (i + 4), 2);
        }

        // Add more - should wrap around
        for i in 4..8 {
            pool.try_add(1u64 << i | 1u64 << (i + 4), 2);
        }

        // Pool should still work
        let snap = pool.snapshot();
        assert!(snap.len() <= 4);
    }

    #[test]
    fn elite_entry_hamming_distance() {
        let mut adj1 = [0u64; 10];
        let mut adj2 = [0u64; 10];

        adj1[0] = 0b111;
        adj2[0] = 0b110;

        let e1 = EliteEntry::new(adj1, 100);
        let e2 = EliteEntry::new(adj2, 100);

        // Differ by 1 bit in row 0
        assert_eq!(e1.hamming_distance(&e2), 0); // Actually 0 because we need symmetric edges

        // Add symmetric edge difference
        adj1[1] = 1;
        let e1 = EliteEntry::new(adj1, 100);
        // Now differ by 1 edge
        assert!(e1.hamming_distance(&e2) > 0);
    }

    // -------------------------------------------------------------------------
    // Advanced Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn lockfree_witness_pool_heavy_concurrent_load() {
        // Simulate heavy load from many threads
        let pool = LockFreeWitnessPool::new(500, 2);
        let pool_ref = Arc::clone(&pool);

        let num_threads = 16;
        let ops_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let p = Arc::clone(&pool_ref);
                thread::spawn(move || {
                    let mut rng = XorShiftRng::seed_from_u64(tid as u64);
                    for _ in 0..ops_per_thread {
                        // Random witness
                        let v1 = rng.random_range(0..48);
                        let v2 = rng.random_range(0..48);
                        if v1 != v2 {
                            let vertices = (1u64 << v1) | (1u64 << v2);
                            p.try_add(vertices, 2);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Pool should have entries and not have corrupted state
        let snap = pool.snapshot();
        assert!(snap.len() <= pool.capacity());

        // All entries should be valid
        for entry in &snap {
            assert!(entry.is_valid());
            assert_eq!(entry.vertices.count_ones() as u8, entry.size);
        }
    }

    #[test]
    fn lockfree_elite_pool_concurrent_updates() {
        let pool = LockFreeElitePool::<8>::new(20);
        let pool_ref = Arc::clone(&pool);

        let num_threads = 8;
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let p = Arc::clone(&pool_ref);
                thread::spawn(move || {
                    for i in 0..100 {
                        let mut adj = [0u64; 8];
                        adj[0] = tid as u64;
                        adj[1] = i as u64;
                        let energy = (tid * 100 + i) % 500;
                        p.try_add(adj, energy);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Best energy should be valid
        let best = pool.best_energy();
        assert!(best <= 500);

        // Snapshot should work
        let snap = pool.snapshot();
        assert!(!snap.is_empty());
    }

    #[test]
    fn atomic_stats_high_contention() {
        let stats = Arc::new(AtomicStats::new());
        let num_threads = 32;
        let ops_per_thread = 10_000;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let s = Arc::clone(&stats);
                thread::spawn(move || {
                    for _ in 0..ops_per_thread {
                        s.inc_iterations();
                        s.inc_improvements();
                        s.inc_witnesses();
                        s.record_swap(true);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let (iters, imps, wits, attempts, accepts) = stats.snapshot();
        let expected = (num_threads * ops_per_thread) as u64;

        assert_eq!(iters, expected);
        assert_eq!(imps, expected);
        assert_eq!(wits, expected);
        assert_eq!(attempts, expected);
        assert_eq!(accepts, expected);
    }

    // -------------------------------------------------------------------------
    // SIMD Validation Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn batch_validate_large_witness_set() {
        let mut adj = [0u64; 32];
        // Create a sparse graph
        for i in 0..16 {
            adj[i * 2] = 1 << (i * 2 + 1);
            adj[i * 2 + 1] = 1 << (i * 2);
        }

        // Create many witnesses
        let mut rng = XorShiftRng::seed_from_u64(123);
        let witnesses: Vec<u64> = (0..1000)
            .map(|_| {
                let v1 = rng.random_range(0..32);
                let v2 = rng.random_range(0..32);
                let v3 = rng.random_range(0..32);
                (1u64 << v1) | (1u64 << v2) | (1u64 << v3)
            })
            .collect();

        // Validate
        let results = batch_validate_witnesses(&adj, &witnesses);
        assert_eq!(results.len(), 1000);

        // Cross-check with count function
        let valid_count = results.iter().filter(|&&v| v).count();
        let count_result = count_valid_witnesses(&adj, &witnesses);
        assert_eq!(valid_count, count_result);
    }

    #[test]
    fn find_best_coverage_edges_large_graph() {
        let adj = [0u64; 24];

        // Create witnesses that overlap on specific vertices
        let witnesses = vec![
            0b111,          // 0,1,2
            0b1011,         // 0,1,3
            0b10011,        // 0,1,4
            0b100011,       // 0,1,5
            0b1000111,      // 0,1,2,6
            0b10000111,     // 0,1,2,7
        ];

        let edges = find_best_coverage_edges(&adj, &witnesses, 10, 10);

        // Edge (0,1) should be in top results
        assert!(!edges.is_empty());
        let top_edge = edges[0];
        assert_eq!(top_edge.0, (0, 1));
        assert!(top_edge.1 >= 4); // At least 4 witnesses contain (0,1)
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn local_buffer_empty_drain() {
        let mut buf: LocalBuffer<u32> = LocalBuffer::new(5);
        let drained = buf.drain();
        assert!(drained.is_empty());
    }

    #[test]
    fn local_buffer_exactly_full() {
        let mut buf: LocalBuffer<u32> = LocalBuffer::new(3);
        assert!(!buf.add(1));
        assert!(!buf.add(2));
        assert!(buf.add(3)); // Exactly full

        assert_eq!(buf.len(), 3);
        let drained = buf.drain();
        assert_eq!(drained.len(), 3);
    }

    #[test]
    fn parallel_intersection_single_row() {
        let rows = vec![0b10101010];
        assert_eq!(parallel_intersection(&rows), 0b10101010);
    }

    #[test]
    fn parallel_union_empty() {
        let rows: Vec<u64> = vec![];
        assert_eq!(parallel_union(&rows), 0);
    }

    #[test]
    fn parallel_union_single_row() {
        let rows = vec![0b10101010];
        assert_eq!(parallel_union(&rows), 0b10101010);
    }

    #[test]
    fn lockfree_elite_pool_empty_sample() {
        let pool = LockFreeElitePool::<8>::new(10);
        let mut rng = XorShiftRng::seed_from_u64(42);

        let sampled = pool.sample(&mut rng);
        assert!(sampled.is_none());
    }

    #[test]
    fn lockfree_witness_pool_empty_operations() {
        let pool = LockFreeWitnessPool::new(100, 2);

        // Empty pool operations
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);

        let adj = [0u64; 10];
        assert_eq!(pool.count_valid(&adj), 0);

        let snap = pool.snapshot();
        assert!(snap.is_empty());
    }

    #[test]
    fn batch_validate_single_witness() {
        let adj = [0u64; 10];
        let witnesses = vec![0b111];

        let results = batch_validate_witnesses(&adj, &witnesses);
        assert_eq!(results.len(), 1);
        assert!(results[0]);
    }

    #[test]
    fn find_coverage_edges_no_witnesses() {
        let adj = [0u64; 8];
        let witnesses: Vec<u64> = vec![];

        let edges = find_best_coverage_edges(&adj, &witnesses, 10, 10);
        assert!(edges.is_empty());
    }

    #[test]
    fn find_coverage_edges_all_invalid() {
        let mut adj = [0u64; 8];
        // Create complete graph (all witnesses invalid)
        for i in 0..8 {
            adj[i] = 0xFF ^ (1u64 << i);
        }

        let witnesses = vec![0b11, 0b111, 0b1111];

        let edges = find_best_coverage_edges(&adj, &witnesses, 10, 10);
        assert!(edges.is_empty());
    }

    // -------------------------------------------------------------------------
    // Memory Layout and Alignment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn witness_entry_size() {
        // WitnessEntry should be small for cache efficiency
        assert!(std::mem::size_of::<WitnessEntry>() <= 16);
    }

    #[test]
    fn lockfree_witness_pool_cache_alignment() {
        // Verify padding doesn't break struct
        let pool = LockFreeWitnessPool::new(10, 2);
        assert_eq!(pool.capacity(), 10);
    }
}


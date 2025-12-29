//! Elite Pool: Shared repository of high-quality solutions across workers.
//!
//! The elite pool enables cross-worker communication and maintains diversity
//! to prevent premature convergence.

use crate::graph::RamseyState;
use std::sync::{Arc, RwLock};

/// A solution stored in the elite pool with its energy and diversity info.
#[derive(Clone, Debug)]
pub struct EliteSolution<const N: usize> {
    /// The graph state.
    pub state: RamseyState<N>,
    /// The energy (lower is better).
    pub energy: usize,
    /// Hash of adjacency for quick diversity check.
    #[allow(dead_code)]
    adj_hash: u64,
}

impl<const N: usize> EliteSolution<N> {
    /// Creates a new elite solution.
    pub fn new(state: RamseyState<N>, energy: usize) -> Self {
        let adj_hash = compute_adj_hash(state.adj());
        Self {
            state,
            energy,
            adj_hash,
        }
    }

    /// Computes the Hamming distance to another solution.
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let mut dist = 0;
        for i in 0..N {
            dist += (self.state.adj()[i] ^ other.state.adj()[i]).count_ones() as usize;
        }
        dist / 2 // Each edge counted twice
    }
}

/// Computes a hash of the adjacency matrix for quick diversity checks.
fn compute_adj_hash<const N: usize>(adj: &[u64; N]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
    for &row in adj {
        hash ^= row;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Thread-safe elite pool that maintains diverse high-quality solutions.
#[derive(Clone)]
pub struct ElitePool<const N: usize> {
    inner: Arc<RwLock<ElitePoolInner<N>>>,
}

struct ElitePoolInner<const N: usize> {
    solutions: Vec<EliteSolution<N>>,
    max_size: usize,
    min_diversity: usize, // Minimum Hamming distance for uniqueness
}

impl<const N: usize> ElitePool<N> {
    /// Creates a new elite pool with the given capacity.
    pub fn new(max_size: usize) -> Self {
        // Minimum diversity: at least 5% of edges should differ
        let total_edges = N * (N - 1) / 2;
        let min_diversity = (total_edges / 20).max(3);

        Self {
            inner: Arc::new(RwLock::new(ElitePoolInner {
                solutions: Vec::with_capacity(max_size),
                max_size,
                min_diversity,
            })),
        }
    }

    /// Returns the number of solutions in the pool.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().solutions.len()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to add a solution to the pool.
    ///
    /// Returns true if the solution was added (it was good enough and different enough).
    pub fn try_add(&self, state: RamseyState<N>, energy: usize) -> bool {
        let candidate = EliteSolution::new(state, energy);
        let mut inner = self.inner.write().unwrap();

        // Check if too similar to existing solution
        for existing in &inner.solutions {
            if candidate.hamming_distance(existing) < inner.min_diversity {
                // Too similar; only replace if strictly better
                if candidate.energy >= existing.energy {
                    return false;
                }
            }
        }

        // Find position to insert (maintain sorted by energy)
        let pos = inner
            .solutions
            .binary_search_by_key(&candidate.energy, |s| s.energy)
            .unwrap_or_else(|p| p);

        // If pool is full and this is worse than worst, reject
        if inner.solutions.len() >= inner.max_size && pos >= inner.max_size {
            return false;
        }

        // Insert and maintain max size
        inner.solutions.insert(pos, candidate);

        // Eviction with diversity awareness
        while inner.solutions.len() > inner.max_size {
            // Remove the worst solution, but try to maintain diversity
            // Simple strategy: remove the worst (last)
            inner.solutions.pop();
        }

        true
    }

    /// Samples a random solution from the pool.
    ///
    /// Returns None if the pool is empty.
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Option<RamseyState<N>> {
        let inner = self.inner.read().unwrap();
        if inner.solutions.is_empty() {
            return None;
        }

        // Bias toward better solutions (lower energy = more likely)
        // Use exponential weighting
        let weights: Vec<f64> = inner
            .solutions
            .iter()
            .enumerate()
            .map(|(i, _)| (-0.1 * i as f64).exp())
            .collect();

        let total: f64 = weights.iter().sum();
        let mut r = rng.random_range(0.0..total);

        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return Some(inner.solutions[i].state.clone());
            }
        }

        // Fallback to best
        Some(inner.solutions[0].state.clone())
    }

    /// Samples a solution biased toward diversity from the given reference.
    ///
    /// Prefers solutions that are different from the reference.
    pub fn sample_diverse<R: rand::Rng>(
        &self,
        rng: &mut R,
        reference: &RamseyState<N>,
    ) -> Option<RamseyState<N>> {
        let inner = self.inner.read().unwrap();
        if inner.solutions.is_empty() {
            return None;
        }

        let ref_solution = EliteSolution::new(reference.clone(), usize::MAX);

        // Weight by diversity (higher distance = more likely)
        let weights: Vec<f64> = inner
            .solutions
            .iter()
            .map(|s| {
                let dist = s.hamming_distance(&ref_solution) as f64;
                dist.sqrt() // Square root to moderate the effect
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if total < 1e-10 {
            return Some(inner.solutions[0].state.clone());
        }

        let mut r = rng.random_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return Some(inner.solutions[i].state.clone());
            }
        }

        Some(inner.solutions[0].state.clone())
    }

    /// Returns the best energy in the pool.
    pub fn best_energy(&self) -> Option<usize> {
        let inner = self.inner.read().unwrap();
        inner.solutions.first().map(|s| s.energy)
    }

    /// Returns the best solution in the pool.
    pub fn best(&self) -> Option<RamseyState<N>> {
        let inner = self.inner.read().unwrap();
        inner.solutions.first().map(|s| s.state.clone())
    }

    /// Returns statistics about the pool.
    pub fn stats(&self) -> ElitePoolStats {
        let inner = self.inner.read().unwrap();
        if inner.solutions.is_empty() {
            return ElitePoolStats {
                count: 0,
                best_energy: usize::MAX,
                worst_energy: 0,
                mean_energy: 0.0,
                mean_diversity: 0.0,
            };
        }

        let energies: Vec<usize> = inner.solutions.iter().map(|s| s.energy).collect();
        let mean_energy = energies.iter().sum::<usize>() as f64 / energies.len() as f64;

        // Compute average pairwise diversity
        let mut total_diversity = 0usize;
        let mut pairs = 0usize;
        for i in 0..inner.solutions.len() {
            for j in (i + 1)..inner.solutions.len() {
                total_diversity += inner.solutions[i].hamming_distance(&inner.solutions[j]);
                pairs += 1;
            }
        }
        let mean_diversity = if pairs > 0 {
            total_diversity as f64 / pairs as f64
        } else {
            0.0
        };

        ElitePoolStats {
            count: inner.solutions.len(),
            best_energy: energies[0],
            worst_energy: *energies.last().unwrap(),
            mean_energy,
            mean_diversity,
        }
    }

    /// Performs crossover between two solutions from the pool.
    ///
    /// Returns a new graph that combines adjacencies from both parents.
    pub fn crossover<R: rand::Rng>(&self, rng: &mut R) -> Option<RamseyState<N>> {
        let inner = self.inner.read().unwrap();
        if inner.solutions.len() < 2 {
            return inner.solutions.first().map(|s| s.state.clone());
        }

        // Select two different parents
        let idx1 = rng.random_range(0..inner.solutions.len());
        let mut idx2 = rng.random_range(0..inner.solutions.len());
        while idx2 == idx1 && inner.solutions.len() > 1 {
            idx2 = rng.random_range(0..inner.solutions.len());
        }

        let parent1 = &inner.solutions[idx1].state;
        let parent2 = &inner.solutions[idx2].state;

        // Uniform crossover: for each potential edge, randomly pick from one parent
        let mut child_adj = [0u64; N];

        for i in 0..N {
            for j in (i + 1)..N {
                let bit = 1u64 << j;
                let has_in_p1 = (parent1.adj()[i] & bit) != 0;
                let has_in_p2 = (parent2.adj()[i] & bit) != 0;

                let has_in_child = if has_in_p1 == has_in_p2 {
                    has_in_p1
                } else {
                    rng.random_bool(0.5)
                };

                if has_in_child {
                    child_adj[i] |= 1u64 << j;
                    child_adj[j] |= 1u64 << i;
                }
            }
        }

        Some(RamseyState::<N>::from_adj(child_adj))
    }
}

/// Statistics about the elite pool.
#[derive(Clone, Debug)]
pub struct ElitePoolStats {
    /// Number of solutions in the pool.
    pub count: usize,
    /// Best (lowest) energy.
    pub best_energy: usize,
    /// Worst (highest) energy in the pool.
    pub worst_energy: usize,
    /// Average energy.
    pub mean_energy: f64,
    /// Average pairwise Hamming distance (diversity measure).
    pub mean_diversity: f64,
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
    fn elite_pool_basic_operations() {
        let pool = ElitePool::<10>::new(5);
        assert!(pool.is_empty());

        let mut rng = XorShiftRng::seed_from_u64(42);
        let state = RamseyState::<10>::new_random(&mut rng, 0.3);

        assert!(pool.try_add(state.clone(), 100));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.best_energy(), Some(100));

        // Add a better solution
        let state2 = RamseyState::<10>::new_random(&mut rng, 0.3);
        assert!(pool.try_add(state2, 50));
        assert_eq!(pool.best_energy(), Some(50));
    }

    #[test]
    fn elite_pool_maintains_size_limit() {
        let pool = ElitePool::<8>::new(3);
        let mut rng = XorShiftRng::seed_from_u64(123);

        for i in 0..10 {
            let state = RamseyState::<8>::new_random(&mut rng, 0.3);
            pool.try_add(state, 100 - i);
        }

        assert!(pool.len() <= 3);
    }

    #[test]
    fn elite_pool_sampling_works() {
        let pool = ElitePool::<8>::new(10);
        let mut rng = XorShiftRng::seed_from_u64(456);

        // Add several solutions
        for i in 0..5 {
            let state = RamseyState::<8>::new_random(&mut rng, 0.3);
            pool.try_add(state, 100 + i * 10);
        }

        // Sample should return something
        let sampled = pool.sample(&mut rng);
        assert!(sampled.is_some());
    }

    #[test]
    fn elite_pool_crossover_works() {
        let pool = ElitePool::<8>::new(10);
        let mut rng = XorShiftRng::seed_from_u64(789);

        // Add two solutions
        let state1 = RamseyState::<8>::new_random(&mut rng, 0.3);
        let state2 = RamseyState::<8>::new_random(&mut rng, 0.3);
        pool.try_add(state1, 100);
        pool.try_add(state2, 100);

        // Crossover should work
        let child = pool.crossover(&mut rng);
        assert!(child.is_some());
    }

    #[test]
    fn elite_pool_diversity_check() {
        let pool = ElitePool::<8>::new(10);
        let mut rng = XorShiftRng::seed_from_u64(111);

        let state = RamseyState::<8>::new_random(&mut rng, 0.3);
        pool.try_add(state.clone(), 100);

        // Adding the exact same state should fail (too similar)
        let added = pool.try_add(state.clone(), 100);
        assert!(!added, "Duplicate should not be added");
    }

    #[test]
    fn hamming_distance_is_symmetric() {
        let mut rng = XorShiftRng::seed_from_u64(222);
        let s1 = EliteSolution::new(RamseyState::<10>::new_random(&mut rng, 0.3), 100);
        let s2 = EliteSolution::new(RamseyState::<10>::new_random(&mut rng, 0.3), 100);

        assert_eq!(s1.hamming_distance(&s2), s2.hamming_distance(&s1));
    }

    #[test]
    fn hamming_distance_to_self_is_zero() {
        let mut rng = XorShiftRng::seed_from_u64(333);
        let s = EliteSolution::new(RamseyState::<10>::new_random(&mut rng, 0.3), 100);

        assert_eq!(s.hamming_distance(&s), 0);
    }
}


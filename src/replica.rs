//! Replica Exchange (Parallel Tempering) for escaping local minima.
//!
//! Parallel tempering maintains multiple replicas at different temperatures.
//! Periodically, adjacent replicas attempt to swap states based on the
//! Metropolis criterion. This allows:
//!
//! - Hot replicas to explore freely and discover new basins
//! - Cold replicas to exploit and refine good solutions
//! - Information flow between temperatures via swaps
//!
//! This is different from the portfolio approach: portfolio uses different
//! strategies, while replica exchange uses a structured temperature ladder
//! with principled swap acceptance.

use crate::graph::RamseyState;
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================================
// Temperature Ladder
// ============================================================================

/// Generates a geometric temperature ladder.
///
/// Returns temperatures from `t_min` to `t_max` with geometric spacing.
/// This is the standard approach for replica exchange.
pub fn geometric_ladder(t_min: f64, t_max: f64, num_replicas: usize) -> Vec<f64> {
    if num_replicas <= 1 {
        return vec![t_min];
    }

    let ratio = (t_max / t_min).powf(1.0 / (num_replicas - 1) as f64);
    (0..num_replicas)
        .map(|i| t_min * ratio.powi(i as i32))
        .collect()
}

/// Generates an adaptive temperature ladder based on target swap rate.
///
/// The temperatures are spaced so that adjacent replicas have approximately
/// equal probability of swapping (targeting ~23% swap rate, which is optimal
/// for random-walk efficiency in temperature space).
pub fn adaptive_ladder(t_min: f64, t_max: f64, num_replicas: usize, energy_variance: f64) -> Vec<f64> {
    if num_replicas <= 1 {
        return vec![t_min];
    }

    // For Gaussian energy distributions, optimal spacing is:
    // T_{i+1} / T_i = 1 + sqrt(2 / (N * C_v))
    // where C_v is heat capacity. We approximate with:
    let spacing = if energy_variance > 0.0 {
        1.0 + (2.0 / energy_variance.sqrt()).min(0.5)
    } else {
        (t_max / t_min).powf(1.0 / (num_replicas - 1) as f64)
    };

    let mut temps = Vec::with_capacity(num_replicas);
    let mut t = t_min;
    for _ in 0..num_replicas {
        temps.push(t.min(t_max));
        t *= spacing;
    }

    // Ensure we reach t_max
    if let Some(last) = temps.last_mut() {
        *last = t_max;
    }

    temps
}

// ============================================================================
// Replica State
// ============================================================================

/// State of a single replica in the exchange system.
#[derive(Clone, Debug)]
pub struct ReplicaState<const N: usize> {
    /// The graph state.
    pub state: RamseyState<N>,
    /// Current energy.
    pub energy: usize,
    /// Temperature index (position in ladder).
    pub temp_idx: usize,
    /// Number of swap attempts.
    pub swap_attempts: u64,
    /// Number of successful swaps.
    pub swap_successes: u64,
}

impl<const N: usize> ReplicaState<N> {
    /// Creates a new replica state.
    pub fn new(state: RamseyState<N>, energy: usize, temp_idx: usize) -> Self {
        Self {
            state,
            energy,
            temp_idx,
            swap_attempts: 0,
            swap_successes: 0,
        }
    }

    /// Returns the swap acceptance rate.
    pub fn swap_rate(&self) -> f64 {
        if self.swap_attempts == 0 {
            0.0
        } else {
            self.swap_successes as f64 / self.swap_attempts as f64
        }
    }
}

// ============================================================================
// Swap Criterion
// ============================================================================

/// Computes the Metropolis acceptance probability for swapping two replicas.
///
/// For replicas i and j with energies E_i, E_j at temperatures T_i, T_j:
/// P(accept) = min(1, exp(ΔβΔE))
/// where Δβ = 1/T_i - 1/T_j and ΔE = E_i - E_j
#[inline]
pub fn swap_probability(energy_i: usize, energy_j: usize, temp_i: f64, temp_j: f64) -> f64 {
    let delta_beta = (1.0 / temp_i) - (1.0 / temp_j);
    let delta_energy = energy_i as f64 - energy_j as f64;
    let exponent = delta_beta * delta_energy;

    if exponent >= 0.0 {
        1.0
    } else {
        exponent.exp()
    }
}

/// Attempts to swap two replicas, returning true if swap occurred.
#[inline]
pub fn attempt_swap<const N: usize, R: Rng>(
    replica_i: &mut ReplicaState<N>,
    replica_j: &mut ReplicaState<N>,
    temps: &[f64],
    rng: &mut R,
) -> bool {
    let temp_i = temps[replica_i.temp_idx];
    let temp_j = temps[replica_j.temp_idx];

    let prob = swap_probability(replica_i.energy, replica_j.energy, temp_i, temp_j);

    replica_i.swap_attempts += 1;
    replica_j.swap_attempts += 1;

    if rng.random_bool(prob) {
        // Swap states and energies
        std::mem::swap(&mut replica_i.state, &mut replica_j.state);
        std::mem::swap(&mut replica_i.energy, &mut replica_j.energy);

        replica_i.swap_successes += 1;
        replica_j.swap_successes += 1;

        true
    } else {
        false
    }
}

// ============================================================================
// Shared Replica Exchange Coordinator
// ============================================================================

/// Coordinates replica exchange across workers.
///
/// Each worker registers its state periodically, and the coordinator
/// attempts swaps between adjacent temperature levels.
pub struct ReplicaExchangeCoordinator {
    /// Number of temperature levels.
    num_temps: usize,
    /// Temperature ladder.
    temperatures: Vec<f64>,
    /// Atomic counters for swap statistics.
    swap_attempts: Vec<AtomicU64>,
    swap_successes: Vec<AtomicU64>,
    /// Exchange interval (iterations between swap attempts).
    exchange_interval: u64,
}

impl ReplicaExchangeCoordinator {
    /// Creates a new coordinator with the given temperature ladder.
    pub fn new(temperatures: Vec<f64>, exchange_interval: u64) -> Arc<Self> {
        let num_temps = temperatures.len();
        Arc::new(Self {
            num_temps,
            temperatures,
            swap_attempts: (0..num_temps).map(|_| AtomicU64::new(0)).collect(),
            swap_successes: (0..num_temps).map(|_| AtomicU64::new(0)).collect(),
            exchange_interval,
        })
    }

    /// Creates a coordinator with geometric temperature ladder.
    pub fn with_geometric_ladder(
        t_min: f64,
        t_max: f64,
        num_temps: usize,
        exchange_interval: u64,
    ) -> Arc<Self> {
        let temps = geometric_ladder(t_min, t_max, num_temps);
        Self::new(temps, exchange_interval)
    }

    /// Returns the number of temperature levels.
    pub fn num_temps(&self) -> usize {
        self.num_temps
    }

    /// Returns the temperature for a given index.
    pub fn temperature(&self, idx: usize) -> f64 {
        self.temperatures.get(idx).copied().unwrap_or(1.0)
    }

    /// Returns the temperature ladder.
    pub fn temperatures(&self) -> &[f64] {
        &self.temperatures
    }

    /// Returns the exchange interval.
    pub fn exchange_interval(&self) -> u64 {
        self.exchange_interval
    }

    /// Records a swap attempt at the given temperature index.
    pub fn record_attempt(&self, temp_idx: usize) {
        if temp_idx < self.num_temps {
            self.swap_attempts[temp_idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a successful swap at the given temperature index.
    pub fn record_success(&self, temp_idx: usize) {
        if temp_idx < self.num_temps {
            self.swap_successes[temp_idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns swap statistics for all temperature levels.
    pub fn swap_stats(&self) -> Vec<(u64, u64, f64)> {
        (0..self.num_temps)
            .map(|i| {
                let attempts = self.swap_attempts[i].load(Ordering::Relaxed);
                let successes = self.swap_successes[i].load(Ordering::Relaxed);
                let rate = if attempts > 0 {
                    successes as f64 / attempts as f64
                } else {
                    0.0
                };
                (attempts, successes, rate)
            })
            .collect()
    }

    /// Assigns a worker to a temperature index based on worker_id.
    pub fn assign_temp_idx(&self, worker_id: usize) -> usize {
        worker_id % self.num_temps
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Estimates energy variance from a sample of energies.
pub fn estimate_energy_variance(energies: &[usize]) -> f64 {
    if energies.len() < 2 {
        return 1.0;
    }

    let mean = energies.iter().sum::<usize>() as f64 / energies.len() as f64;
    let variance = energies
        .iter()
        .map(|&e| (e as f64 - mean).powi(2))
        .sum::<f64>()
        / (energies.len() - 1) as f64;

    variance.max(1.0)
}

/// Returns the optimal number of temperature levels for a given number of workers.
///
/// Rule of thumb: sqrt(num_workers) to sqrt(num_workers) * 2 temperature levels
/// allows good coverage while ensuring each level has multiple workers.
pub fn optimal_num_temps(num_workers: usize) -> usize {
    let sqrt_w = (num_workers as f64).sqrt();
    (sqrt_w * 1.5).round().max(2.0) as usize
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_ladder_basic() {
        let ladder = geometric_ladder(1.0, 10.0, 5);
        assert_eq!(ladder.len(), 5);
        assert!((ladder[0] - 1.0).abs() < 0.001);
        assert!((ladder[4] - 10.0).abs() < 0.001);

        // Check geometric spacing
        for i in 1..ladder.len() {
            let ratio = ladder[i] / ladder[i - 1];
            let expected_ratio = (10.0_f64).powf(1.0 / 4.0);
            assert!(
                (ratio - expected_ratio).abs() < 0.001,
                "Ratio mismatch at {i}"
            );
        }
    }

    #[test]
    fn geometric_ladder_single() {
        let ladder = geometric_ladder(5.0, 20.0, 1);
        assert_eq!(ladder.len(), 1);
        assert!((ladder[0] - 5.0).abs() < 0.001);
    }

    #[test]
    fn swap_probability_same_temp() {
        // At same temperature, swap prob depends only on energy difference
        let prob = swap_probability(100, 100, 1.0, 1.0);
        assert!((prob - 1.0).abs() < 0.001, "Same energy should always swap");

        let prob = swap_probability(100, 200, 1.0, 1.0);
        assert!((prob - 1.0).abs() < 0.001, "Same temp, any energy should swap");
    }

    #[test]
    fn swap_probability_favorable() {
        // Lower energy at HIGHER temp (T_i=1.0) and higher energy at lower temp (T_j=0.5)
        // should always swap (moves lower energy to colder replica)
        let prob = swap_probability(50, 100, 1.0, 0.5);
        assert!(prob > 0.99, "Favorable swap should have high probability: {prob}");
    }

    #[test]
    fn swap_probability_unfavorable() {
        // Higher energy at higher temp is already the natural state, no need to swap
        // But the formula still gives some probability
        // Lower energy at lower temp (natural state) - checking swap probability
        let prob = swap_probability(50, 100, 0.5, 1.0);
        // This is actually unfavorable: lower energy already at lower temp
        // ΔE = -50, Δβ = 1, product = -50, exp(-50) ≈ 0
        assert!(prob < 0.01, "Unfavorable swap should have very low probability: {prob}");
    }

    #[test]
    fn attempt_swap_works() {
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;

        let mut rng = XorShiftRng::seed_from_u64(42);
        let temps = vec![1.0, 2.0];

        let state1 = RamseyState::<5>::empty();
        let state2 = RamseyState::<5>::complete();

        let mut replica1 = ReplicaState::new(state1, 100, 0);
        let mut replica2 = ReplicaState::new(state2, 50, 1);

        // Do many swap attempts
        let mut swaps = 0;
        for _ in 0..1000 {
            if attempt_swap(&mut replica1, &mut replica2, &temps, &mut rng) {
                swaps += 1;
            }
        }

        assert!(replica1.swap_attempts == 1000);
        assert!(replica2.swap_attempts == 1000);
        assert!(swaps > 0, "Should have some successful swaps");
        assert!(swaps < 1000, "Should not swap every time");
    }

    #[test]
    fn coordinator_assigns_temps() {
        let coord = ReplicaExchangeCoordinator::with_geometric_ladder(1.0, 10.0, 4, 1000);

        assert_eq!(coord.assign_temp_idx(0), 0);
        assert_eq!(coord.assign_temp_idx(1), 1);
        assert_eq!(coord.assign_temp_idx(4), 0); // Wraps
        assert_eq!(coord.assign_temp_idx(7), 3);
    }

    #[test]
    fn coordinator_records_stats() {
        let coord = ReplicaExchangeCoordinator::with_geometric_ladder(1.0, 10.0, 3, 1000);

        coord.record_attempt(0);
        coord.record_attempt(0);
        coord.record_success(0);
        coord.record_attempt(1);

        let stats = coord.swap_stats();
        assert_eq!(stats[0].0, 2); // attempts
        assert_eq!(stats[0].1, 1); // successes
        assert!((stats[0].2 - 0.5).abs() < 0.001); // rate
        assert_eq!(stats[1].0, 1);
        assert_eq!(stats[1].1, 0);
    }

    #[test]
    fn estimate_energy_variance_basic() {
        let energies = vec![100, 110, 90, 105, 95];
        let var = estimate_energy_variance(&energies);
        assert!(var > 0.0);

        // For these values, variance should be around 62.5
        assert!(var > 50.0 && var < 80.0);
    }

    #[test]
    fn optimal_num_temps_scales() {
        assert!(optimal_num_temps(1) >= 2);
        assert!(optimal_num_temps(16) >= 4);
        assert!(optimal_num_temps(128) >= 10);
        assert!(optimal_num_temps(128) <= 20);
    }

    #[test]
    fn adaptive_ladder_produces_increasing() {
        let ladder = adaptive_ladder(0.5, 20.0, 8, 100.0);

        assert_eq!(ladder.len(), 8);
        for i in 1..ladder.len() {
            assert!(
                ladder[i] >= ladder[i - 1],
                "Ladder should be non-decreasing"
            );
        }
    }

    #[test]
    fn replica_state_swap_rate() {
        let state = RamseyState::<5>::empty();
        let mut replica = ReplicaState::new(state, 100, 0);

        assert_eq!(replica.swap_rate(), 0.0);

        replica.swap_attempts = 10;
        replica.swap_successes = 3;
        assert!((replica.swap_rate() - 0.3).abs() < 0.001);
    }
}


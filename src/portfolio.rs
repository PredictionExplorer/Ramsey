//! Portfolio Search: Heterogeneous parallel search with island model.
//!
//! Instead of running N identical search chains, we run N different configurations
//! that explore the search space in complementary ways. This is crucial for
//! escaping local minima and utilizing many cores effectively.
//!
//! Key ideas:
//! - Different temperatures (hot explorers vs cold exploiters)
//! - Different objective weights (C4-focused vs IS-focused vs balanced)
//! - Different move strategies (conservative vs aggressive)
//! - Periodic migration of good solutions between islands

use rand::Rng;

// ============================================================================
// Island Configuration
// ============================================================================

/// Configuration for a single island in the portfolio.
///
/// Each island has different hyperparameters to encourage diverse exploration.
#[derive(Clone, Debug)]
pub struct IslandConfig {
    /// Island identifier.
    pub id: usize,
    /// Name for logging.
    pub name: &'static str,
    /// Temperature multiplier (1.0 = default, >1 = hotter, <1 = colder).
    pub temp_multiplier: f64,
    /// C4 weight multiplier (higher = more focus on eliminating C4s).
    pub c4_weight_multiplier: f64,
    /// IS weight multiplier (higher = more focus on breaking ISs).
    pub is_weight_multiplier: f64,
    /// Compound move probability.
    pub compound_move_prob: f64,
    /// Degree bias probability.
    pub degree_bias_prob: f64,
    /// Tabu tenure multiplier.
    pub tabu_multiplier: f64,
    /// Whether to use coverage-based move selection.
    pub use_coverage_moves: bool,
    /// Whether to prefer C4-safe moves when near feasible.
    pub prefer_c4_safe: bool,
    /// Kick strength multiplier.
    pub kick_strength_multiplier: f64,
    /// Reheat temperature multiplier.
    pub reheat_multiplier: f64,
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self {
            id: 0,
            name: "default",
            temp_multiplier: 1.0,
            c4_weight_multiplier: 1.0,
            is_weight_multiplier: 1.0,
            compound_move_prob: 0.15,
            degree_bias_prob: 0.4,
            tabu_multiplier: 1.0,
            use_coverage_moves: true,
            prefer_c4_safe: true,
            kick_strength_multiplier: 1.0,
            reheat_multiplier: 1.0,
        }
    }
}

impl IslandConfig {
    /// Creates a diverse portfolio of island configurations.
    ///
    /// The portfolio is designed to cover different search strategies:
    /// - Exploiters (cold, focused)
    /// - Explorers (hot, diverse)
    /// - Specialists (focused on specific constraints)
    /// - Generalists (balanced)
    pub fn portfolio(num_islands: usize) -> Vec<IslandConfig> {
        let mut configs = Vec::with_capacity(num_islands);

        // Strategy distribution:
        // 25% cold exploiters
        // 25% hot explorers
        // 25% C4-focused
        // 25% IS-focused with coverage

        for i in 0..num_islands {
            let phase = i % 8;
            let config = match phase {
                // Cold exploiter - fine-tunes near optima
                0 => IslandConfig {
                    id: i,
                    name: "cold_exploiter",
                    temp_multiplier: 0.3,
                    c4_weight_multiplier: 1.0,
                    is_weight_multiplier: 1.0,
                    compound_move_prob: 0.05,
                    degree_bias_prob: 0.3,
                    tabu_multiplier: 2.0, // Longer tabu
                    use_coverage_moves: true,
                    prefer_c4_safe: true,
                    kick_strength_multiplier: 0.5,
                    reheat_multiplier: 0.5,
                },
                // Hot explorer - escapes local minima
                1 => IslandConfig {
                    id: i,
                    name: "hot_explorer",
                    temp_multiplier: 3.0,
                    c4_weight_multiplier: 1.0,
                    is_weight_multiplier: 1.0,
                    compound_move_prob: 0.3,
                    degree_bias_prob: 0.5,
                    tabu_multiplier: 0.5, // Shorter tabu
                    use_coverage_moves: false,
                    prefer_c4_safe: false,
                    kick_strength_multiplier: 2.0,
                    reheat_multiplier: 2.0,
                },
                // C4 terminator - aggressively removes cycles
                2 => IslandConfig {
                    id: i,
                    name: "c4_terminator",
                    temp_multiplier: 1.0,
                    c4_weight_multiplier: 3.0, // High C4 penalty
                    is_weight_multiplier: 0.5,
                    compound_move_prob: 0.2,
                    degree_bias_prob: 0.6, // High-degree vertices often in C4s
                    tabu_multiplier: 1.0,
                    use_coverage_moves: false,
                    prefer_c4_safe: true,
                    kick_strength_multiplier: 1.0,
                    reheat_multiplier: 1.0,
                },
                // IS hunter - uses coverage to break many ISs
                3 => IslandConfig {
                    id: i,
                    name: "is_hunter",
                    temp_multiplier: 1.0,
                    c4_weight_multiplier: 0.5,
                    is_weight_multiplier: 2.0, // High IS penalty
                    compound_move_prob: 0.1,
                    degree_bias_prob: 0.3,
                    tabu_multiplier: 1.5,
                    use_coverage_moves: true, // Key feature
                    prefer_c4_safe: true,
                    kick_strength_multiplier: 1.0,
                    reheat_multiplier: 1.0,
                },
                // Balanced warm
                4 => IslandConfig {
                    id: i,
                    name: "balanced_warm",
                    temp_multiplier: 1.5,
                    c4_weight_multiplier: 1.0,
                    is_weight_multiplier: 1.0,
                    compound_move_prob: 0.15,
                    degree_bias_prob: 0.4,
                    tabu_multiplier: 1.0,
                    use_coverage_moves: true,
                    prefer_c4_safe: true,
                    kick_strength_multiplier: 1.5,
                    reheat_multiplier: 1.5,
                },
                // Compound mover - relies on multi-edge moves
                5 => IslandConfig {
                    id: i,
                    name: "compound_mover",
                    temp_multiplier: 1.2,
                    c4_weight_multiplier: 1.0,
                    is_weight_multiplier: 1.0,
                    compound_move_prob: 0.4, // High compound probability
                    degree_bias_prob: 0.4,
                    tabu_multiplier: 0.5,
                    use_coverage_moves: true,
                    prefer_c4_safe: false,
                    kick_strength_multiplier: 1.5,
                    reheat_multiplier: 1.0,
                },
                // Strict C4-safe - never creates C4s
                6 => IslandConfig {
                    id: i,
                    name: "strict_c4_safe",
                    temp_multiplier: 0.8,
                    c4_weight_multiplier: 100.0, // Essentially forbid C4s
                    is_weight_multiplier: 1.0,
                    compound_move_prob: 0.1,
                    degree_bias_prob: 0.3,
                    tabu_multiplier: 1.5,
                    use_coverage_moves: true,
                    prefer_c4_safe: true,
                    kick_strength_multiplier: 0.5,
                    reheat_multiplier: 0.5,
                },
                // Aggressive kicker - frequent restarts
                _ => IslandConfig {
                    id: i,
                    name: "aggressive_kicker",
                    temp_multiplier: 2.0,
                    c4_weight_multiplier: 1.0,
                    is_weight_multiplier: 1.0,
                    compound_move_prob: 0.2,
                    degree_bias_prob: 0.5,
                    tabu_multiplier: 0.3,
                    use_coverage_moves: false,
                    prefer_c4_safe: false,
                    kick_strength_multiplier: 3.0, // Strong kicks
                    reheat_multiplier: 3.0,
                },
            };
            configs.push(config);
        }

        configs
    }

    /// Creates a random configuration for additional diversity.
    pub fn random<R: Rng>(rng: &mut R, id: usize) -> Self {
        Self {
            id,
            name: "random",
            temp_multiplier: rng.random_range(0.2..4.0),
            c4_weight_multiplier: rng.random_range(0.3..5.0),
            is_weight_multiplier: rng.random_range(0.3..3.0),
            compound_move_prob: rng.random_range(0.05..0.4),
            degree_bias_prob: rng.random_range(0.2..0.7),
            tabu_multiplier: rng.random_range(0.3..3.0),
            use_coverage_moves: rng.random_bool(0.5),
            prefer_c4_safe: rng.random_bool(0.7),
            kick_strength_multiplier: rng.random_range(0.5..3.0),
            reheat_multiplier: rng.random_range(0.5..3.0),
        }
    }

    /// Applies temperature multiplier to base temperature.
    pub fn apply_temp(&self, base_temp: f64) -> f64 {
        base_temp * self.temp_multiplier
    }

    /// Applies C4 weight multiplier to base C4 weight.
    pub fn apply_c4_weight(&self, base_weight: usize) -> usize {
        ((base_weight as f64) * self.c4_weight_multiplier).round() as usize
    }

    /// Applies IS weight multiplier to base IS weight.
    pub fn apply_is_weight(&self, base_weight: usize) -> usize {
        ((base_weight as f64) * self.is_weight_multiplier).round() as usize
    }

    /// Applies tabu size multiplier to base tabu size.
    pub fn apply_tabu_size(&self, base_size: usize) -> usize {
        ((base_size as f64) * self.tabu_multiplier).round() as usize
    }

    /// Applies kick strength multiplier to base kick strength.
    pub fn apply_kick_strength(&self, base_strength: usize) -> usize {
        ((base_strength as f64) * self.kick_strength_multiplier).round().max(1.0) as usize
    }

    /// Applies reheat temperature multiplier to base reheat temperature.
    pub fn apply_reheat_temp(&self, base_temp: f64) -> f64 {
        base_temp * self.reheat_multiplier
    }
}

// ============================================================================
// Migration Strategy
// ============================================================================

/// Strategy for migrating solutions between islands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationStrategy {
    /// Ring topology: island i sends to island (i+1) % n.
    Ring,
    /// Random: each island sends to a random other island.
    Random,
    /// Broadcast best: best solution goes to all islands.
    BroadcastBest,
    /// Tournament: pairs of islands compete, loser adopts winner's solution.
    Tournament,
}

impl Default for MigrationStrategy {
    fn default() -> Self {
        Self::Ring
    }
}

impl MigrationStrategy {
    /// Determines the target island for migration.
    pub fn target<R: Rng>(&self, source: usize, num_islands: usize, rng: &mut R) -> usize {
        match self {
            Self::Ring => (source + 1) % num_islands,
            Self::Random => {
                let mut target = rng.random_range(0..num_islands);
                while target == source && num_islands > 1 {
                    target = rng.random_range(0..num_islands);
                }
                target
            }
            Self::BroadcastBest | Self::Tournament => {
                // These are handled specially at the coordinator level
                (source + 1) % num_islands
            }
        }
    }
}

// ============================================================================
// Portfolio Statistics
// ============================================================================

/// Statistics for monitoring portfolio performance.
#[derive(Clone, Debug, Default)]
pub struct PortfolioStats {
    /// Best energy per island type.
    pub best_by_type: Vec<(&'static str, usize)>,
    /// Number of improvements per island type.
    pub improvements_by_type: Vec<(&'static str, u64)>,
    /// Number of migrations.
    pub migrations: u64,
    /// Solutions found by island type.
    pub solutions_by_type: Vec<(&'static str, u64)>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_creates_diverse_configs() {
        let configs = IslandConfig::portfolio(16);

        assert_eq!(configs.len(), 16);

        // Check diversity of temp_multiplier
        let temps: Vec<f64> = configs.iter().map(|c| c.temp_multiplier).collect();
        let min_temp = temps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_temp = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_temp > min_temp * 2.0, "Should have diverse temperatures");

        // Check diversity of c4_weight_multiplier
        let weights: Vec<f64> = configs.iter().map(|c| c.c4_weight_multiplier).collect();
        let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_w > min_w * 2.0, "Should have diverse C4 weights");

        // Check that all IDs are unique
        let ids: Vec<usize> = configs.iter().map(|c| c.id).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "IDs should be unique");
            }
        }
    }

    #[test]
    fn portfolio_has_exploiters_and_explorers() {
        let configs = IslandConfig::portfolio(8);

        let has_cold = configs.iter().any(|c| c.temp_multiplier < 0.5);
        let has_hot = configs.iter().any(|c| c.temp_multiplier > 2.0);

        assert!(has_cold, "Should have cold exploiters");
        assert!(has_hot, "Should have hot explorers");
    }

    #[test]
    fn config_apply_multipliers() {
        let config = IslandConfig {
            temp_multiplier: 2.0,
            c4_weight_multiplier: 3.0,
            is_weight_multiplier: 0.5,
            tabu_multiplier: 1.5,
            kick_strength_multiplier: 2.0,
            reheat_multiplier: 0.5,
            ..Default::default()
        };

        assert!((config.apply_temp(10.0) - 20.0).abs() < 0.001);
        assert_eq!(config.apply_c4_weight(50), 150);
        assert_eq!(config.apply_is_weight(100), 50);
        assert_eq!(config.apply_tabu_size(64), 96);
        assert_eq!(config.apply_kick_strength(8), 16);
        assert!((config.apply_reheat_temp(2.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn random_config_in_valid_ranges() {
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;

        let mut rng = XorShiftRng::seed_from_u64(42);

        for i in 0..100 {
            let config = IslandConfig::random(&mut rng, i);

            assert!(config.temp_multiplier >= 0.2 && config.temp_multiplier <= 4.0);
            assert!(config.c4_weight_multiplier >= 0.3 && config.c4_weight_multiplier <= 5.0);
            assert!(config.compound_move_prob >= 0.05 && config.compound_move_prob <= 0.4);
            assert_eq!(config.id, i);
        }
    }

    #[test]
    fn migration_ring_wraps_around() {
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;

        let mut rng = XorShiftRng::seed_from_u64(42);
        let strategy = MigrationStrategy::Ring;

        assert_eq!(strategy.target(0, 4, &mut rng), 1);
        assert_eq!(strategy.target(3, 4, &mut rng), 0); // Wraps
        assert_eq!(strategy.target(7, 8, &mut rng), 0); // Wraps
    }

    #[test]
    fn migration_random_avoids_self() {
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;

        let mut rng = XorShiftRng::seed_from_u64(42);
        let strategy = MigrationStrategy::Random;

        for _ in 0..100 {
            let source = rng.random_range(0..8);
            let target = strategy.target(source, 8, &mut rng);
            assert_ne!(source, target, "Random migration should not target self");
        }
    }

    #[test]
    fn portfolio_covers_all_strategies() {
        let configs = IslandConfig::portfolio(8);

        let has_coverage = configs.iter().any(|c| c.use_coverage_moves);
        let has_no_coverage = configs.iter().any(|c| !c.use_coverage_moves);
        let has_c4_safe = configs.iter().any(|c| c.prefer_c4_safe);
        let has_compound = configs.iter().any(|c| c.compound_move_prob > 0.3);

        assert!(has_coverage, "Should have coverage-based islands");
        assert!(has_no_coverage, "Should have non-coverage islands");
        assert!(has_c4_safe, "Should have C4-safe islands");
        assert!(has_compound, "Should have compound-move islands");
    }

    #[test]
    fn default_config_is_reasonable() {
        let config = IslandConfig::default();

        assert!((config.temp_multiplier - 1.0).abs() < 0.001);
        assert!((config.c4_weight_multiplier - 1.0).abs() < 0.001);
        assert!(config.compound_move_prob > 0.0 && config.compound_move_prob < 1.0);
        assert!(config.use_coverage_moves);
        assert!(config.prefer_c4_safe);
    }
}


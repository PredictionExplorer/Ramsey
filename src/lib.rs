//! # Ramsey Search Engine
//!
//! A high-performance Rust library for finding critical graphs in Ramsey theory.
//!
//! This crate provides:
//! - A compact bitset graph state with **incremental** tracking of 4-cycles \(C_4\).
//! - A fast, **exact** oracle for detecting independent sets of a given size
//!   (implemented as a clique search in the complement graph with greedy-coloring bounds).
//! - A parallel stochastic search driver (LAHC + SA hybrid).
//!
//! ## Quick Start
//!
//! ```no_run
//! use ramsey::search::{run_search, SearchConfig};
//!
//! // Run a search for R(C4, K5) > 12
//! let cfg = SearchConfig {
//!     chains: 8,
//!     seed: Some(12345),
//!     ..Default::default()
//! };
//! run_search::<13>(&cfg, 5);
//! ```
//!
//! ## Validating Known Witnesses
//!
//! ```
//! use ramsey::validate::validate_known_graphs;
//!
//! // Validate bundled witness graphs
//! validate_known_graphs().expect("all witnesses should be valid");
//! ```
//!
//! ## Working with Graphs Directly
//!
//! ```
//! use ramsey::graph::RamseyState;
//! use ramsey::iset::IndependentSetOracle;
//!
//! // Create an empty 6-vertex graph
//! let mut state = RamseyState::<6>::empty();
//!
//! // Add some edges
//! state.flip_edge(0, 1);
//! state.flip_edge(1, 2);
//! state.flip_edge(2, 3);
//!
//! // Check properties
//! assert_eq!(state.c4_count(), 0);
//! assert_eq!(state.edge_count(), 3);
//!
//! // Check for independent sets
//! let mut oracle = IndependentSetOracle::<6>::new();
//! assert!(oracle.has_independent_set_of_size(state.adj(), 3));
//! ```
//!
//! ## Modules
//!
//! - [`graph`]: Graph state with incremental C4 tracking and parsing utilities.
//! - [`iset`]: Exact independent-set oracle using branch-and-bound with coloring bounds.
//! - [`search`]: Parallel stochastic search driver (LAHC + SA hybrid).
//! - [`validate`]: Deterministic validation of witness graphs.
//!
//! ## Performance Notes
//!
//! - The graph representation uses `u64` bitsets, limiting graphs to 64 vertices.
//! - C4 score updates are O(degree) instead of O(n²) due to incremental tracking.
//! - The independent-set oracle uses Tomita-style pruning with greedy coloring bounds.
//! - For maximum performance, compile with: `RUSTFLAGS="-C target-cpu=native" cargo build --release`

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::inline_always)] // Intentional for hot-path code
#![allow(clippy::many_single_char_names)] // Mathematical variable names
#![allow(clippy::needless_range_loop)] // Often clearer for matrix indexing
#![allow(clippy::doc_markdown)] // LaTeX-style notation in docs
#![allow(clippy::multiple_crate_versions)] // Cargo.lock management is external

pub mod graph;
pub mod iset;
pub mod search;
pub mod validate;

/// Re-export commonly used types for convenience.
pub mod prelude {
    pub use crate::graph::{parse_adjacency_matrix, RamseyState};
    pub use crate::iset::IndependentSetOracle;
    pub use crate::search::{run_record_search, run_search, SearchConfig};
    pub use crate::validate::validate_known_graphs;
}

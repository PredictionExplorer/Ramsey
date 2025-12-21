# Ramsey Search: Finding R(C₄, K₁₁)

A high-performance Rust engine for searching critical graphs in Ramsey theory. The primary target is proving lower bounds for R(C₄, K₁₁) by finding a 39-vertex graph with:

1. **No C₄ cycles**: No four vertices form a 4-cycle.
2. **No K₁₁ independent set**: Every set of 11 vertices contains at least one edge.

If such a graph is found, it proves R(C₄, K₁₁) > 39.

## Features

### Performance Optimizations

- **Incremental C₄ Tracking**: Edge flips update cycle counts in O(degree) instead of O(n²).
- **Precomputed Lookup Tables**: Binomial coefficients cached at compile time.
- **Bit-Parallel Operations**: All set operations use 64-bit bitsets with hardware popcount.
- **Cache-Friendly Layout**: Common neighbor matrix stored contiguously.
- **Stack-Allocated LAHC Buffer**: Avoids heap allocation in hot loops.
- **LTO + Native CPU**: Release profile enables full link-time optimization.

### Algorithmic Improvements

- **Exact Independent Set Oracle**: Branch-and-bound with greedy coloring bounds (Tomita-style pruning).
- **Staged Evaluation**: Expensive IS checks only run when graph is C₄-free.
- **Greedy IS Heuristic**: Fast lower-bound check before exact verification.
- **C₄-Guided Moves**: When C₄s exist, preferentially remove edges in cycles.
- **IS-Guided Moves**: When C₄-free, add edges inside violating independent sets, minimizing new C₄s.

### Hybrid Metaheuristic

- **Late Acceptance Hill Climbing (LAHC)**: Accept moves better than history buffer.
- **Simulated Annealing (SA)**: Probabilistic acceptance with temperature schedule.
- **Adaptive Reheating**: Temperature resets when stuck in local minima.

### Parallel Architecture

- **Rayon-Powered**: Independent search chains across all CPU cores.
- **Auto-Detection**: Automatically detects logical CPU cores and scales chains (default: cores \* 2).
- **First-to-Win**: Atomic flag stops all workers when solution found.
- **Deterministic Seeds**: Optional seed for reproducible runs.

## Quick Start

### Prerequisites

- Rust 1.70+ (stable)
- Cargo

### Build & Run

```bash
# Fast validation of bundled witness graphs
cargo run --release -- --test

# Run record search (39 vertices, K=11)
cargo run --release

# Run smaller test cases
cargo run --release -- --case 13 5 --workers 8

# Maximum performance (native CPU optimizations)
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/ramsey
```

### Command-Line Options

```
ramsey [OPTIONS]

OPTIONS:
  --case N K         Run a specific case (supported: 6 3, 9 4, 13 5, 39 11)
  --workers N        Number of parallel search chains (default: 200)
  --p P              Initial edge probability (default: 0.18)
  --seed SEED        Deterministic base seed for reproducibility
  --test/--validate  Validate bundled witness graphs (fast, deterministic)
  --help             Show this help message
```

## Test Suite

The project includes 44 comprehensive unit tests covering:

- **Incremental Invariants**: Flip operations match full recomputation (5,000+ random flips)
- **Reversibility**: Double-flip restores original state
- **Edge Cases**: Empty graphs, complete graphs, paths, stars, cycles
- **Parser Validation**: Rejects malformed, non-square, asymmetric matrices
- **Oracle Correctness**: Exact IS matches brute-force on small random graphs
- **Greedy Validity**: Greedy IS always returns valid independent sets
- **Witness Validation**: Bundled graphs are verified C₄-free with small α

Run tests:

```bash
cargo test
```

## Library Usage

```rust
use ramsey::prelude::*;

// Create and manipulate graphs
let mut state = RamseyState::<10>::empty();
state.flip_edge(0, 1);
state.flip_edge(1, 2);
println!("C4 count: {}", state.c4_count());

// Check for independent sets
let mut oracle = IndependentSetOracle::<10>::new();
if oracle.has_independent_set_of_size(state.adj(), 5) {
    println!("Has IS of size 5");
}

// Run a search
let cfg = SearchConfig {
    chains: 8,
    seed: Some(42),
    ..Default::default()
};
run_search::<13, 5>(&cfg);
```

## Bundled Witness Graphs

The repository includes verified witness graphs for known Ramsey bounds:

| File               | N   | K   | Proves         |
| ------------------ | --- | --- | -------------- |
| `graph_n6_k3.txt`  | 6   | 3   | R(C₄, K₃) > 5  |
| `graph_n9_k4.txt`  | 9   | 4   | R(C₄, K₄) > 8  |
| `graph_n13_k5.txt` | 13  | 5   | R(C₄, K₅) > 12 |

## Output

When a solution is found, the adjacency matrix is saved to `graph_nN_kK.txt`. This file serves as mathematical proof of the lower bound.

## Architecture

```
src/
├── lib.rs       # Crate root with documentation
├── graph.rs     # RamseyState with incremental C4 tracking
├── iset.rs      # Exact independent set oracle
├── search.rs    # Parallel LAHC/SA search driver
├── validate.rs  # Witness graph validation
└── main.rs      # CLI entrypoint
```

## License

MIT

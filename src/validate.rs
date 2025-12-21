//! Fast deterministic validation of known small witness graphs.

use crate::graph::{parse_adjacency_matrix, RamseyState};
use crate::iset::IndependentSetOracle;

// ============================================================================
// Public API
// ============================================================================

/// Validates the bundled witness graphs for:
/// - `R(C4, K3) > 5`  via a 6-vertex witness
/// - `R(C4, K4) > 8`  via a 9-vertex witness
/// - `R(C4, K5) > 12` via a 13-vertex witness
///
/// # Errors
/// Returns an error message if any bundled graph fails validation.
pub fn validate_known_graphs() -> Result<(), String> {
    validate_case::<6, 3>(include_str!("../graph_n6_k3.txt"), "graph_n6_k3.txt")?;
    validate_case::<9, 4>(include_str!("../graph_n9_k4.txt"), "graph_n9_k4.txt")?;
    validate_case::<13, 5>(include_str!("../graph_n13_k5.txt"), "graph_n13_k5.txt")?;
    Ok(())
}

/// Validates a single witness graph.
///
/// Returns `Ok(())` if the graph is C4-free and has no independent set of size `K`.
///
/// # Errors
/// Returns an error message if parsing fails or the graph violates constraints.
pub fn validate_witness<const N: usize, const K: usize>(
    text: &str,
    name: &str,
) -> Result<(), String> {
    validate_case::<N, K>(text, name)
}

/// Validates an adjacency matrix against the constraints.
///
/// # Errors
/// Returns an error message if the graph contains a C4 or has an independent set of size K.
pub fn validate_adj<const N: usize, const K: usize>(adj: &[u64; N]) -> Result<(), String> {
    let state = RamseyState::<N>::from_adj(*adj);

    if state.c4_score_twice() != 0 {
        return Err(format!(
            "graph has {} C4 cycles (expected 0)",
            state.c4_count()
        ));
    }

    let mut oracle = IndependentSetOracle::<N>::new();
    if oracle.has_independent_set_of_size(adj, K) {
        return Err(format!(
            "graph has an independent set of size {K} (expected none)"
        ));
    }

    Ok(())
}

// ============================================================================
// Internal
// ============================================================================

fn validate_case<const N: usize, const K: usize>(text: &str, name: &str) -> Result<(), String> {
    let parsed = parse_adjacency_matrix(text).map_err(|e| format!("{name}: {e}"))?;
    let adj = parsed.to_fixed::<N>().map_err(|e| format!("{name}: {e}"))?;
    let state = RamseyState::<N>::from_adj(adj);

    if state.c4_score_twice() != 0 {
        return Err(format!(
            "{name}: expected no C4, but found c4_count={}",
            state.c4_count()
        ));
    }

    let mut oracle = IndependentSetOracle::<N>::new();
    if oracle.has_independent_set_of_size(state.adj(), K) {
        return Err(format!(
            "{name}: expected no independent set of size {K}, but one exists"
        ));
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_graphs_are_valid_witnesses() {
        validate_known_graphs().unwrap();
    }

    #[test]
    fn validate_adj_rejects_graph_with_c4() {
        // C4: 0-1-2-3-0
        let adj: [u64; 4] = [0b1010, 0b0101, 0b1010, 0b0101];
        let result = validate_adj::<4, 2>(&adj);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("C4"));
    }

    #[test]
    fn validate_adj_rejects_large_independent_set() {
        // Empty graph: alpha = 4
        let adj: [u64; 4] = [0, 0, 0, 0];
        let result = validate_adj::<4, 3>(&adj);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("independent set"));
    }

    #[test]
    fn validate_adj_accepts_valid_witness() {
        // Complete graph K4: C4-free is false for K4, let's use a path instead
        // Path 0-1-2-3: no C4, alpha = 2 (take 0, 2)
        let adj: [u64; 4] = [
            0b0010, // 0 -- 1
            0b0101, // 1 -- 0, 2
            0b1010, // 2 -- 1, 3
            0b0100, // 3 -- 2
        ];
        // This is C4-free and has alpha = 2
        let result = validate_adj::<4, 3>(&adj);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_witness_parses_and_checks() {
        let text = "0100\n1010\n0101\n0010\n";
        let result = validate_witness::<4, 3>(text, "test_path");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_witness_rejects_malformed_input() {
        let text = "0100\n101\n0101\n0010\n"; // Non-square
        let result = validate_witness::<4, 3>(text, "test_bad");
        assert!(result.is_err());
    }

    #[test]
    fn bundled_graph_n6_k3_has_correct_size() {
        let text = include_str!("../graph_n6_k3.txt");
        let parsed = parse_adjacency_matrix(text).unwrap();
        assert_eq!(parsed.n, 6);
    }

    #[test]
    fn bundled_graph_n9_k4_has_correct_size() {
        let text = include_str!("../graph_n9_k4.txt");
        let parsed = parse_adjacency_matrix(text).unwrap();
        assert_eq!(parsed.n, 9);
    }

    #[test]
    fn bundled_graph_n13_k5_has_correct_size() {
        let text = include_str!("../graph_n13_k5.txt");
        let parsed = parse_adjacency_matrix(text).unwrap();
        assert_eq!(parsed.n, 13);
    }
}

//! Graph state and utilities specialized for small Ramsey searches (currently \(N \le 64\)).

use rand::Rng;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

// ============================================================================
// Compile-time lookup tables
// ============================================================================

/// Precomputed `choose(n, 2) = n*(n-1)/2` for n in 0..=64.
/// Used to avoid repeated division in the hot path.
const CHOOSE2: [usize; 65] = {
    let mut table = [0usize; 65];
    let mut i = 0usize;
    while i < 65 {
        table[i] = (i * i.saturating_sub(1)) / 2;
        i += 1;
    }
    table
};

/// Returns `n * (n-1) / 2` via lookup.
#[inline(always)]
const fn choose2(n: usize) -> usize {
    debug_assert!(n < CHOOSE2.len());
    CHOOSE2[n]
}

/// Returns a mask with the lowest `N` bits set.
#[inline(always)]
pub const fn all_bits<const N: usize>() -> u64 {
    if N >= 64 {
        u64::MAX
    } else {
        (1u64 << N) - 1
    }
}

#[inline(always)]
const fn bit(v: usize) -> u64 {
    1u64 << v
}

// ============================================================================
// RamseyState
// ============================================================================

/// A graph state optimized for R(C4, K_n) searches.
///
/// Representation:
/// - `adj[v]` is the neighbor bitset of vertex `v`.
/// - `common_neighbors[i][j]` stores \(|N(i) \cap N(j)|\).
/// - `c4_score_twice` is \(\sum_{i<j} \binom{|N(i)\cap N(j)|}{2}\).
///   This value is **twice** the number of 4-cycles in the graph; importantly,
///   `c4_score_twice == 0` iff the graph contains no 4-cycles.
#[derive(Clone, Debug)]
pub struct RamseyState<const N: usize> {
    adj: [u64; N],
    common_neighbors: [[u8; N]; N],
    c4_score_twice: usize,
}

impl<const N: usize> RamseyState<N> {
    /// Creates a new state from a provided adjacency bitset array.
    ///
    /// # Panics
    /// Panics in debug builds if the input adjacency contains out-of-range bits,
    /// self-loops, or is not symmetric.
    pub fn from_adj(adj: [u64; N]) -> Self {
        debug_assert!(N <= 64, "This implementation assumes N <= 64");
        let mask = all_bits::<N>();

        // Validate invariants (debug-only; this is hot-path code in release builds).
        for i in 0..N {
            debug_assert_eq!(adj[i] & !mask, 0, "adj contains bits outside N");
            debug_assert_eq!((adj[i] >> i) & 1, 0, "self-loop at vertex {i}");
        }
        for i in 0..N {
            for j in (i + 1)..N {
                let aij = (adj[i] >> j) & 1;
                let aji = (adj[j] >> i) & 1;
                debug_assert_eq!(aij, aji, "adj is not symmetric at ({i},{j})");
            }
        }

        let (common_neighbors, c4_score_twice) = compute_common_neighbors_and_c4_score_twice(&adj);
        Self {
            adj,
            common_neighbors,
            c4_score_twice,
        }
    }

    /// Creates an empty graph (no edges).
    pub fn empty() -> Self {
        Self {
            adj: [0u64; N],
            common_neighbors: [[0u8; N]; N],
            c4_score_twice: 0,
        }
    }

    /// Creates a complete graph (all edges present).
    pub fn complete() -> Self {
        let mask = all_bits::<N>();
        let mut adj = [0u64; N];
        for i in 0..N {
            adj[i] = mask & !bit(i); // all bits except self
        }
        Self::from_adj(adj)
    }

    /// Initializes a random graph and precomputes incremental bookkeeping.
    pub fn new_random<R: Rng>(rng: &mut R, p: f64) -> Self {
        debug_assert!(N <= 64, "This implementation assumes N <= 64");
        debug_assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");

        let mut adj = [0u64; N];
        for i in 0..N {
            for j in (i + 1)..N {
                if rng.random_bool(p) {
                    adj[i] |= bit(j);
                    adj[j] |= bit(i);
                }
            }
        }
        Self::from_adj(adj)
    }

    /// Returns a reference to the adjacency bitsets.
    #[inline(always)]
    pub fn adj(&self) -> &[u64; N] {
        &self.adj
    }

    /// Returns whether the edge `(u, v)` exists.
    #[inline(always)]
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        debug_assert!(u < N && v < N);
        (self.adj[u] & bit(v)) != 0
    }

    /// Returns the degree of vertex `v`.
    #[inline(always)]
    pub fn degree(&self, v: usize) -> u32 {
        debug_assert!(v < N);
        self.adj[v].count_ones()
    }

    /// Returns the total number of edges in the graph.
    #[inline]
    pub fn edge_count(&self) -> usize {
        let mut sum = 0u32;
        for i in 0..N {
            sum += self.adj[i].count_ones();
        }
        (sum as usize) / 2
    }

    /// Returns the current "twice-count" C4 score.
    ///
    /// `0` means there are no 4-cycles; positive means at least one 4-cycle exists.
    #[inline(always)]
    pub fn c4_score_twice(&self) -> usize {
        self.c4_score_twice
    }

    /// Returns the exact number of 4-cycles in the graph.
    ///
    /// This is intended for logging/debugging; the search can safely use `c4_score_twice() == 0`
    /// as the constraint check.
    #[inline]
    pub fn c4_count(&self) -> usize {
        self.c4_score_twice / 2
    }

    /// Returns the number of common neighbors between vertices `u` and `v`.
    #[inline(always)]
    pub fn common_neighbor_count(&self, u: usize, v: usize) -> u8 {
        debug_assert!(u < N && v < N);
        self.common_neighbors[u][v]
    }

    /// Returns a reference to the common neighbors matrix.
    #[inline(always)]
    pub fn common_neighbors(&self) -> &[[u8; N]; N] {
        &self.common_neighbors
    }

    /// Computes the C4 delta for adding edge (u,v) WITHOUT modifying state.
    ///
    /// Returns the change in c4_score_twice that would occur if edge (u,v) were added.
    /// Positive means more C4s would be created.
    ///
    /// This is O(deg(u) + deg(v)) but avoids the overhead of flip + cache invalidation + unflip.
    #[inline]
    pub fn c4_delta_if_add(&self, u: usize, v: usize) -> isize {
        debug_assert!(u < N && v < N && u != v);
        debug_assert!(!self.has_edge(u, v), "Edge already exists");

        let neighbors_u = self.adj[u];
        let neighbors_v = self.adj[v];
        let u_mask = bit(u);
        let v_mask = bit(v);

        let mut delta: isize = 0;

        // When adding (u,v):
        // - For each w in N(v) \ {u}: common[u][w] increases by 1
        // - For each w in N(u) \ {v}: common[v][w] increases by 1
        // Delta to c4_score_twice for increasing common[x][y] from old to old+1 is: old

        let mut t = neighbors_v & !u_mask;
        while t != 0 {
            let w = t.trailing_zeros() as usize;
            t &= t - 1;
            delta += self.common_neighbors[u][w] as isize;
        }

        let mut t = neighbors_u & !v_mask;
        while t != 0 {
            let w = t.trailing_zeros() as usize;
            t &= t - 1;
            delta += self.common_neighbors[v][w] as isize;
        }

        delta
    }

    /// Computes the C4 delta for removing edge (u,v) WITHOUT modifying state.
    ///
    /// Returns the change in c4_score_twice that would occur if edge (u,v) were removed.
    /// Negative means fewer C4s (improvement).
    #[inline]
    pub fn c4_delta_if_remove(&self, u: usize, v: usize) -> isize {
        debug_assert!(u < N && v < N && u != v);
        debug_assert!(self.has_edge(u, v), "Edge doesn't exist");

        let neighbors_u = self.adj[u];
        let neighbors_v = self.adj[v];
        let u_mask = bit(u);
        let v_mask = bit(v);

        let mut delta: isize = 0;

        // When removing (u,v):
        // - For each w in N(v) \ {u}: common[u][w] decreases by 1
        // - For each w in N(u) \ {v}: common[v][w] decreases by 1
        // Delta to c4_score_twice for decreasing common[x][y] from old to old-1 is: -(old-1)

        let mut t = neighbors_v & !u_mask;
        while t != 0 {
            let w = t.trailing_zeros() as usize;
            t &= t - 1;
            let old = self.common_neighbors[u][w] as isize;
            delta -= old - 1;
        }

        let mut t = neighbors_u & !v_mask;
        while t != 0 {
            let w = t.trailing_zeros() as usize;
            t &= t - 1;
            let old = self.common_neighbors[v][w] as isize;
            delta -= old - 1;
        }

        delta
    }

    /// Computes the C4 delta for flipping edge (u,v) WITHOUT modifying state.
    #[inline]
    pub fn c4_delta_if_flip(&self, u: usize, v: usize) -> isize {
        if self.has_edge(u, v) {
            self.c4_delta_if_remove(u, v)
        } else {
            self.c4_delta_if_add(u, v)
        }
    }

    /// Returns true if adding edge (u,v) would create zero new C4s.
    ///
    /// This checks that all affected common neighbor pairs have count 0,
    /// meaning no new choose2 contributions would be added.
    #[inline]
    pub fn is_c4_safe_to_add(&self, u: usize, v: usize) -> bool {
        debug_assert!(u < N && v < N && u != v);
        // The delta is 0 iff for all w in N(v): common[u][w] = 0
        // AND for all w in N(u): common[v][w] = 0
        self.c4_delta_if_add(u, v) == 0
    }

    /// Returns all non-edges that are C4-safe to add.
    ///
    /// This is useful for restricting moves to only feasibility-preserving ones.
    pub fn c4_safe_non_edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for u in 0..N {
            for v in (u + 1)..N {
                if !self.has_edge(u, v) && self.is_c4_safe_to_add(u, v) {
                    result.push((u, v));
                }
            }
        }
        result
    }

    /// Returns the count of C4-safe non-edges.
    #[inline]
    pub fn c4_safe_non_edge_count(&self) -> usize {
        let mut count = 0;
        for u in 0..N {
            for v in (u + 1)..N {
                if !self.has_edge(u, v) && self.is_c4_safe_to_add(u, v) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Flips an edge and incrementally updates the C4 score in `O(deg(u)+deg(v))`.
    ///
    /// # Panics
    /// Panics in debug builds if `u == v` or indices are out of range.
    #[inline]
    pub fn flip_edge(&mut self, u: usize, v: usize) {
        debug_assert!(u < N && v < N);
        debug_assert!(u != v);

        let u_mask = bit(u);
        let v_mask = bit(v);
        let adding = (self.adj[u] & v_mask) == 0;

        let neighbors_u = self.adj[u];
        let neighbors_v = self.adj[v];

        if adding {
            // Adding edge (u, v): v becomes a new common neighbor for (u, w) for all w in N(v)\{u},
            // and u becomes a new common neighbor for (v, w) for all w in N(u)\{v}.
            self.update_common_neighbors_add(u, neighbors_v & !u_mask);
            self.update_common_neighbors_add(v, neighbors_u & !v_mask);
            self.adj[u] |= v_mask;
            self.adj[v] |= u_mask;
        } else {
            // Removing edge (u, v): update adjacency first, then common neighbors.
            self.adj[u] &= !v_mask;
            self.adj[v] &= !u_mask;
            // neighbors_v was captured before removal, so it contains u; we must exclude u.
            self.update_common_neighbors_remove(u, neighbors_v & !u_mask);
            self.update_common_neighbors_remove(v, neighbors_u & !v_mask);
        }
    }

    /// Helper for adding an edge: updates common neighbors for (x, w) for all w in `neighbors`.
    #[inline(always)]
    fn update_common_neighbors_add(&mut self, x: usize, neighbors: u64) {
        let mut t = neighbors;
        while t != 0 {
            let w = t.trailing_zeros() as usize;
            t &= t - 1; // Clear lowest set bit (faster than `t &= !(1 << w)`)

            let old = self.common_neighbors[x][w] as usize;
            let new = old + 1;
            self.common_neighbors[x][w] = new as u8;
            self.common_neighbors[w][x] = new as u8;
            // Delta: choose2(new) - choose2(old) = old
            self.c4_score_twice += old;
        }
    }

    /// Helper for removing an edge: updates common neighbors for (x, w) for all w in `neighbors`.
    #[inline(always)]
    fn update_common_neighbors_remove(&mut self, x: usize, neighbors: u64) {
        let mut t = neighbors;
        while t != 0 {
            let w = t.trailing_zeros() as usize;
            t &= t - 1;

            let old = self.common_neighbors[x][w] as usize;
            debug_assert!(old > 0);
            let new = old - 1;
            self.common_neighbors[x][w] = new as u8;
            self.common_neighbors[w][x] = new as u8;
            // Delta: choose2(new) - choose2(old) = -(old - 1)
            self.c4_score_twice -= old - 1;
        }
    }

    /// Greedily constructs an independent set; returns `true` if it reaches size `k`.
    ///
    /// This is a **lower bound** on \(\alpha(G)\). If it reaches size `k`, the graph definitely
    /// violates the "no independent set of size k" constraint; if it does not, the graph may still
    /// violate it (so callers should fall back to an exact oracle when needed).
    pub fn greedy_find_independent_set_of_size(&self, k: usize, out: &mut Vec<usize>) -> bool {
        out.clear();
        if k == 0 {
            return true;
        }

        let mask = all_bits::<N>();
        let mut candidates = mask;

        while candidates != 0 {
            // Pick a vertex with minimal degree within the current candidate set.
            let mut best_v = candidates.trailing_zeros() as usize;
            let mut best_deg = u32::MAX;

            let mut t = candidates;
            while t != 0 {
                let v = t.trailing_zeros() as usize;
                t &= t - 1;
                let deg = (self.adj[v] & candidates).count_ones();
                if deg < best_deg {
                    best_deg = deg;
                    best_v = v;
                    if deg == 0 {
                        break; // Can't do better than 0
                    }
                }
            }

            out.push(best_v);
            if out.len() >= k {
                out.truncate(k);
                return true;
            }

            // Next candidates must be non-neighbors of best_v (and exclude best_v itself).
            candidates &= !self.adj[best_v];
            candidates &= !bit(best_v);
        }

        false
    }

    /// If the graph currently contains at least one 4-cycle, attempts to sample an edge
    /// that belongs to some 4-cycle (and thus is a good candidate to remove).
    ///
    /// This is a heuristic move generator: it returns `None` if it fails to find a witness
    /// within `max_tries` attempts.
    pub fn sample_c4_edge_to_remove<R: Rng>(
        &self,
        rng: &mut R,
        max_tries: usize,
    ) -> Option<(usize, usize)> {
        if self.c4_score_twice == 0 {
            return None;
        }

        for _ in 0..max_tries {
            let i = rng.random_range(0..N);
            let mut j = rng.random_range(0..N);
            while j == i {
                j = rng.random_range(0..N);
            }

            if self.common_neighbors[i][j] < 2 {
                continue;
            }

            let mut common = self.adj[i] & self.adj[j];
            if common.count_ones() < 2 {
                continue;
            }

            let a = pick_random_bit(rng, common);
            common &= !bit(a);
            let b = pick_random_bit(rng, common);

            // We found a C4: i - a - j - b - i. Pick one of its edges.
            let (u, v) = match rng.random_range(0..4u32) {
                0 => (i, a),
                1 => (a, j),
                2 => (j, b),
                _ => (b, i),
            };

            debug_assert!(self.has_edge(u, v));
            return Some((u, v));
        }

        None
    }

    /// Finds all edges that are part of at least one C4. Returns up to `limit` edges.
    pub fn find_c4_edges(&self, limit: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::with_capacity(limit);
        let mut seen = std::collections::HashSet::new();

        for i in 0..N {
            for j in (i + 1)..N {
                let cn = self.common_neighbors[i][j] as usize;
                if cn >= 2 {
                    // Vertices i and j have cn >= 2 common neighbors.
                    // Every edge from i to these common neighbors is in a C4.
                    // Every edge from j to these common neighbors is in a C4.
                    let mut common = self.adj[i] & self.adj[j];
                    while common != 0 {
                        let w = common.trailing_zeros() as usize;
                        common &= common - 1;

                        // Edges (i, w) and (j, w) are in C4s formed by i, j and other common neighbors.
                        for v in [i, j] {
                            let (u1, u2) = if v < w { (v, w) } else { (w, v) };
                            if seen.insert((u1, u2)) {
                                result.push((u1, u2));
                                if result.len() >= limit {
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Saves the adjacency matrix to a file as an `N x N` matrix of `0/1` characters.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or written.
    pub fn save_to_file(&self, filename: impl AsRef<Path>) -> io::Result<()> {
        let mut f = File::create(filename)?;
        self.write_to(&mut f)
    }

    /// Writes the adjacency matrix to a writer as an `N x N` matrix of `0/1` characters.
    ///
    /// # Errors
    /// Returns an error if writing fails.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        for i in 0..N {
            for j in 0..N {
                let edge = (self.adj[i] >> j) & 1;
                write!(w, "{edge}")?;
            }
            writeln!(w)?;
        }
        Ok(())
    }

    /// Loads a graph from a file containing an `N x N` adjacency matrix.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the matrix is malformed.
    pub fn load_from_file(filename: impl AsRef<Path>) -> Result<Self, GraphParseError> {
        let file = File::open(filename).map_err(|e| GraphParseError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut text = String::new();
        for line in reader.lines() {
            let line = line.map_err(|e| GraphParseError::Io(e.to_string()))?;
            text.push_str(&line);
            text.push('\n');
        }
        let parsed = parse_adjacency_matrix(&text)?;
        let adj = parsed.to_fixed::<N>()?;
        Ok(Self::from_adj(adj))
    }

    #[cfg(test)]
    fn recompute_for_test(&self) -> ([[u8; N]; N], usize) {
        compute_common_neighbors_and_c4_score_twice(&self.adj)
    }
}

/// Picks a random bit from a non-zero mask.
#[inline(always)]
fn pick_random_bit<R: Rng>(rng: &mut R, mask: u64) -> usize {
    debug_assert!(mask != 0);
    let count = mask.count_ones() as usize;
    let idx = rng.random_range(0..count);
    let mut t = mask;
    for _ in 0..idx {
        t &= t - 1;
    }
    t.trailing_zeros() as usize
}

/// Computes the common-neighbor matrix and the "twice-count" C4 score from scratch.
fn compute_common_neighbors_and_c4_score_twice<const N: usize>(
    adj: &[u64; N],
) -> ([[u8; N]; N], usize) {
    let mut common_neighbors = [[0u8; N]; N];
    let mut c4_score_twice = 0usize;

    for i in 0..N {
        for j in (i + 1)..N {
            let common = (adj[i] & adj[j]).count_ones() as u8;
            common_neighbors[i][j] = common;
            common_neighbors[j][i] = common;
            c4_score_twice += choose2(common as usize);
        }
    }
    (common_neighbors, c4_score_twice)
}

// ============================================================================
// Parsing
// ============================================================================

/// Parsed adjacency matrix (0/1) representation.
#[derive(Clone, Debug)]
pub struct ParsedAdjacencyMatrix {
    /// Number of vertices (rows/cols).
    pub n: usize,
    /// Row bitsets (length `n`).
    pub rows: Vec<u64>,
}

impl ParsedAdjacencyMatrix {
    /// Converts into a fixed-size adjacency array.
    ///
    /// # Errors
    /// Returns an error if the parsed matrix size doesn't match `N`.
    pub fn to_fixed<const N: usize>(&self) -> Result<[u64; N], GraphParseError> {
        if self.n != N {
            return Err(GraphParseError::MismatchedOrder {
                expected: N,
                got: self.n,
            });
        }
        let mut out = [0u64; N];
        for (i, &row) in self.rows.iter().enumerate() {
            out[i] = row;
        }
        Ok(out)
    }
}

/// Errors encountered while parsing/validating an adjacency matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum GraphParseError {
    /// No non-empty rows were found.
    Empty,
    /// Matrix is not square.
    NonSquare {
        /// The row index with wrong length.
        row: usize,
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// Encountered a non `0/1` character.
    InvalidChar {
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
        /// The invalid character.
        ch: char,
    },
    /// The matrix is larger than 64 vertices, which doesn't fit in a `u64` bitset.
    TooManyVertices {
        /// Number of vertices in the matrix.
        n: usize,
    },
    /// Diagonal contains a `1`.
    SelfLoop {
        /// The vertex with a self-loop.
        vertex: usize,
    },
    /// `A[i][j] != A[j][i]`.
    NotSymmetric {
        /// Row index.
        i: usize,
        /// Column index.
        j: usize,
        /// Value at A[i][j].
        a_ij: u8,
        /// Value at A[j][i].
        a_ji: u8,
    },
    /// Attempted to convert to a fixed-size adjacency array with a different order.
    MismatchedOrder {
        /// Expected order (N).
        expected: usize,
        /// Actual order from parsed matrix.
        got: usize,
    },
    /// I/O error (file not found, etc.).
    Io(String),
}

impl fmt::Display for GraphParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphParseError::Empty => write!(f, "adjacency matrix is empty"),
            GraphParseError::NonSquare { row, expected, got } => write!(
                f,
                "adjacency matrix is not square: row {row} has length {got}, expected {expected}"
            ),
            GraphParseError::InvalidChar { row, col, ch } => {
                write!(
                    f,
                    "invalid character at ({row}, {col}): {ch:?} (expected '0' or '1')"
                )
            }
            GraphParseError::TooManyVertices { n } => {
                write!(
                    f,
                    "matrix has {n} vertices; this implementation supports n <= 64"
                )
            }
            GraphParseError::SelfLoop { vertex } => {
                write!(f, "self-loop detected at vertex {vertex}")
            }
            GraphParseError::NotSymmetric { i, j, a_ij, a_ji } => write!(
                f,
                "matrix is not symmetric at ({i},{j}): A[i][j]={a_ij}, A[j][i]={a_ji}"
            ),
            GraphParseError::MismatchedOrder { expected, got } => write!(
                f,
                "matrix order mismatch: expected {expected} vertices, got {got}"
            ),
            GraphParseError::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for GraphParseError {}

/// Parses a `0/1` adjacency matrix from text.
///
/// Rules:
/// - Blank lines are ignored.
/// - The matrix must be square, symmetric, and have a zero diagonal.
/// - `n` must be `<= 64`.
///
/// # Errors
/// Returns an error if the input is empty, non-square, contains invalid characters,
/// has self-loops, or is not symmetric.
pub fn parse_adjacency_matrix(text: &str) -> Result<ParsedAdjacencyMatrix, GraphParseError> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return Err(GraphParseError::Empty);
    }
    let n = lines.len();
    if n > 64 {
        return Err(GraphParseError::TooManyVertices { n });
    }

    let mut rows = Vec::with_capacity(n);
    for (i, line) in lines.iter().enumerate() {
        let bytes = line.as_bytes();
        if bytes.len() != n {
            return Err(GraphParseError::NonSquare {
                row: i,
                expected: n,
                got: bytes.len(),
            });
        }
        let mut mask = 0u64;
        for (j, &b) in bytes.iter().enumerate() {
            match b {
                b'0' => {}
                b'1' => mask |= bit(j),
                _ => {
                    return Err(GraphParseError::InvalidChar {
                        row: i,
                        col: j,
                        ch: b as char,
                    })
                }
            }
        }
        rows.push(mask);
    }

    // Validate diagonal and symmetry.
    for i in 0..n {
        if ((rows[i] >> i) & 1) != 0 {
            return Err(GraphParseError::SelfLoop { vertex: i });
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let a_ij = ((rows[i] >> j) & 1) as u8;
            let a_ji = ((rows[j] >> i) & 1) as u8;
            if a_ij != a_ji {
                return Err(GraphParseError::NotSymmetric { i, j, a_ij, a_ji });
            }
        }
    }

    Ok(ParsedAdjacencyMatrix { n, rows })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    // -------------------------------------------------------------------------
    // Incremental invariant tests
    // -------------------------------------------------------------------------

    #[test]
    fn flip_edge_matches_recompute_invariants() {
        const N: usize = 16;
        let mut rng = XorShiftRng::seed_from_u64(0xC0FFEE);
        let mut state = RamseyState::<N>::new_random(&mut rng, 0.35);

        for _ in 0..5_000 {
            let u = rng.random_range(0..N);
            let mut v = rng.random_range(0..N);
            while v == u {
                v = rng.random_range(0..N);
            }
            state.flip_edge(u, v);
            let (cn, c4) = state.recompute_for_test();
            assert_eq!(state.c4_score_twice, c4, "C4 score mismatch after flip");
            assert_eq!(state.common_neighbors, cn, "common_neighbors mismatch");
        }
    }

    #[test]
    fn flip_edge_is_reversible() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xBEEF);
        let state_orig = RamseyState::<N>::new_random(&mut rng, 0.4);

        for _ in 0..1_000 {
            let mut state = state_orig.clone();
            let u = rng.random_range(0..N);
            let mut v = rng.random_range(0..N);
            while v == u {
                v = rng.random_range(0..N);
            }

            let before_adj = state.adj;
            let before_c4 = state.c4_score_twice;
            let before_cn = state.common_neighbors;

            state.flip_edge(u, v);
            state.flip_edge(u, v); // flip back

            assert_eq!(state.adj, before_adj, "adj not restored");
            assert_eq!(state.c4_score_twice, before_c4, "C4 not restored");
            assert_eq!(state.common_neighbors, before_cn, "common_neighbors not restored");
        }
    }

    // -------------------------------------------------------------------------
    // Edge-case graph tests
    // -------------------------------------------------------------------------

    #[test]
    fn empty_graph_has_no_c4() {
        let state = RamseyState::<10>::empty();
        assert_eq!(state.c4_score_twice(), 0);
        assert_eq!(state.edge_count(), 0);
    }

    #[test]
    fn complete_graph_properties() {
        let state = RamseyState::<5>::complete();
        // K5 has 5*4/2 = 10 edges
        assert_eq!(state.edge_count(), 10);
        // K5 has C(5,4) * 3 = 15 copies of C4
        assert_eq!(state.c4_count(), 15);
    }

    #[test]
    fn single_c4_cycle() {
        // 4-cycle: 0-1-2-3-0
        let adj = [0b1010u64, 0b0101u64, 0b1010u64, 0b0101u64];
        let state = RamseyState::<4>::from_adj(adj);
        assert_eq!(state.c4_score_twice(), 2);
        assert_eq!(state.c4_count(), 1);
    }

    #[test]
    fn path_graph_has_no_c4() {
        // Path: 0-1-2-3-4
        let adj: [u64; 5] = [
            0b00010, // 0 -- 1
            0b00101, // 1 -- 0, 2
            0b01010, // 2 -- 1, 3
            0b10100, // 3 -- 2, 4
            0b01000, // 4 -- 3
        ];
        let state = RamseyState::<5>::from_adj(adj);
        assert_eq!(state.c4_count(), 0);
        assert_eq!(state.edge_count(), 4);
    }

    #[test]
    fn star_graph_has_no_c4() {
        // Star with center 0 and leaves 1,2,3,4
        let adj: [u64; 5] = [
            0b11110, // 0 -- 1,2,3,4
            0b00001, // 1 -- 0
            0b00001, // 2 -- 0
            0b00001, // 3 -- 0
            0b00001, // 4 -- 0
        ];
        let state = RamseyState::<5>::from_adj(adj);
        assert_eq!(state.c4_count(), 0);
        assert_eq!(state.edge_count(), 4);
    }

    // -------------------------------------------------------------------------
    // Greedy independent set tests
    // -------------------------------------------------------------------------

    #[test]
    fn greedy_is_returns_valid_independent_set() {
        const N: usize = 16;
        let mut rng = XorShiftRng::seed_from_u64(0xFACE);

        for _ in 0..100 {
            let state = RamseyState::<N>::new_random(&mut rng, 0.35);
            let mut out = Vec::new();

            for k in 1..=N {
                if state.greedy_find_independent_set_of_size(k, &mut out) {
                    // Verify the output is actually independent
                    assert_eq!(out.len(), k);
                    for i in 0..out.len() {
                        for j in (i + 1)..out.len() {
                            assert!(
                                !state.has_edge(out[i], out[j]),
                                "greedy returned non-independent set: edge ({}, {})",
                                out[i],
                                out[j]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn greedy_is_on_empty_graph() {
        let state = RamseyState::<8>::empty();
        let mut out = Vec::new();

        // Empty graph has alpha = N
        assert!(state.greedy_find_independent_set_of_size(8, &mut out));
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn greedy_is_on_complete_graph() {
        let state = RamseyState::<8>::complete();
        let mut out = Vec::new();

        // Complete graph has alpha = 1
        assert!(state.greedy_find_independent_set_of_size(1, &mut out));
        assert!(!state.greedy_find_independent_set_of_size(2, &mut out));
    }

    // -------------------------------------------------------------------------
    // C4 edge sampling tests
    // -------------------------------------------------------------------------

    #[test]
    fn c4_edge_sampling_returns_edge_in_c4() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xCAFE);

        for _ in 0..50 {
            let state = RamseyState::<N>::new_random(&mut rng, 0.4);

            if state.c4_score_twice() == 0 {
                // No C4, sampling should return None
                assert!(state.sample_c4_edge_to_remove(&mut rng, 100).is_none());
                continue;
            }

            // Has C4, sampling should return an edge that's in a C4
            if let Some((u, v)) = state.sample_c4_edge_to_remove(&mut rng, 100) {
                assert!(state.has_edge(u, v), "returned non-edge");
                assert!(u != v);
                assert!(u < N && v < N);
            }
        }
    }

    #[test]
    fn find_c4_edges_on_c4_free_graph() {
        // Path graph is C4-free
        let adj: [u64; 5] = [
            0b00010, 0b00101, 0b01010, 0b10100, 0b01000,
        ];
        let state = RamseyState::<5>::from_adj(adj);
        assert!(state.find_c4_edges(100).is_empty());
    }

    // -------------------------------------------------------------------------
    // Invariant validation tests
    // -------------------------------------------------------------------------

    // These tests only run in debug mode because release mode uses panic = "abort"
    // which prevents the test harness from catching panics.
    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn from_adj_panics_on_self_loop() {
        let mut adj = [0u64; 4];
        adj[0] = 0b0001; // Edge 0-0
        let _ = RamseyState::<4>::from_adj(adj);
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn from_adj_panics_on_asymmetry() {
        let mut adj = [0u64; 4];
        adj[0] = 0b0010; // 0-1
        // adj[1] should be 0b0001 but is 0
        let _ = RamseyState::<4>::from_adj(adj);
    }

    // -------------------------------------------------------------------------
    // Round-trip save/load tests
    // -------------------------------------------------------------------------

    #[test]
    fn write_and_parse_roundtrip() {
        const N: usize = 10;
        let mut rng = XorShiftRng::seed_from_u64(0x1234);
        let state = RamseyState::<N>::new_random(&mut rng, 0.3);

        let mut buf = Vec::new();
        state.write_to(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();

        let parsed = parse_adjacency_matrix(&text).unwrap();
        let adj2 = parsed.to_fixed::<N>().unwrap();
        let state2 = RamseyState::<N>::from_adj(adj2);

        assert_eq!(state.adj, state2.adj);
        assert_eq!(state.c4_score_twice, state2.c4_score_twice);
        assert_eq!(state.common_neighbors, state2.common_neighbors);
    }

    // -------------------------------------------------------------------------
    // Parser error tests
    // -------------------------------------------------------------------------

    #[test]
    fn parse_adjacency_matrix_rejects_non_square() {
        let err = parse_adjacency_matrix("010\n10\n").unwrap_err();
        assert!(matches!(err, GraphParseError::NonSquare { .. }));
    }

    #[test]
    fn parse_adjacency_matrix_rejects_invalid_char() {
        let err = parse_adjacency_matrix("0a\n00\n").unwrap_err();
        assert!(matches!(err, GraphParseError::InvalidChar { .. }));
    }

    #[test]
    fn parse_adjacency_matrix_rejects_self_loop() {
        let err = parse_adjacency_matrix("10\n01\n").unwrap_err();
        assert_eq!(err, GraphParseError::SelfLoop { vertex: 0 });
    }

    #[test]
    fn parse_adjacency_matrix_rejects_non_symmetric() {
        let err = parse_adjacency_matrix("01\n00\n").unwrap_err();
        assert!(matches!(err, GraphParseError::NotSymmetric { .. }));
    }

    #[test]
    fn parse_adjacency_matrix_rejects_empty() {
        let err = parse_adjacency_matrix("").unwrap_err();
        assert_eq!(err, GraphParseError::Empty);
    }

    #[test]
    fn parse_adjacency_matrix_rejects_whitespace_only() {
        let err = parse_adjacency_matrix("   \n\n  \n").unwrap_err();
        assert_eq!(err, GraphParseError::Empty);
    }

    // -------------------------------------------------------------------------
    // Lookup table tests
    // -------------------------------------------------------------------------

    #[test]
    fn choose2_lookup_is_correct() {
        for n in 0usize..65 {
            let expected = (n * n.saturating_sub(1)) / 2;
            assert_eq!(choose2(n), expected, "choose2({n}) mismatch");
        }
    }

    // -------------------------------------------------------------------------
    // Degree and edge count tests
    // -------------------------------------------------------------------------

    #[test]
    fn degree_is_correct() {
        // Star graph with center 0
        let adj: [u64; 4] = [
            0b1110, // 0 -- 1,2,3
            0b0001, // 1 -- 0
            0b0001, // 2 -- 0
            0b0001, // 3 -- 0
        ];
        let state = RamseyState::<4>::from_adj(adj);
        assert_eq!(state.degree(0), 3);
        assert_eq!(state.degree(1), 1);
        assert_eq!(state.degree(2), 1);
        assert_eq!(state.degree(3), 1);
    }

    #[test]
    fn handshaking_lemma_holds() {
        const N: usize = 32;
        let mut rng = XorShiftRng::seed_from_u64(42);
        for _ in 0..10 {
            let state = RamseyState::<N>::new_random(&mut rng, 0.25);
            let mut sum_deg = 0;
            for i in 0..N {
                sum_deg += state.degree(i);
            }
            assert_eq!(sum_deg as usize, 2 * state.edge_count());
        }
    }

    #[test]
    fn common_neighbors_consistency() {
        const N: usize = 20;
        let mut rng = XorShiftRng::seed_from_u64(123);
        let mut state = RamseyState::<N>::new_random(&mut rng, 0.4);

        for _ in 0..1000 {
            let u = rng.random_range(0..N);
            let mut v = rng.random_range(0..N);
            while v == u {
                v = rng.random_range(0..N);
            }
            state.flip_edge(u, v);

            // Cross-check all common neighbor counts
            for i in 0..N {
                for j in (i + 1)..N {
                    let expected = (state.adj[i] & state.adj[j]).count_ones() as u8;
                    assert_eq!(
                        state.common_neighbor_count(i, j),
                        expected,
                        "Mismatch for ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn find_c4_edges_is_accurate() {
        const N: usize = 10;
        // Create a known C4: 0-1-2-3-0
        let mut adj = [0u64; N];
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        for (u, v) in edges {
            adj[u] |= 1 << v;
            adj[v] |= 1 << u;
        }
        let state = RamseyState::<N>::from_adj(adj);
        let c4_edges = state.find_c4_edges(100);

        // Every edge in that cycle should be part of at least one C4
        for (u, v) in edges {
            assert!(
                c4_edges.contains(&(u.min(v), u.max(v))),
                "Edge ({u}, {v}) missing from C4 list"
            );
        }
    }

    #[test]
    fn all_bits_mask_correctness() {
        assert_eq!(all_bits::<0>(), 0);
        assert_eq!(all_bits::<1>(), 1);
        assert_eq!(all_bits::<32>(), 0xFFFF_FFFF);
        assert_eq!(all_bits::<64>(), u64::MAX);
    }

    // -------------------------------------------------------------------------
    // Delta computation tests
    // -------------------------------------------------------------------------

    #[test]
    fn c4_delta_if_add_matches_actual() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xDE17A);

        for _ in 0..500 {
            let mut state = RamseyState::<N>::new_random(&mut rng, 0.3);

            // Find a non-edge
            let mut u = rng.random_range(0..N);
            let mut v = rng.random_range(0..N);
            while v == u || state.has_edge(u, v) {
                u = rng.random_range(0..N);
                v = rng.random_range(0..N);
            }

            let c4_before = state.c4_score_twice();
            let predicted_delta = state.c4_delta_if_add(u, v);

            state.flip_edge(u, v); // Add the edge
            let c4_after = state.c4_score_twice();
            let actual_delta = c4_after as isize - c4_before as isize;

            assert_eq!(
                predicted_delta, actual_delta,
                "Delta mismatch for adding ({u}, {v}): predicted {predicted_delta}, actual {actual_delta}"
            );
        }
    }

    #[test]
    fn c4_delta_if_remove_matches_actual() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xDE17A2);

        for _ in 0..500 {
            let mut state = RamseyState::<N>::new_random(&mut rng, 0.4);

            // Find an existing edge
            let mut u = rng.random_range(0..N);
            let mut v = rng.random_range(0..N);
            while v == u || !state.has_edge(u, v) {
                u = rng.random_range(0..N);
                v = rng.random_range(0..N);
            }

            let c4_before = state.c4_score_twice();
            let predicted_delta = state.c4_delta_if_remove(u, v);

            state.flip_edge(u, v); // Remove the edge
            let c4_after = state.c4_score_twice();
            let actual_delta = c4_after as isize - c4_before as isize;

            assert_eq!(
                predicted_delta, actual_delta,
                "Delta mismatch for removing ({u}, {v}): predicted {predicted_delta}, actual {actual_delta}"
            );
        }
    }

    #[test]
    fn c4_delta_if_flip_matches_actual() {
        const N: usize = 15;
        let mut rng = XorShiftRng::seed_from_u64(0xF11B);

        for _ in 0..1000 {
            let mut state = RamseyState::<N>::new_random(&mut rng, 0.35);

            let u = rng.random_range(0..N);
            let mut v = rng.random_range(0..N);
            while v == u {
                v = rng.random_range(0..N);
            }

            let c4_before = state.c4_score_twice();
            let predicted_delta = state.c4_delta_if_flip(u, v);

            state.flip_edge(u, v);
            let c4_after = state.c4_score_twice();
            let actual_delta = c4_after as isize - c4_before as isize;

            assert_eq!(
                predicted_delta, actual_delta,
                "Delta mismatch for flipping ({u}, {v})"
            );
        }
    }

    #[test]
    fn is_c4_safe_to_add_correct() {
        const N: usize = 8;
        let mut rng = XorShiftRng::seed_from_u64(0x5AFE);

        for _ in 0..100 {
            let state = RamseyState::<N>::new_random(&mut rng, 0.3);

            for u in 0..N {
                for v in (u + 1)..N {
                    if state.has_edge(u, v) {
                        continue;
                    }

                    let is_safe = state.is_c4_safe_to_add(u, v);
                    let delta = state.c4_delta_if_add(u, v);

                    if is_safe {
                        assert_eq!(
                            delta, 0,
                            "is_c4_safe_to_add returned true but delta is {delta}"
                        );
                    }
                    if delta == 0 {
                        assert!(
                            is_safe,
                            "delta is 0 but is_c4_safe_to_add returned false"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn c4_safe_non_edges_all_valid() {
        const N: usize = 10;
        let mut rng = XorShiftRng::seed_from_u64(0x5AFE2);

        for _ in 0..50 {
            let state = RamseyState::<N>::new_random(&mut rng, 0.25);
            let safe_edges = state.c4_safe_non_edges();

            for (u, v) in safe_edges {
                assert!(!state.has_edge(u, v), "Safe edge ({u}, {v}) already exists");
                assert!(
                    state.is_c4_safe_to_add(u, v),
                    "Edge ({u}, {v}) is not actually safe"
                );
                assert_eq!(
                    state.c4_delta_if_add(u, v),
                    0,
                    "Edge ({u}, {v}) would create C4s"
                );
            }
        }
    }

    #[test]
    fn c4_safe_count_matches_list() {
        const N: usize = 12;
        let mut rng = XorShiftRng::seed_from_u64(0xC0047);

        for _ in 0..20 {
            let state = RamseyState::<N>::new_random(&mut rng, 0.3);
            let count = state.c4_safe_non_edge_count();
            let list = state.c4_safe_non_edges();
            assert_eq!(count, list.len());
        }
    }
}

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_maybe_rayon::*;

use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::iop::target::Target;
use crate::iop::wire::Wire;

/// When `true`, `get_sigma_polys` parallelizes across whole sigma columns with a single
/// `par_chunks` traversal instead of entering and joining a fresh inner Rayon traversal
/// sequentially for every column. `par_chunks` is an indexed parallel iterator, so the collected
/// column order and the sequential element order within every column are unchanged; only task
/// placement moves. Flip to `false` to restore the previous sequential-outer schedule exactly.
const OUTER_PARALLEL_SIGMA_COLUMNS: bool = false;

/// Skip the numerator factor at a routed position. The matching denominator factor belongs to
/// its unique sigma predecessor in the same base-domain row and quotient-degree chunk.
pub(crate) const PERMUTATION_SKIP_NUMERATOR: u8 = 1;
/// Skip the denominator factor at a routed position. Its sigma successor supplies the identical
/// numerator factor in the same base-domain row and quotient-degree chunk.
pub(crate) const PERMUTATION_SKIP_DENOMINATOR: u8 = 2;
pub(crate) const PERMUTATION_SKIP_BOTH: u8 =
    PERMUTATION_SKIP_NUMERATOR | PERMUTATION_SKIP_DENOMINATOR;

/// The skip mask is stored as paired numerator/denominator bitplanes per base-row quotient
/// chunk. Within each side plane, the least-significant bit is the chunk's first routed column.
/// Zero therefore always means "skip nothing", including for malformed or truncated runtime data.
fn permutation_factor_skip_layout(
    num_routed_wires: usize,
    quotient_degree_factor: usize,
) -> Option<(usize, usize, usize)> {
    if num_routed_wires == 0 || quotient_degree_factor == 0 {
        return None;
    }
    let num_chunks = num_routed_wires.div_ceil(quotient_degree_factor);
    let bytes_per_side = quotient_degree_factor.div_ceil(8);
    let chunk_stride = bytes_per_side.checked_mul(2)?;
    let row_stride = num_chunks.checked_mul(chunk_stride)?;
    Some((num_chunks, bytes_per_side, row_stride))
}

/// Exact byte length of the paired permutation-factor skip bitplanes.
///
/// This is public so compact, trusted circuit embeddings can validate and load a plan generated
/// at build time without re-walking the entire sigma permutation during prover startup.
pub fn permutation_factor_skip_mask_len(
    degree: usize,
    num_routed_wires: usize,
    quotient_degree_factor: usize,
) -> Option<usize> {
    let (_, _, row_stride) =
        permutation_factor_skip_layout(num_routed_wires, quotient_degree_factor)?;
    degree.checked_mul(row_stride)
}

/// Read the two skip bits for one row-major routed position from the chunk bitplanes.
/// Out-of-range indices conservatively return zero (skip nothing).
#[inline(always)]
pub(crate) fn permutation_factor_skips(
    mask: &[u8],
    routed_index: usize,
    num_routed_wires: usize,
    quotient_degree_factor: usize,
) -> u8 {
    let Some((num_chunks, bytes_per_side, row_stride)) =
        permutation_factor_skip_layout(num_routed_wires, quotient_degree_factor)
    else {
        return 0;
    };
    let row = routed_index / num_routed_wires;
    let column = routed_index % num_routed_wires;
    let chunk = column / quotient_degree_factor;
    let lane = column % quotient_degree_factor;
    let chunk_base = row
        .checked_mul(row_stride)
        .and_then(|base| base.checked_add(chunk * bytes_per_side * 2));
    let Some(chunk_base) = chunk_base else {
        return 0;
    };
    debug_assert!(chunk < num_chunks);
    let byte_in_side = lane >> 3;
    let bit = 1 << (lane & 7);
    let numerator = mask
        .get(chunk_base + byte_in_side)
        .is_some_and(|byte| byte & bit != 0) as u8;
    let denominator = mask
        .get(chunk_base + bytes_per_side + byte_in_side)
        .is_some_and(|byte| byte & bit != 0) as u8;
    numerator * PERMUTATION_SKIP_NUMERATOR + denominator * PERMUTATION_SKIP_DENOMINATOR
}

#[inline]
fn set_permutation_factor_skip(mask: &mut [u8], byte: usize, bit: u8) -> bool {
    if mask[byte] & bit != 0 {
        return false;
    }
    mask[byte] |= bit;
    true
}

/// Disjoint Set Forest data-structure following <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
#[derive(Debug)]
pub struct Forest {
    /// A map of parent pointers, stored as indices.
    ///
    /// Entries are `u32` rather than `usize`: a forest index is bounded by
    /// `num_wires * degree + num_virtual_targets`, which [`Forest::new`] asserts fits in a `u32`.
    /// Halving the entry width halves the resident size of this map and the memory traffic of
    /// every read of it — most importantly `PartitionWitness`'s per-write lookup and its
    /// `full_witness` drain, which are DRAM-bandwidth bound on large circuits. Every value is
    /// zero-extended at the indexing site, so all derived quantities are unchanged.
    pub(crate) parents: Vec<u32>,

    num_wires: usize,
    num_routed_wires: usize,
    degree: usize,
}

impl Forest {
    pub fn new(
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
    ) -> Self {
        let capacity = num_wires * degree + num_virtual_targets;
        assert!(
            capacity <= u32::MAX as usize,
            "forest of {capacity} targets exceeds the u32 index range of the representative map"
        );
        Self {
            parents: Vec::with_capacity(capacity),
            num_wires,
            num_routed_wires,
            degree,
        }
    }

    /// Reconstructs a forest from a stored, fully compressed representative
    /// map (`ProverOnlyCircuitData::representative_map`). Intended for loaders
    /// that re-derive the sigma polynomials without re-running circuit
    /// construction.
    ///
    /// A malformed map must be rejected before [`Self::wire_partition`], which
    /// deliberately has no bounds checks in its production-sized scan. Every
    /// stored parent must therefore be an in-range root, and the map must cover
    /// all physical wires. Valid maps satisfy these conditions because the
    /// builder calls `compress_paths` over the entire forest before storing it.
    pub fn from_compressed_parents(
        parents: Vec<u32>,
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
    ) -> Option<Self> {
        if num_routed_wires > num_wires
            || num_wires.checked_mul(degree)? > parents.len()
            || parents.iter().any(|&parent| {
                let parent = parent as usize;
                parent >= parents.len() || parents[parent] as usize != parent
            })
        {
            return None;
        }
        Some(Self {
            parents,
            num_wires,
            num_routed_wires,
            degree,
        })
    }

    /// Reconstructs a forest from a trusted, fully compressed representative map without a
    /// validation scan.
    ///
    /// This is intended only for compile-time-generated circuit blobs which are embedded in the
    /// prover executable. Unlike [`Self::from_compressed_parents`], malformed input can panic in
    /// [`Self::wire_partition`]; untrusted deserializers must use the validated constructor.
    pub fn from_trusted_compressed_parents(
        parents: Vec<u32>,
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
    ) -> Self {
        Self {
            parents,
            num_wires,
            num_routed_wires,
            degree,
        }
    }

    /// Consumes the forest and returns its parent map, undoing
    /// [`Self::from_compressed_parents`] without copying.
    pub fn into_parents(self) -> Vec<u32> {
        self.parents
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    /// Add a new partition with a single member.
    ///
    /// The `u32` narrowing is guarded by [`Forest::new`], which asserts that the announced
    /// capacity `num_wires * degree + num_virtual_targets` fits in a `u32`; `sigma_vecs` calls
    /// this exactly that many times. The per-call check is therefore a `debug_assert!`, keeping
    /// the tens-of-millions-of-iterations insertion loop free of a redundant branch.
    pub fn add(&mut self, t: Target) {
        let index = self.parents.len();
        debug_assert_eq!(self.target_index(t), index);
        debug_assert!(
            index <= u32::MAX as usize,
            "forest index {index} exceeds the u32 index range of the representative map"
        );
        self.parents.push(index as u32);
    }

    /// Path compression method, see <https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives>.
    pub fn find(&mut self, x_index: usize) -> usize {
        // Note: We avoid recursion here since the chains can be long, causing stack overflows.
        let mut x_index = x_index as u32;

        // First, find the representative of the set containing `x_index`.
        let mut representative = x_index;
        while self.parents[representative as usize] != representative {
            representative = self.parents[representative as usize];
        }

        // Then, update each node in this chain to point directly to the representative.
        while self.parents[x_index as usize] != x_index {
            let old_parent = self.parents[x_index as usize];
            self.parents[x_index as usize] = representative;
            x_index = old_parent;
        }

        representative as usize
    }

    /// Merge two sets.
    pub fn merge(&mut self, tx: Target, ty: Target) {
        let x_index = self.find(self.target_index(tx));
        let y_index = self.find(self.target_index(ty));

        if x_index == y_index {
            return;
        }

        self.parents[y_index] = x_index as u32;
    }

    /// Compress all paths. After calling this, every `parent` value will point to the node's
    /// representative.
    ///
    /// The final `parents` vector is identical to calling `find(i)` for every `i`: a node is only
    /// ever written when it is a non-root (the `continue` guard), and the value written is always
    /// a root, so roots are stable for the whole pass and every index ends at `root(i)`.
    ///
    /// The writeback loop is load-bearing for performance, not just for `i`. A copy-constraint
    /// class built by repeated `connect` is a *chain*, and the outer loop visits it in the
    /// direction that walks it from the far end: writing the root into `parents[i]` alone leaves
    /// every intermediate node still pointing along the chain, so the next index re-walks almost
    /// the whole thing — quadratic in the class length, over a `parents` array of tens of
    /// millions of entries. Writing the root into every node on the path as we go makes each
    /// later node terminate in one hop.
    pub(crate) fn compress_paths(&mut self) {
        for i in 0..self.parents.len() {
            let parent = self.parents[i];
            if parent as usize == i {
                continue;
            }
            let mut root = parent;
            while self.parents[root as usize] != root {
                root = self.parents[root as usize];
            }
            // Point every node on `i`'s path directly at the root, not just `i`.
            let mut x = i;
            while self.parents[x] != root {
                let next = self.parents[x] as usize;
                self.parents[x] = root;
                x = next;
            }
        }
    }

    /// Assumes `compress_paths` has already been called.
    ///
    /// Each copy class is maintained as a closed cycle at every insertion: the first element of
    /// a class starts as a self-loop, and every later element is spliced in between the current
    /// tail and the head. A scan sequence `a, b, c` therefore evolves as `a->a`, then
    /// `a->b->a`, then `a->b->c->a`, which is exactly the open successor chain built by the
    /// previous implementation plus its closing sweep. This deletes the whole-forest `first`
    /// array and the final serial sweep over every forest entry.
    pub fn wire_partition(&mut self) -> WirePartition {
        let mut sigma = vec![0u32; self.degree * self.num_routed_wires];
        let mut last = vec![u32::MAX; self.parents.len()];

        for row in 0..self.degree {
            for column in 0..self.num_routed_wires {
                let t = Target::Wire(Wire { row, column });
                let parent = self.parents[self.target_index(t)] as usize;
                let index = (column * self.degree + row) as u32;
                let old_tail = last[parent];
                if old_tail == u32::MAX {
                    sigma[index as usize] = index;
                } else {
                    sigma[index as usize] = sigma[old_tail as usize];
                    sigma[old_tail as usize] = index;
                }
                last[parent] = index;
            }
        }

        WirePartition { sigma }
    }

    /// Fallible [`Self::wire_partition`] for compact embedded data.
    ///
    /// Parent bounds are checked inside the mandatory routed-wire scan, so a corrupt embedded
    /// representative map returns `None` without paying a separate whole-map validation pass.
    /// The commitment-cap check performed by the embedded loader then authenticates the derived
    /// sigma polynomials. Generic deserialization still validates global compression first.
    pub fn try_wire_partition(&mut self) -> Option<WirePartition> {
        if self.num_routed_wires > self.num_wires
            || self.num_wires.checked_mul(self.degree)? > self.parents.len()
        {
            return None;
        }
        let routed_positions = self.degree.checked_mul(self.num_routed_wires)?;
        if routed_positions > u32::MAX as usize {
            return None;
        }
        let mut sigma = vec![0u32; routed_positions];
        let mut last = vec![u32::MAX; self.parents.len()];

        for row in 0..self.degree {
            let target_base = row * self.num_wires;
            for column in 0..self.num_routed_wires {
                let parent = self.parents[target_base + column] as usize;
                if parent >= last.len() {
                    return None;
                }
                let index = (column * self.degree + row) as u32;
                let old_tail = last[parent];
                if old_tail == u32::MAX {
                    sigma[index as usize] = index;
                } else {
                    sigma[index as usize] = sigma[old_tail as usize];
                    sigma[old_tail as usize] = index;
                }
                last[parent] = index;
            }
        }

        Some(WirePartition { sigma })
    }
}

#[derive(Debug)]
pub struct WirePartition {
    sigma: Vec<u32>,
}

impl WirePartition {
    /// Derive paired numerator/denominator skip bitplanes per base-row quotient chunk.
    ///
    /// For a sigma edge `p -> q`, the denominator factor at `p` is
    /// `w(p) + beta * id(q) + gamma`, while the numerator factor at `q` is
    /// `w(q) + beta * id(q) + gamma`. Copy constraints make `w(p) == w(q)`. If both endpoints
    /// lie in the same base-domain row and quotient-degree chunk, those identical factors occur
    /// in the same committed partial-product relation and cancel symbolically. We mark the
    /// denominator at `p` and numerator at `q` independently; an internal run can therefore make
    /// an endpoint one-sided and an interior position two-sided. Singleton self-edges naturally
    /// receive both bits, subsuming the old fixed-position mask.
    ///
    /// `WirePartition` is only constructible by [`Forest::wire_partition`], which creates a
    /// routed-wire permutation from a validated forest. The mask builder relies on that invariant
    /// instead of allocating and filling a second production-sized bijection bitmap.
    pub fn permutation_factor_skip_mask(
        &self,
        degree: usize,
        num_routed_wires: usize,
        quotient_degree_factor: usize,
    ) -> Option<Vec<u8>> {
        if !degree.is_power_of_two()
            || num_routed_wires == 0
            || quotient_degree_factor == 0
        {
            return None;
        }
        let routed_positions = degree.checked_mul(num_routed_wires)?;
        if self.sigma.len() != routed_positions {
            return None;
        }

        let (_, bytes_per_side, row_stride) =
            permutation_factor_skip_layout(num_routed_wires, quotient_degree_factor)?;
        let chunk_stride = bytes_per_side * 2;
        let mut skips = vec![
            0u8;
            permutation_factor_skip_mask_len(
                degree,
                num_routed_wires,
                quotient_degree_factor,
            )?
        ];
        let degree_bits = degree.trailing_zeros() as usize;
        let degree_mask = degree - 1;
        // Division by the quotient width depends only on the routed column. Precompute the tiny
        // (80-entry in production) address table once rather than repeating it for every row.
        let skip_columns = (0..num_routed_wires)
            .map(|column| {
                let chunk = column / quotient_degree_factor;
                let lane = column % quotient_degree_factor;
                let byte_in_side = lane >> 3;
                let chunk_base = chunk * chunk_stride;
                (
                    chunk,
                    chunk_base + byte_in_side,
                    chunk_base + bytes_per_side + byte_in_side,
                    1u8 << (lane & 7),
                )
            })
            .collect::<Vec<_>>();
        for source_column in 0..num_routed_wires {
            let (source_chunk, _, source_denominator_byte, source_lane_bit) =
                skip_columns[source_column];
            for source_row in 0..degree {
                let source_column_major = source_column * degree + source_row;
                let target_column_major = self.sigma[source_column_major] as usize;
                debug_assert!(target_column_major < routed_positions);
                let target_column = target_column_major >> degree_bits;
                let target_row = target_column_major & degree_mask;
                let (target_chunk, target_numerator_byte, _, target_lane_bit) =
                    skip_columns[target_column];
                if source_row == target_row && source_chunk == target_chunk {
                    let row_base = source_row * row_stride;
                    let source_skip_byte = row_base + source_denominator_byte;
                    let target_skip_byte = row_base + target_numerator_byte;
                    if !set_permutation_factor_skip(
                        &mut skips,
                        source_skip_byte,
                        source_lane_bit,
                    ) || !set_permutation_factor_skip(
                        &mut skips,
                        target_skip_byte,
                        target_lane_bit,
                    ) {
                        return None;
                    }
                }
            }
        }
        Some(skips)
    }

    pub fn get_sigma_polys<F: Field>(
        &self,
        degree_log: usize,
        k_is: &[F],
        subgroup: &[F],
    ) -> Vec<PolynomialValues<F>> {
        let degree = 1 << degree_log;
        // `degree` is always a power of two here, so `x / degree == x >> degree_log` and
        // `x % degree == x & (degree - 1)` hold exactly.
        let mask = degree - 1;

        if OUTER_PARALLEL_SIGMA_COLUMNS {
            self.sigma
                .par_chunks(degree)
                .map(|chunk| {
                    let values = chunk
                        .iter()
                        .map(|&x| k_is[x as usize >> degree_log] * subgroup[x as usize & mask])
                        .collect::<Vec<_>>();
                    PolynomialValues::new(values)
                })
                .collect()
        } else {
            self.sigma
                .chunks(degree)
                .map(|chunk| {
                    let values = chunk
                        .par_iter()
                        .map(|&x| k_is[x as usize >> degree_log] * subgroup[x as usize & mask])
                        .collect::<Vec<_>>();
                    PolynomialValues::new(values)
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    /// Deterministic pseudo-random stream (no `rand` dependency) for building forests.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Populates a forest exactly the way `CircuitBuilder::sigma_vecs` does: all wire targets
    /// in row-major index order, then all virtual targets, then the copy-constraint merges.
    fn build_forest(
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
        merges: &[(Target, Target)],
    ) -> Forest {
        let mut forest = Forest::new(num_wires, num_routed_wires, degree, num_virtual_targets);
        for row in 0..degree {
            for column in 0..num_wires {
                forest.add(Target::Wire(Wire { row, column }));
            }
        }
        for index in 0..num_virtual_targets {
            forest.add(Target::VirtualTarget { index });
        }
        for &(a, b) in merges {
            forest.merge(a, b);
        }
        forest
    }

    /// Zero-extends the `u32` forest map so it can be compared against `usize` references.
    fn parents_as_usize(forest: &Forest) -> Vec<usize> {
        forest.parents.iter().map(|&p| p as usize).collect()
    }

    /// Verbatim copy of the pre-`u32` `Forest`: `usize` parent pointers throughout. Kept as an
    /// in-test oracle so the narrowed map can be differentially compared entry-for-entry against
    /// the exact code it replaced.
    struct UsizeForest {
        parents: Vec<usize>,
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
    }

    impl UsizeForest {
        fn new(
            num_wires: usize,
            num_routed_wires: usize,
            degree: usize,
            num_virtual_targets: usize,
        ) -> Self {
            let capacity = num_wires * degree + num_virtual_targets;
            Self {
                parents: Vec::with_capacity(capacity),
                num_wires,
                num_routed_wires,
                degree,
            }
        }

        fn target_index(&self, target: Target) -> usize {
            target.index(self.num_wires, self.degree)
        }

        fn add(&mut self, t: Target) {
            let index = self.parents.len();
            debug_assert_eq!(self.target_index(t), index);
            self.parents.push(index);
        }

        fn find(&mut self, mut x_index: usize) -> usize {
            let mut representative = x_index;
            while self.parents[representative] != representative {
                representative = self.parents[representative];
            }
            while self.parents[x_index] != x_index {
                let old_parent = self.parents[x_index];
                self.parents[x_index] = representative;
                x_index = old_parent;
            }
            representative
        }

        fn merge(&mut self, tx: Target, ty: Target) {
            let x_index = self.find(self.target_index(tx));
            let y_index = self.find(self.target_index(ty));
            if x_index == y_index {
                return;
            }
            self.parents[y_index] = x_index;
        }

        fn compress_paths(&mut self) {
            for i in 0..self.parents.len() {
                let parent = self.parents[i];
                if parent == i {
                    continue;
                }
                let mut root = parent;
                while self.parents[root] != root {
                    root = self.parents[root];
                }
                self.parents[i] = root;
            }
        }

        fn wire_partition(&mut self) -> Vec<u32> {
            let mut sigma = vec![0u32; self.degree * self.num_routed_wires];
            let mut last = vec![u32::MAX; self.parents.len()];

            for row in 0..self.degree {
                for column in 0..self.num_routed_wires {
                    let t = Target::Wire(Wire { row, column });
                    let parent = self.parents[self.target_index(t)];
                    let index = (column * self.degree + row) as u32;
                    let old_tail = last[parent];
                    if old_tail == u32::MAX {
                        sigma[index as usize] = index;
                    } else {
                        sigma[index as usize] = sigma[old_tail as usize];
                        sigma[old_tail as usize] = index;
                    }
                    last[parent] = index;
                }
            }

            sigma
        }
    }

    fn build_usize_forest(
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
        merges: &[(Target, Target)],
    ) -> UsizeForest {
        let mut forest = UsizeForest::new(num_wires, num_routed_wires, degree, num_virtual_targets);
        for row in 0..degree {
            for column in 0..num_wires {
                forest.add(Target::Wire(Wire { row, column }));
            }
        }
        for index in 0..num_virtual_targets {
            forest.add(Target::VirtualTarget { index });
        }
        for &(a, b) in merges {
            forest.merge(a, b);
        }
        forest
    }

    fn random_target(
        rng: &mut Lcg,
        num_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
    ) -> Target {
        if num_virtual_targets > 0 && rng.below(8) == 0 {
            Target::VirtualTarget {
                index: rng.below(num_virtual_targets),
            }
        } else {
            Target::Wire(Wire {
                row: rng.below(degree),
                column: rng.below(num_wires),
            })
        }
    }

    fn random_merges(
        rng: &mut Lcg,
        num_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
        count: usize,
    ) -> Vec<(Target, Target)> {
        (0..count)
            .map(|_| {
                (
                    random_target(rng, num_wires, degree, num_virtual_targets),
                    random_target(rng, num_wires, degree, num_virtual_targets),
                )
            })
            .collect()
    }

    /// The promoted tree's `compress_paths`: the general two-walk `find` applied to every index.
    fn reference_compress_paths(parents: &mut [usize]) {
        for i in 0..parents.len() {
            let mut representative = i;
            while parents[representative] != representative {
                representative = parents[representative];
            }
            let mut x_index = i;
            while parents[x_index] != x_index {
                let old_parent = parents[x_index];
                parents[x_index] = representative;
                x_index = old_parent;
            }
        }
    }

    /// The promoted tree's `wire_partition`: open successor chains through `first`/`last`
    /// followed by a closing sweep over the whole forest.
    fn reference_wire_partition(
        parents: &[usize],
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
    ) -> Vec<u32> {
        let mut sigma = vec![0u32; degree * num_routed_wires];
        let mut first = vec![u32::MAX; parents.len()];
        let mut last = vec![u32::MAX; parents.len()];

        for row in 0..degree {
            for column in 0..num_routed_wires {
                let t = Target::Wire(Wire { row, column });
                let parent = parents[t.index(num_wires, degree)];
                let index = (column * degree + row) as u32;
                if first[parent] == u32::MAX {
                    first[parent] = index;
                } else {
                    sigma[last[parent] as usize] = index;
                }
                last[parent] = index;
            }
        }

        for cell in 0..parents.len() {
            if first[cell] != u32::MAX {
                sigma[last[cell] as usize] = first[cell];
            }
        }

        sigma
    }

    /// Independent class-cycle reference: group the routed cells of each representative in scan
    /// order and map `class[i]` to `class[(i + 1) % len]`.
    fn class_cycle_sigma(
        compressed_parents: &[usize],
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
    ) -> Vec<u32> {
        let mut classes: Vec<Vec<u32>> = vec![Vec::new(); compressed_parents.len()];
        for row in 0..degree {
            for column in 0..num_routed_wires {
                let t = Target::Wire(Wire { row, column });
                let root = compressed_parents[t.index(num_wires, degree)];
                classes[root].push((column * degree + row) as u32);
            }
        }

        let mut sigma = vec![0u32; degree * num_routed_wires];
        for class in &classes {
            for (i, &cell) in class.iter().enumerate() {
                sigma[cell as usize] = class[(i + 1) % class.len()];
            }
        }
        sigma
    }

    /// Differential test of circular insertion and one-write compression against a class-cycle
    /// reference, over singleton and multi-element classes with interleaved row/column order.
    #[test]
    fn flat_sigma_insertion_matches_circular_partition_reference() {
        let num_wires = 5;
        let num_routed_wires = 4;
        let num_virtual_targets = 3;

        for degree in [1usize, 2, 7, 16] {
            let mut rng = Lcg(0x5eed_0000 + degree as u64);
            let mut merges = Vec::new();
            if degree >= 2 {
                // Interleaved multi-element classes, a non-routed column, and a virtual target
                // pulled into a routed class, plus a redundant merge.
                merges.push((
                    Target::Wire(Wire { row: 0, column: 1 }),
                    Target::Wire(Wire {
                        row: degree - 1,
                        column: 3,
                    }),
                ));
                merges.push((
                    Target::Wire(Wire {
                        row: degree - 1,
                        column: 0,
                    }),
                    Target::Wire(Wire { row: 0, column: 2 }),
                ));
                merges.push((
                    Target::Wire(Wire { row: 1, column: 4 }),
                    Target::Wire(Wire { row: 0, column: 0 }),
                ));
                merges.push((
                    Target::VirtualTarget { index: 0 },
                    Target::Wire(Wire { row: 1, column: 2 }),
                ));
                merges.push((
                    Target::Wire(Wire { row: 0, column: 1 }),
                    Target::Wire(Wire {
                        row: degree - 1,
                        column: 3,
                    }),
                ));
            } else {
                merges.push((
                    Target::Wire(Wire { row: 0, column: 0 }),
                    Target::Wire(Wire { row: 0, column: 3 }),
                ));
                merges.push((
                    Target::VirtualTarget { index: 1 },
                    Target::Wire(Wire { row: 0, column: 2 }),
                ));
            }
            merges.extend(random_merges(
                &mut rng,
                num_wires,
                degree,
                num_virtual_targets,
                2 * degree + 3,
            ));

            let mut forest = build_forest(
                num_wires,
                num_routed_wires,
                degree,
                num_virtual_targets,
                &merges,
            );
            let original_parents = parents_as_usize(&forest);

            forest.compress_paths();

            // Independently follow the original parent chains from every index and compare the
            // entire compressed-parent vector.
            for i in 0..original_parents.len() {
                let mut root = i;
                while original_parents[root] != root {
                    root = original_parents[root];
                }
                assert_eq!(
                    forest.parents[i] as usize, root,
                    "compressed parent mismatch at index {i} (degree {degree})"
                );
            }

            let compressed = parents_as_usize(&forest);
            let expected = class_cycle_sigma(&compressed, num_wires, num_routed_wires, degree);
            let sigma = forest.wire_partition().sigma;
            assert_eq!(sigma, expected, "sigma mismatch at degree {degree}");
        }
    }

    /// Full-pipeline equivalence against the promoted tree's implementation (two-walk
    /// compression, `first`/`last` chains, closing sweep) on a production-shaped config
    /// (80 routed wires x 2^12 rows) and an odd-sized config.
    #[test]
    fn sigma_matches_promoted_reference_on_production_shapes() {
        // (num_wires, num_routed_wires, degree, num_virtual_targets, merges)
        let configs = [
            (135usize, 80usize, 1usize << 12, 1500usize, 135 * (1 << 12)),
            (7, 5, 999, 41, 5000),
        ];

        for (num_wires, num_routed_wires, degree, num_virtual_targets, num_merges) in configs {
            let mut rng = Lcg(0xfeed_f00d ^ ((degree as u64) << 32) ^ num_wires as u64);
            let merges =
                random_merges(&mut rng, num_wires, degree, num_virtual_targets, num_merges);

            let mut forest = build_forest(
                num_wires,
                num_routed_wires,
                degree,
                num_virtual_targets,
                &merges,
            );
            let mut reference_parents = parents_as_usize(&forest);

            reference_compress_paths(&mut reference_parents);
            let expected_sigma =
                reference_wire_partition(&reference_parents, num_wires, num_routed_wires, degree);

            forest.compress_paths();
            assert_eq!(
                parents_as_usize(&forest),
                reference_parents,
                "compressed parents diverge for degree {degree}"
            );

            let sigma = forest.wire_partition().sigma;
            assert_eq!(sigma, expected_sigma, "sigma diverges for degree {degree}");
        }
    }

    /// M1 differential: the `u32` forest against a verbatim `usize` copy of the code it replaced,
    /// over the whole lifecycle (`add` -> `merge` -> `compress_paths` -> `wire_partition`). Every
    /// parent entry is compared after each mutating stage, so a narrowing bug anywhere in the
    /// forest shows up as a full-vector mismatch rather than only in the derived sigma.
    #[test]
    fn u32_forest_matches_usize_reference() {
        // (num_wires, num_routed_wires, degree, num_virtual_targets, merges)
        let configs = [
            (135usize, 80usize, 1usize << 12, 1500usize, 4 * (1 << 12)),
            (135, 80, 1 << 10, 900, 135 * (1 << 10)),
            (7, 5, 999, 41, 5000),
            (3, 2, 1, 0, 0),
        ];

        for (num_wires, num_routed_wires, degree, num_virtual_targets, num_merges) in configs {
            let mut rng = Lcg(0x0f63_a240 ^ ((degree as u64) << 24) ^ num_wires as u64);
            let merges =
                random_merges(&mut rng, num_wires, degree, num_virtual_targets, num_merges);

            let mut forest = Forest::new(num_wires, num_routed_wires, degree, num_virtual_targets);
            let mut reference =
                UsizeForest::new(num_wires, num_routed_wires, degree, num_virtual_targets);
            for row in 0..degree {
                for column in 0..num_wires {
                    forest.add(Target::Wire(Wire { row, column }));
                    reference.add(Target::Wire(Wire { row, column }));
                }
            }
            for index in 0..num_virtual_targets {
                forest.add(Target::VirtualTarget { index });
                reference.add(Target::VirtualTarget { index });
            }
            assert_eq!(
                parents_as_usize(&forest),
                reference.parents,
                "parents diverge after insertion for degree {degree}"
            );

            for &(a, b) in &merges {
                forest.merge(a, b);
                reference.merge(a, b);
            }
            assert_eq!(
                parents_as_usize(&forest),
                reference.parents,
                "parents diverge after merges for degree {degree}"
            );

            forest.compress_paths();
            reference.compress_paths();
            assert_eq!(
                parents_as_usize(&forest),
                reference.parents,
                "parents diverge after compression for degree {degree}"
            );

            let sigma = forest.wire_partition().sigma;
            assert_eq!(
                sigma,
                reference.wire_partition(),
                "sigma diverges for degree {degree}"
            );
        }

        // A separate check that `find`'s return value and in-place compression agree with the
        // `usize` oracle when driven directly rather than through `compress_paths`.
        let (num_wires, num_routed_wires, degree, num_virtual_targets) = (11usize, 8usize, 64, 20);
        let mut rng = Lcg(0x1e57_a700);
        let merges = random_merges(&mut rng, num_wires, degree, num_virtual_targets, 400);
        let mut forest = build_forest(
            num_wires,
            num_routed_wires,
            degree,
            num_virtual_targets,
            &merges,
        );
        let mut reference = build_usize_forest(
            num_wires,
            num_routed_wires,
            degree,
            num_virtual_targets,
            &merges,
        );
        for i in 0..reference.parents.len() {
            assert_eq!(forest.find(i), reference.find(i), "find diverges at {i}");
            assert_eq!(parents_as_usize(&forest), reference.parents);
        }
    }

    /// `Forest::new` rejects a target count that would not fit the narrowed map rather than
    /// silently truncating indices.
    #[test]
    #[should_panic(expected = "exceeds the u32 index range")]
    fn forest_rejects_capacity_above_u32_max() {
        let _ = Forest::new(135, 80, 1 << 26, 0);
    }

    #[test]
    fn compressed_forest_loader_rejects_malformed_maps() {
        let valid = vec![0, 0, 2, 2];
        let forest = Forest::from_compressed_parents(valid.clone(), 2, 2, 1)
            .expect("fully compressed in-range map");
        assert_eq!(forest.into_parents(), valid);

        assert!(Forest::from_compressed_parents(vec![0], 2, 2, 1).is_none());
        assert!(Forest::from_compressed_parents(vec![0, 1], 1, 2, 1).is_none());
        assert!(Forest::from_compressed_parents(vec![0, 2], 2, 2, 1).is_none());
        assert!(Forest::from_compressed_parents(vec![0, 0, 1], 2, 2, 1).is_none());
        assert!(Forest::from_compressed_parents(vec![], 2, 2, usize::MAX).is_none());
    }

    #[test]
    fn trusted_forest_fallible_partition_checks_bounds_in_its_required_scan() {
        let valid = vec![0, 0, 2, 2];
        let mut checked = Forest::from_compressed_parents(valid.clone(), 2, 2, 1).unwrap();
        let expected = checked.wire_partition().sigma;
        let mut trusted = Forest::from_trusted_compressed_parents(valid, 2, 2, 1);
        assert_eq!(trusted.try_wire_partition().unwrap().sigma, expected);

        let mut short = Forest::from_trusted_compressed_parents(vec![0], 2, 2, 1);
        assert!(short.try_wire_partition().is_none());
        let mut out_of_range = Forest::from_trusted_compressed_parents(vec![0, 2], 2, 2, 1);
        assert!(out_of_range.try_wire_partition().is_none());
        let mut too_many_routed =
            Forest::from_trusted_compressed_parents(vec![0, 1], 1, 2, 1);
        assert!(too_many_routed.try_wire_partition().is_none());
    }

    #[test]
    fn permutation_factor_skip_mask_lengths_cover_chunk_padding() {
        assert_eq!(permutation_factor_skip_mask_len(4, 80, 8), Some(80));
        assert_eq!(permutation_factor_skip_mask_len(3, 10, 4), Some(18));
        assert_eq!(permutation_factor_skip_mask_len(2, 10, 9), Some(16));
        assert_eq!(permutation_factor_skip_mask_len(1, 1, 1), Some(2));
        assert_eq!(permutation_factor_skip_mask_len(1, 0, 8), None);
        assert_eq!(permutation_factor_skip_mask_len(1, 80, 0), None);
        assert_eq!(
            permutation_factor_skip_mask_len(usize::MAX, 80, 8),
            None
        );
    }

    /// `get_sigma_polys` (shift/mask decode, outer-parallel columns) against the promoted
    /// tree's sequential division/remainder arithmetic.
    #[test]
    fn sigma_polys_match_division_reference() {
        type F = GoldilocksField;

        let num_wires = 135;
        let num_routed_wires = 80;
        let degree_log = 12;
        let degree = 1 << degree_log;
        let num_virtual_targets = 700;

        let mut rng = Lcg(0xabcd_ef01);
        let merges = random_merges(&mut rng, num_wires, degree, num_virtual_targets, 3 * degree);
        let mut forest = build_forest(
            num_wires,
            num_routed_wires,
            degree,
            num_virtual_targets,
            &merges,
        );
        forest.compress_paths();
        let partition = forest.wire_partition();

        let k_is: Vec<F> = (0..num_routed_wires)
            .map(|_| F::from_canonical_u64(rng.next_u64()))
            .collect();
        let subgroup: Vec<F> = (0..degree)
            .map(|_| F::from_canonical_u64(rng.next_u64()))
            .collect();

        let expected: Vec<PolynomialValues<F>> = partition
            .sigma
            .chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .iter()
                    .map(|&x| k_is[x as usize / degree] * subgroup[x as usize % degree])
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect();

        let actual = partition.get_sigma_polys(degree_log, &k_is, &subgroup);
        assert_eq!(actual, expected);
    }

    /// The runtime cancellation mask follows directed sigma edges, not representative identity.
    /// Virtual/advice aliases do not enter sigma; singleton routed components still self-cancel,
    /// and nontrivial edges cancel only when both endpoints occupy one row and quotient chunk.
    #[test]
    fn permutation_factor_skip_mask_matches_sigma_edges_with_aliases() {
        let (num_wires, num_routed_wires, degree, num_virtual_targets) = (5, 3, 4, 2);
        let quotient_degree_factor = 2;
        let merges = [
            // Singleton routed component with a virtual alias.
            (
                Target::Wire(Wire { row: 0, column: 0 }),
                Target::VirtualTarget { index: 0 },
            ),
            // Singleton routed component with an advice-wire alias.
            (
                Target::Wire(Wire { row: 0, column: 1 }),
                Target::Wire(Wire { row: 2, column: 4 }),
            ),
            // Two separate multi-routed components.
            (
                Target::Wire(Wire { row: 0, column: 2 }),
                Target::Wire(Wire { row: 1, column: 0 }),
            ),
            (
                Target::Wire(Wire { row: 1, column: 1 }),
                Target::Wire(Wire { row: 1, column: 2 }),
            ),
            (
                Target::Wire(Wire { row: 1, column: 2 }),
                Target::VirtualTarget { index: 1 },
            ),
            // A non-singleton two-cycle wholly inside row 2, quotient chunk 0.
            (
                Target::Wire(Wire { row: 2, column: 0 }),
                Target::Wire(Wire { row: 2, column: 1 }),
            ),
        ];
        let mut forest = build_forest(
            num_wires,
            num_routed_wires,
            degree,
            num_virtual_targets,
            &merges,
        );
        forest.compress_paths();
        let partition = forest.wire_partition();
        let mask = partition
            .permutation_factor_skip_mask(
                degree,
                num_routed_wires,
                quotient_degree_factor,
            )
            .expect("valid sigma permutation");
        assert_eq!(
            mask.len(),
            degree
                * num_routed_wires.div_ceil(quotient_degree_factor)
                * quotient_degree_factor.div_ceil(8)
                * 2
        );

        let mut expected = vec![0u8; degree * num_routed_wires];
        for source_column in 0..num_routed_wires {
            for source_row in 0..degree {
                let source_column_major = source_column * degree + source_row;
                let target_column_major = partition.sigma[source_column_major] as usize;
                let target_column = target_column_major / degree;
                let target_row = target_column_major % degree;
                if source_row == target_row
                    && source_column / quotient_degree_factor
                        == target_column / quotient_degree_factor
                {
                    expected[source_row * num_routed_wires + source_column] |=
                        PERMUTATION_SKIP_DENOMINATOR;
                    expected[target_row * num_routed_wires + target_column] |=
                        PERMUTATION_SKIP_NUMERATOR;
                }
            }
        }
        for (index, &expected_state) in expected.iter().enumerate() {
            assert_eq!(
                permutation_factor_skips(
                    &mask,
                    index,
                    num_routed_wires,
                    quotient_degree_factor,
                ),
                expected_state,
                "mask/sigma edge mismatch at row-major position {index}"
            );
        }
        assert_eq!(
            permutation_factor_skips(&mask, 0, num_routed_wires, quotient_degree_factor),
            PERMUTATION_SKIP_BOTH
        );
        assert_eq!(
            permutation_factor_skips(&mask, 1, num_routed_wires, quotient_degree_factor),
            PERMUTATION_SKIP_BOTH
        );
        assert_eq!(
            permutation_factor_skips(&mask, 2, num_routed_wires, quotient_degree_factor),
            0
        );
        assert_eq!(
            permutation_factor_skips(
                &mask,
                2 * num_routed_wires,
                num_routed_wires,
                quotient_degree_factor,
            ),
            PERMUTATION_SKIP_BOTH
        );
        assert_eq!(
            permutation_factor_skips(
                &mask,
                2 * num_routed_wires + 1,
                num_routed_wires,
                quotient_degree_factor,
            ),
            PERMUTATION_SKIP_BOTH
        );
    }
}

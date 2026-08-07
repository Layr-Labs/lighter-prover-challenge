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

/// Disjoint Set Forest data-structure following <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
pub struct Forest {
    /// A map of parent pointers, stored as indices.
    pub(crate) parents: Vec<usize>,

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
        Self {
            parents: Vec::with_capacity(capacity),
            num_wires,
            num_routed_wires,
            degree,
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    /// Add a new partition with a single member.
    pub fn add(&mut self, t: Target) {
        let index = self.parents.len();
        debug_assert_eq!(self.target_index(t), index);
        self.parents.push(index);
    }

    /// Path compression method, see <https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives>.
    pub fn find(&mut self, mut x_index: usize) -> usize {
        // Note: We avoid recursion here since the chains can be long, causing stack overflows.

        // First, find the representative of the set containing `x_index`.
        let mut representative = x_index;
        while self.parents[representative] != representative {
            representative = self.parents[representative];
        }

        // Then, update each node in this chain to point directly to the representative.
        while self.parents[x_index] != x_index {
            let old_parent = self.parents[x_index];
            self.parents[x_index] = representative;
            x_index = old_parent;
        }

        representative
    }

    /// Merge two sets.
    pub fn merge(&mut self, tx: Target, ty: Target) {
        let x_index = self.find(self.target_index(tx));
        let y_index = self.find(self.target_index(ty));

        if x_index == y_index {
            return;
        }

        self.parents[y_index] = x_index;
    }

    /// Compress all paths. After calling this, every `parent` value will point to the node's
    /// representative.
    ///
    /// This dedicated full pass visits every index once and gives it its own direct-root write,
    /// so the general `find`'s second chain walk (which rewrites intermediate nodes) is
    /// unnecessary: each intermediate node receives its direct-root assignment when the outer
    /// loop reaches it. Roots are stable during this pass, so the final `parents` vector is
    /// identical to calling `find(i)` for every `i`.
    /// Parallel refinement: roots are fixed by the merge history and invariant under
    /// compression order, so per-element root resolution is a pure function of the
    /// immutable pre-pass array. Resolving every root against a read-only borrow and
    /// installing the result wholesale produces the identical final vector while
    /// spreading the tens-of-millions-entry walk across cores.
    pub(crate) fn compress_paths(&mut self) {
        let parents = &self.parents;
        let roots: Vec<usize> = (0..parents.len())
            .into_par_iter()
            .map(|i| {
                let mut root = i;
                while parents[root] != root {
                    root = parents[root];
                }
                root
            })
            .collect();
        self.parents = roots;
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

        WirePartition { sigma }
    }
}

pub struct WirePartition {
    sigma: Vec<u32>,
}

impl WirePartition {
    pub(crate) fn get_sigma_polys<F: Field>(
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
            let original_parents = forest.parents.clone();

            forest.compress_paths();

            // Independently follow the original parent chains from every index and compare the
            // entire compressed-parent vector.
            for i in 0..original_parents.len() {
                let mut root = i;
                while original_parents[root] != root {
                    root = original_parents[root];
                }
                assert_eq!(
                    forest.parents[i], root,
                    "compressed parent mismatch at index {i} (degree {degree})"
                );
            }

            let expected = class_cycle_sigma(&forest.parents, num_wires, num_routed_wires, degree);
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
            let mut reference_parents = forest.parents.clone();

            reference_compress_paths(&mut reference_parents);
            let expected_sigma =
                reference_wire_partition(&reference_parents, num_wires, num_routed_wires, degree);

            forest.compress_paths();
            assert_eq!(
                forest.parents, reference_parents,
                "compressed parents diverge for degree {degree}"
            );

            let sigma = forest.wire_partition().sigma;
            assert_eq!(sigma, expected_sigma, "sigma diverges for degree {degree}");
        }
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
}

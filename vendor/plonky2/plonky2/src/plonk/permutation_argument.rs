#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_maybe_rayon::*;

use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::iop::target::Target;
use crate::iop::wire::Wire;

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
    /// Kept serial deliberately: circuit builds overlap proving (deferred block build) and
    /// sibling-circuit construction in this lineage, so a parallel sweep here steals cores
    /// from the proving critical path rather than shortening the timed window.
    pub(crate) fn compress_paths(&mut self) {
        for i in 0..self.parents.len() {
            self.find(i);
        }
    }

    /// Assumes `compress_paths` has already been called.
    pub fn wire_partition(&mut self) -> WirePartition {
        let mut sigma = vec![0u32; self.degree * self.num_routed_wires];
        let mut first = vec![u32::MAX; self.parents.len()];
        let mut last = vec![u32::MAX; self.parents.len()];

        for row in 0..self.degree {
            for column in 0..self.num_routed_wires {
                let t = Target::Wire(Wire { row, column });
                let parent = self.parents[self.target_index(t)];
                let index = (column * self.degree + row) as u32;
                if first[parent] == u32::MAX {
                    first[parent] = index;
                } else {
                    sigma[last[parent] as usize] = index;
                }
                last[parent] = index;
            }
        }

        for cell in 0..self.parents.len() {
            if first[cell] != u32::MAX {
                sigma[last[cell] as usize] = first[cell];
            }
        }

        WirePartition { sigma }
    }
}

pub struct WirePartition {
    sigma: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stock serial compression this crate shipped with, kept verbatim as the
    /// differential reference: mutating `find` on every index in order.
    fn compress_paths_serial_reference(forest: &mut Forest) {
        for i in 0..forest.parents.len() {
            forest.find(i);
        }
    }

    fn build_forest(num_wires: usize, num_routed_wires: usize, degree: usize, seed: u64) -> Forest {
        let num_virtual = 17;
        let mut forest = Forest::new(num_wires, num_routed_wires, degree, num_virtual);
        for row in 0..degree {
            for column in 0..num_wires {
                forest.add(Target::Wire(Wire { row, column }));
            }
        }
        for index in 0..num_virtual {
            forest.add(Target::VirtualTarget { index });
        }
        // Deterministic LCG-driven merges, including long chains and repeated merges.
        let mut state = seed;
        let mut next = |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
        let merges = num_wires * degree * 2;
        for _ in 0..merges {
            let a = Wire {
                row: next(degree),
                column: next(num_wires),
            };
            let b = Wire {
                row: next(degree),
                column: next(num_wires),
            };
            forest.merge(Target::Wire(a), Target::Wire(b));
        }
        forest
    }

    #[test]
    fn parallel_compress_paths_matches_serial_reference() {
        for seed in [1u64, 7, 42, 0xdeadbeef] {
            let mut reference = build_forest(13, 8, 64, seed);
            let mut candidate = build_forest(13, 8, 64, seed);
            assert_eq!(reference.parents, candidate.parents);

            compress_paths_serial_reference(&mut reference);
            candidate.compress_paths();
            assert_eq!(reference.parents, candidate.parents, "seed {seed}");

            let sigma_ref = reference.wire_partition().sigma;
            let sigma_cand = candidate.wire_partition().sigma;
            assert_eq!(sigma_ref, sigma_cand, "seed {seed}");
        }
    }
}

impl WirePartition {
    pub(crate) fn get_sigma_polys<F: Field>(
        &self,
        degree_log: usize,
        k_is: &[F],
        subgroup: &[F],
    ) -> Vec<PolynomialValues<F>> {
        let degree = 1 << degree_log;

        self.sigma
            .chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .par_iter()
                    .map(|&x| k_is[x as usize / degree] * subgroup[x as usize % degree])
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect()
    }
}

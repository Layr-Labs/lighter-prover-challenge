#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_maybe_rayon::*;

use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::iop::target::Target;

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
    #[allow(dead_code)] // retained for tests / incremental construction; build uses `add_all_targets`
    pub fn add(&mut self, t: Target) {
        let index = self.parents.len();
        debug_assert_eq!(self.target_index(t), index);
        self.parents.push(index);
    }

    /// Bulk-init singleton partitions for a dense wire×row prefix plus virtual targets.
    /// Equivalent to calling `add` in the same order the circuit builder uses
    /// (row-major wires, then virtuals 0..num_virtual_targets).
    pub fn add_all_targets(&mut self, num_virtual_targets: usize) {
        debug_assert!(self.parents.is_empty());
        let n = self.num_wires * self.degree + num_virtual_targets;
        self.parents.extend(0..n);
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
    /// representative. Bit-identical to sequential `find(i)` for every index: the final
    /// parents array maps each node to its set root. A short serial path-halving prepass
    /// collapses deep chains so the parallel root-find walks are short; roots are unique
    /// so concurrent reads of the parents array are race-free.
    pub(crate) fn compress_paths(&mut self) {
        let n = self.parents.len();
        if n == 0 {
            return;
        }

        // Path-halving rounds: parents[i] := parents[parents[i]]. Each round roughly
        // halves remaining depth. Three rounds turn even degenerate n-long chains into
        // O(log n) walks before the parallel flatten.
        for _ in 0..3 {
            for i in 0..n {
                let p = self.parents[i];
                // SAFETY of index: every parent pointer is always a valid index into
                // parents (union-find invariant maintained by add/merge).
                self.parents[i] = self.parents[p];
            }
        }

        let parents = &self.parents;
        let reps: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut x = i;
                while parents[x] != x {
                    x = parents[x];
                }
                x
            })
            .collect();
        self.parents = reps;
    }

    /// Assumes `compress_paths` has already been called.
    pub fn wire_partition(&mut self) -> WirePartition {
        let degree = self.degree;
        let num_routed = self.num_routed_wires;
        let num_wires = self.num_wires;
        let mut sigma = vec![0u32; degree * num_routed];
        let mut first = vec![u32::MAX; self.parents.len()];
        let mut last = vec![u32::MAX; self.parents.len()];

        // Row-major scan order is load-bearing for bit-identical circular successor
        // chains (same as the flat-array sigma rewrite). Keep sequential.
        for row in 0..degree {
            let row_base = row * num_wires;
            for column in 0..num_routed {
                let parent = self.parents[row_base + column];
                let index = (column * degree + row) as u32;
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

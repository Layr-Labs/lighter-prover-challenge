#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_maybe_rayon::*;
use core::cmp::Ordering;

use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::iop::target::Target;
use crate::iop::wire::Wire;

/// Disjoint Set Forest data-structure following <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
pub struct Forest {
    /// A map of parent pointers, stored as indices.
    pub(crate) parents: Vec<usize>,

    /// Rank of each set, used for union-by-rank to keep trees flat.
    ranks: Vec<u8>,

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
            ranks: Vec::with_capacity(capacity),
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
        self.ranks.push(0);
    }

    /// Path compression method, see <https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives>.
    pub fn find(&mut self, mut x_index: usize) -> usize {
        // First, find the representative of the set containing `x_index`.
        let mut representative = x_index;
        while self.parents[representative] != representative {
            representative = self.parents[representative];
        }

        // Then, update each node in this chain to point directly to the representative.
        // Micro-opt: compare against `representative` directly instead of re-reading memory.
        while x_index != representative {
            let old_parent = self.parents[x_index];
            self.parents[x_index] = representative;
            x_index = old_parent;
        }

        representative
    }

    /// Merge two sets using union by rank for near-constant amortized time.
    pub fn merge(&mut self, tx: Target, ty: Target) {
        let x_index = self.find(self.target_index(tx));
        let y_index = self.find(self.target_index(ty));

        if x_index == y_index {
            return;
        }

        // Union by rank: attach the shorter tree under the taller one.
        match self.ranks[x_index].cmp(&self.ranks[y_index]) {
            Ordering::Less => {
                self.parents[x_index] = y_index;
            }
            Ordering::Greater => {
                self.parents[y_index] = x_index;
            }
            Ordering::Equal => {
                self.parents[y_index] = x_index;
                self.ranks[x_index] = self.ranks[x_index].saturating_add(1);
            }
        }
    }

    /// Compress all paths. After calling this, every `parent` value will point to the node's
    /// representative.
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
        let mut representatives = Vec::new();

        // Loop reordering: iterate column-first so that target_index(column, row) produces
        // contiguous indices, yielding much better cache locality for `self.parents` and `sigma`.
        for column in 0..self.num_routed_wires {
            for row in 0..self.degree {
                let t = Target::Wire(Wire { row, column });
                let parent = self.parents[self.target_index(t)];
                let index = (column * self.degree + row) as u32;

                if first[parent] == u32::MAX {
                    first[parent] = index;
                    representatives.push(parent);
                } else {
                    sigma[last[parent] as usize] = index;
                }
                last[parent] = index;
            }
        }

        // Only iterate over representatives that actually appeared, skipping virtual targets.
        for &rep in &representatives {
            sigma[last[rep] as usize] = first[rep];
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
        let degree_mask = degree - 1;

        self.sigma
            .chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .par_iter()
                    .map(|&x| {
                        let x = x as usize;
                        // Explicit bit-math: degree is a power of two, so div/mod become shift/mask.
                        k_is[x >> degree_log] * subgroup[x & degree_mask]
                    })
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect()
    }
}
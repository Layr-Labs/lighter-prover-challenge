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
    pub(crate) fn compress_paths(&mut self) {
        // A full pass only needs to write each entry's representative; it does not need
        // `find`'s second walk because every entry is visited exactly once here. In
        // particular, singleton entries now take one parent load instead of two.
        for i in 0..self.parents.len() {
            let parent = self.parents[i];
            if parent == i {
                continue;
            }

            let mut representative = parent;
            loop {
                let next = self.parents[representative];
                if next == representative {
                    break;
                }
                representative = next;
            }
            self.parents[i] = representative;
        }
    }

    /// Assumes `compress_paths` has already been called.
    pub fn wire_partition(&mut self) -> WirePartition {
        let mut sigma = vec![0u32; self.degree * self.num_routed_wires];
        let mut last = vec![u32::MAX; self.parents.len()];

        for row in 0..self.degree {
            for column in 0..self.num_routed_wires {
                let t = Target::Wire(Wire { row, column });
                let parent = self.parents[self.target_index(t)];
                let index = (column * self.degree + row) as u32;
                let previous = last[parent];
                if previous == u32::MAX {
                    // Start a one-element circular list.
                    sigma[index as usize] = index;
                } else {
                    // Insert `index` after the old tail while retaining the head in the
                    // old tail's successor slot: old_tail -> index -> head.
                    sigma[index as usize] = sigma[previous as usize];
                    sigma[previous as usize] = index;
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

        self.sigma
            .par_chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .iter()
                    .map(|&x| {
                        let encoded = x as usize;
                        k_is[encoded >> degree_log] * subgroup[encoded & (degree - 1)]
                    })
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_sigma(forest: &Forest) -> Vec<u32> {
        let mut classes = vec![Vec::<u32>::new(); forest.parents.len()];
        for row in 0..forest.degree {
            for column in 0..forest.num_routed_wires {
                let target = Target::Wire(Wire { row, column });
                let parent = forest.parents[forest.target_index(target)];
                classes[parent].push((column * forest.degree + row) as u32);
            }
        }

        let mut sigma = vec![0u32; forest.degree * forest.num_routed_wires];
        for class in classes.into_iter().filter(|class| !class.is_empty()) {
            for i in 0..class.len() {
                sigma[class[i] as usize] = class[(i + 1) % class.len()];
            }
        }
        sigma
    }

    #[test]
    fn flat_sigma_insertion_matches_circular_partition_reference() {
        for degree in [1, 2, 7, 16] {
            let num_wires = 5;
            let num_routed_wires = 4;
            let mut forest = Forest::new(num_wires, num_routed_wires, degree, 0);
            for row in 0..degree {
                for column in 0..num_wires {
                    forest.add(Target::Wire(Wire { row, column }));
                }
            }

            let mut anchors = [None; 5];
            for row in 0..degree {
                for column in 0..num_routed_wires {
                    let target = Target::Wire(Wire { row, column });
                    let class = (row * 3 + column * 2) % anchors.len();
                    if let Some(anchor) = anchors[class] {
                        forest.merge(anchor, target);
                    } else {
                        anchors[class] = Some(target);
                    }
                }
            }
            let expected_parents = (0..forest.parents.len())
                .map(|start| {
                    let mut representative = start;
                    while forest.parents[representative] != representative {
                        representative = forest.parents[representative];
                    }
                    representative
                })
                .collect::<Vec<_>>();
            forest.compress_paths();
            assert_eq!(forest.parents, expected_parents, "degree={degree}");
            let expected = reference_sigma(&forest);
            let actual = forest.wire_partition().sigma;
            assert_eq!(actual, expected, "degree={degree}");
        }
    }
}

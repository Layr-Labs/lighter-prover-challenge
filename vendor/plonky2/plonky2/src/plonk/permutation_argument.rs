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
        for i in 0..self.parents.len() {
            self.find(i);
        }
    }

    /// Assumes `compress_paths` has already been called.
    pub fn wire_partition(&mut self) -> WirePartition {
        // Thread every routed wire onto its copy class's circular successor chain in one dense
        // row-major pass, using two flat per-representative cursors instead of a map of class
        // vectors plus a neighbor map. `parents` is fully compressed, so each routed wire's entry
        // is its class representative. A closing sweep links each class's last wire back to its
        // first, which also maps singleton classes to themselves. The successor of a wire is the
        // next routed wire of its class in scan order, circularly — exactly the neighbor the
        // map-based construction produced, so the resulting sigma polynomials are bit-identical.
        const UNSET: u32 = u32::MAX;
        let num_slots = self.degree * self.num_routed_wires;
        let mut first = vec![UNSET; self.parents.len()];
        let mut last = vec![UNSET; self.parents.len()];
        let mut next = vec![0u32; num_slots];

        let mut scan_index = 0u32;
        for row in 0..self.degree {
            let row_base = row * self.num_wires;
            for column in 0..self.num_routed_wires {
                let rep = self.parents[row_base + column];
                if first[rep] == UNSET {
                    first[rep] = scan_index;
                } else {
                    next[last[rep] as usize] = scan_index;
                }
                last[rep] = scan_index;
                scan_index += 1;
            }
        }
        for rep in 0..first.len() {
            if first[rep] != UNSET {
                next[last[rep] as usize] = first[rep];
            }
        }

        WirePartition { next }
    }
}

pub struct WirePartition {
    /// For each routed wire, indexed in row-major scan order (`row * num_routed_wires + column`),
    /// the scan index of the next wire in its copy class (circular; singletons point to
    /// themselves).
    next: Vec<u32>,
}

impl WirePartition {
    pub(crate) fn get_sigma_polys<F: Field>(
        &self,
        degree_log: usize,
        k_is: &[F],
        subgroup: &[F],
    ) -> Vec<PolynomialValues<F>> {
        let degree = 1 << degree_log;
        let sigma = self.get_sigma_map(degree, k_is.len());

        sigma
            .chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .par_iter()
                    .map(|&x| k_is[x / degree] * subgroup[x % degree])
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect()
    }

    /// Generates sigma in the context of Plonk, which is a map from `[kn]` to `[kn]`, where `k` is
    /// the number of routed wires and `n` is the number of gates.
    fn get_sigma_map(&self, degree: usize, num_routed_wires: usize) -> Vec<usize> {
        // A wire's "neighbor" in the context of Plonk's "extended copy constraints" check is the
        // next wire in the given wire's partition, looping around at the end; a wire with a
        // partition all to itself is its own neighbor. The successor chain holds exactly these
        // neighbors, keyed by row-major scan index.
        let mut sigma = Vec::with_capacity(num_routed_wires * degree);
        for column in 0..num_routed_wires {
            for row in 0..degree {
                let neighbor = self.next[row * num_routed_wires + column] as usize;
                let neighbor_row = neighbor / num_routed_wires;
                let neighbor_column = neighbor % num_routed_wires;
                sigma.push(neighbor_column * degree + neighbor_row);
            }
        }
        sigma
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::poseidon::PoseidonHash;
    use crate::iop::wire::Wire;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;

    /// The previous map-of-classes plus neighbor-map sigma construction, kept verbatim as the
    /// reference the flat successor-chain construction must match bit for bit.
    fn reference_sigma_map(
        parents: &[usize],
        degree: usize,
        num_wires: usize,
        num_routed_wires: usize,
    ) -> Vec<usize> {
        let mut partition = HashMap::<usize, Vec<Wire>>::new();
        for row in 0..degree {
            for column in 0..num_routed_wires {
                let w = Wire { row, column };
                let t = Target::Wire(w);
                partition
                    .entry(parents[t.index(num_wires, degree)])
                    .or_default()
                    .push(w);
            }
        }
        let partition: Vec<Vec<Wire>> = partition.into_values().collect();

        let mut neighbors = HashMap::with_capacity(partition.len());
        for subset in &partition {
            for n in 0..subset.len() {
                neighbors.insert(subset[n], subset[(n + 1) % subset.len()]);
            }
        }

        let mut sigma = Vec::with_capacity(num_routed_wires * degree);
        for column in 0..num_routed_wires {
            for row in 0..degree {
                let wire = Wire { row, column };
                let neighbor = neighbors[&wire];
                sigma.push(neighbor.column * degree + neighbor.row);
            }
        }
        sigma
    }

    #[test]
    fn flat_sigma_map_matches_reference_on_recursive_circuit() {
        // Real production-shaped circuits: a hash chain with public inputs, and a recursive
        // verifier of its proof shape — copy classes of many sizes including singletons.
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let x = builder.add_virtual_target();
        let mut state = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![x]);
        for _ in 0..4 {
            state = builder.hash_n_to_hash_no_pad::<PoseidonHash>(state.elements.to_vec());
        }
        builder.register_public_inputs(&state.elements);
        let inner = builder.build::<C>();

        let mut builder = CircuitBuilder::<F, D>::new(config);
        let proof = builder.add_virtual_proof_with_pis(&inner.common);
        let verifier_data = builder.constant_verifier_data(&inner.verifier_only);
        builder.verify_proof::<C>(&proof, &verifier_data, &inner.common);
        builder.register_public_inputs(&proof.public_inputs);
        let outer = builder.build::<C>();

        for circuit in [&inner, &outer] {
            let num_wires = circuit.common.config.num_wires;
            let num_routed_wires = circuit.common.config.num_routed_wires;
            let degree = circuit.common.degree();

            // `representative_map` is the fully compressed forest the build used.
            let mut forest = Forest {
                parents: circuit.prover_only.representative_map.clone(),
                num_wires,
                num_routed_wires,
                degree,
            };
            let flat = forest
                .wire_partition()
                .get_sigma_map(degree, num_routed_wires);
            let reference = reference_sigma_map(
                &circuit.prover_only.representative_map,
                degree,
                num_wires,
                num_routed_wires,
            );
            assert_eq!(flat, reference);
        }
    }
}

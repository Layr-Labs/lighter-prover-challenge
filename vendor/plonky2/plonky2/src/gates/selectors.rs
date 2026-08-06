#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::ops::Range;

use hashbrown::HashMap;
use serde::Serialize;

use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialValues;
use crate::gates::gate::{GateInstance, GateRef};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_builder::LookupWire;

/// Placeholder value to indicate that a gate doesn't use a selector polynomial.
pub(crate) const UNUSED_SELECTOR: usize = u32::MAX as usize;

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct SelectorsInfo {
    pub(crate) selector_indices: Vec<usize>,
    pub(crate) groups: Vec<Range<usize>>,
}

impl SelectorsInfo {
    pub fn num_selectors(&self) -> usize {
        self.groups.len()
    }
}

/// Enum listing the different selectors for lookup constraints:
/// - `TransSre` is for Sum and RE transition constraints.
/// - `TransLdc` is for LDC transition constraints.
/// - `InitSre` is for the initial constraint of Sum and Re.
/// - `LastLdc` is for the final LDC (and Sum) constraint.
/// - `StartEnd` indicates where lookup end selectors begin.
pub enum LookupSelectors {
    TransSre = 0,
    TransLdc,
    InitSre,
    LastLdc,
    StartEnd,
}

/// Returns selector polynomials for each LUT. We have two constraint domains (remember that gates are stored upside down):
/// - [last_lut_row, first_lut_row] (Sum and RE transition constraints),
/// - [last_lu_row, last_lut_row - 1] (LDC column transition constraints).
///
/// We also add two more:
/// - {first_lut_row + 1} where we check the initial values of sum and RE (which are 0),
/// - {last_lu_row} where we check that the last value of LDC is 0.
///
/// Conceptually they're part of the selector ends lookups, but since we can have one polynomial for *all* LUTs it's here.
pub(crate) fn selectors_lookup<F: RichField + Extendable<D>, const D: usize>(
    _gates: &[GateRef<F, D>],
    instances: &[GateInstance<F, D>],
    lookup_rows: &[LookupWire],
) -> Vec<PolynomialValues<F>> {
    let n = instances.len();
    let mut lookup_selectors = Vec::with_capacity(LookupSelectors::StartEnd as usize);
    for _ in 0..LookupSelectors::StartEnd as usize {
        lookup_selectors.push(PolynomialValues::<F>::new(vec![F::ZERO; n]));
    }

    for &LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in lookup_rows
    {
        for row in last_lut_row..first_lut_row + 1 {
            lookup_selectors[LookupSelectors::TransSre as usize].values[row] = F::ONE;
        }
        for row in last_lu_row..last_lut_row {
            lookup_selectors[LookupSelectors::TransLdc as usize].values[row] = F::ONE;
        }
        lookup_selectors[LookupSelectors::InitSre as usize].values[first_lut_row + 1] = F::ONE;
        lookup_selectors[LookupSelectors::LastLdc as usize].values[last_lu_row] = F::ONE;
    }
    lookup_selectors
}

/// Returns selectors for checking the validity of the LUTs.
/// Each selector equals one on its respective LUT's `last_lut_row`, and 0 elsewhere.
pub(crate) fn selector_ends_lookups<F: RichField + Extendable<D>, const D: usize>(
    lookup_rows: &[LookupWire],
    instances: &[GateInstance<F, D>],
) -> Vec<PolynomialValues<F>> {
    let n = instances.len();
    let mut lookups_ends = Vec::with_capacity(lookup_rows.len());
    for &LookupWire {
        last_lu_gate: _,
        last_lut_gate: last_lut_row,
        first_lut_gate: _,
    } in lookup_rows
    {
        let mut lookup_ends = PolynomialValues::<F>::new(vec![F::ZERO; n]);
        lookup_ends.values[last_lut_row] = F::ONE;
        lookups_ends.push(lookup_ends);
    }
    lookups_ends
}

/// Returns the selector polynomials and related information.
///
/// Selector polynomials are computed as follows:
/// Partition the gates into (the smallest amount of) groups `{ G_i }`, such that for each group `G`
/// `|G| + max_{g in G} g.degree() <= max_degree`. These groups are constructed greedily from
/// the list of gates sorted by degree.
/// We build a selector polynomial `S_i` for each group `G_i`, with
/// S_i\[j\] =
///     if j-th row gate=g_k in G_i
///         k
///     else
///         UNUSED_SELECTOR
pub(crate) fn selector_polynomials<F: RichField + Extendable<D>, const D: usize>(
    gates: &[GateRef<F, D>],
    instances: &[GateInstance<F, D>],
    max_degree: usize,
) -> (Vec<PolynomialValues<F>>, SelectorsInfo) {
    let n = instances.len();
    let num_gates = gates.len();
    let max_gate_degree = gates.last().expect("No gates?").0.degree();

    let gate_indices: HashMap<*const (), usize> = gates
        .iter()
        .enumerate()
        .map(|(index, gate)| (gate.as_ptr(), index))
        .collect();
    assert_eq!(gate_indices.len(), gates.len());
    let index = |gate: &GateRef<F, D>| {
        *gate_indices
            .get(&gate.as_ptr())
            .expect("gate instance does not use its canonical gate reference")
    };

    // Special case if we can use only one selector polynomial.
    if max_gate_degree + num_gates - 1 <= max_degree {
        // We *want* `groups` to be a vector containing one Range (all gates are in one selector group),
        // but Clippy doesn't trust us.
        #[allow(clippy::single_range_in_vec_init)]
        return (
            vec![PolynomialValues::new(
                instances
                    .iter()
                    .map(|g| F::from_canonical_usize(index(&g.gate_ref)))
                    .collect(),
            )],
            SelectorsInfo {
                selector_indices: vec![0; num_gates],
                groups: vec![0..num_gates],
            },
        );
    }

    if max_gate_degree >= max_degree {
        panic!(
            "{} has too high degree. Consider increasing `quotient_degree_factor`.",
            gates.last().unwrap().0.id()
        );
    }

    // Greedily construct the groups.
    let mut groups = Vec::new();
    let mut start = 0;
    while start < num_gates {
        let mut size = 0;
        while (start + size < gates.len()) && (size + gates[start + size].0.degree() < max_degree) {
            size += 1;
        }
        groups.push(start..start + size);
        start += size;
    }

    // `selector_indices[i] = j` iff the `i`-th gate uses the `j`-th selector polynomial.
    let mut selector_indices = vec![0; num_gates];
    for (group, range) in groups.iter().enumerate() {
        selector_indices[range.clone()].fill(group);
    }

    // Start with the placeholder for every inactive group, then overwrite the one active
    // selector for each row.
    let unused = F::from_canonical_usize(UNUSED_SELECTOR);
    let mut polynomials = (0..groups.len())
        .map(|_| PolynomialValues::new(vec![unused; n]))
        .collect::<Vec<_>>();
    for (row, instance) in instances.iter().enumerate() {
        let i = index(&instance.gate_ref);
        polynomials[selector_indices[i]].values[row] = F::from_canonical_usize(i);
    }

    (
        polynomials,
        SelectorsInfo {
            selector_indices,
            groups,
        },
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    use crate::gates::arithmetic_base::ArithmeticGate;

    const D: usize = 2;
    type F = GoldilocksField;

    fn legacy_selector_polynomials(
        gates: &[GateRef<F, D>],
        instances: &[GateInstance<F, D>],
        max_degree: usize,
    ) -> (Vec<PolynomialValues<F>>, SelectorsInfo) {
        let n = instances.len();
        let num_gates = gates.len();
        let max_gate_degree = gates.last().expect("No gates?").0.degree();
        let index = |id| gates.iter().position(|g| g.0.id() == id).unwrap();

        if max_gate_degree + num_gates - 1 <= max_degree {
            #[allow(clippy::single_range_in_vec_init)]
            return (
                vec![PolynomialValues::new(
                    instances
                        .iter()
                        .map(|g| F::from_canonical_usize(index(g.gate_ref.0.id())))
                        .collect(),
                )],
                SelectorsInfo {
                    selector_indices: vec![0; num_gates],
                    groups: vec![0..num_gates],
                },
            );
        }

        assert!(max_gate_degree < max_degree);
        let mut groups = Vec::new();
        let mut start = 0;
        while start < num_gates {
            let mut size = 0;
            while (start + size < gates.len())
                && (size + gates[start + size].0.degree() < max_degree)
            {
                size += 1;
            }
            groups.push(start..start + size);
            start += size;
        }
        let group = |i| groups.iter().position(|range| range.contains(&i)).unwrap();
        let selector_indices = (0..num_gates).map(group).collect();
        let unused = F::from_canonical_usize(UNUSED_SELECTOR);
        let mut polynomials = vec![PolynomialValues::zero(n); groups.len()];
        for (row, instance) in instances.iter().enumerate() {
            let gate = index(instance.gate_ref.0.id());
            let active_group = group(gate);
            for group in 0..groups.len() {
                polynomials[group].values[row] = if group == active_group {
                    F::from_canonical_usize(gate)
                } else {
                    unused
                };
            }
        }
        (
            polynomials,
            SelectorsInfo {
                selector_indices,
                groups,
            },
        )
    }

    fn workload_like_instances(
        num_gates: usize,
        rows: usize,
    ) -> (Vec<GateRef<F, D>>, Vec<GateInstance<F, D>>) {
        let mut gates = (1..=num_gates)
            .map(|num_ops| GateRef::new(ArithmeticGate { num_ops }))
            .collect::<Vec<_>>();
        gates.sort_unstable_by_key(|gate| (gate.0.degree(), gate.0.id()));
        let instances = (0..rows)
            .map(|row| GateInstance {
                gate_ref: gates[(row.wrapping_mul(17) + row / 7) % gates.len()].clone(),
                constants: Vec::new(),
            })
            .collect();
        (gates, instances)
    }

    fn assert_matches_legacy(num_gates: usize, rows: usize, max_degree: usize) -> SelectorsInfo {
        let (gates, instances) = workload_like_instances(num_gates, rows);
        let expected = legacy_selector_polynomials(&gates, &instances, max_degree);
        let actual = selector_polynomials(&gates, &instances, max_degree);
        assert_eq!(actual.1, expected.1);
        assert_eq!(actual.0.len(), expected.0.len());
        for (actual, expected) in actual.0.iter().zip(&expected.0) {
            assert_eq!(actual.values, expected.values);
        }
        actual.1
    }

    #[test]
    fn selector_polynomials_indexed_matches_legacy_one_selector() {
        let info = assert_matches_legacy(8, 257, 64);
        assert_eq!(info.groups.len(), 1);
    }

    #[test]
    fn selector_polynomials_indexed_matches_legacy_multiple_groups() {
        let info = assert_matches_legacy(32, 1021, 9);
        assert!(info.groups.len() > 1);
    }
}

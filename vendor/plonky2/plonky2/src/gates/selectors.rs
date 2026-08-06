#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::ops::Range;

use hashbrown::HashMap;
use plonky2_maybe_rayon::*;
use serde::Serialize;

use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialValues;
use crate::gates::gate::{GateInstance, GateRef};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_builder::LookupWire;
use crate::util::SendPtr;

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

    // Map each gate type (unique by id in `gates`, since `CircuitBuilder`
    // stores them in a `HashSet` keyed by id) to its position in the
    // degree-sorted slice. This reproduces the old linear scan `index`
    // exactly, but allocates one `String` per gate type instead of one
    // `String` per gate per row.
    let id_to_index: HashMap<String, usize> = gates
        .iter()
        .enumerate()
        .map(|(i, g)| (g.0.id(), i))
        .collect();
    // The gate-type index of every row, computed once up front so the hot
    // fill loops below never allocate.
    let gate_index_of_row: Vec<usize> = instances
        .iter()
        .map(|inst| id_to_index[&inst.gate_ref.0.id()])
        .collect();

    // Special case if we can use only one selector polynomial.
    if max_gate_degree + num_gates - 1 <= max_degree {
        // We *want* `groups` to be a vector containing one Range (all gates are in one selector group),
        // but Clippy doesn't trust us.
        #[allow(clippy::single_range_in_vec_init)]
        return (
            vec![PolynomialValues::new(
                gate_index_of_row
                    .par_iter()
                    .map(|&i| F::from_canonical_usize(i))
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

    let group = |i| groups.iter().position(|range| range.contains(&i)).unwrap();

    // `selector_indices[i] = j` iff the `i`-th gate uses the `j`-th selector polynomial.
    let selector_indices = (0..num_gates).map(group).collect();

    // Placeholder value to indicate that a gate doesn't use a selector polynomial.
    let unused = F::from_canonical_usize(UNUSED_SELECTOR);
    let num_groups = groups.len();

    // Fill every selector polynomial in one flat, column-major buffer:
    // `flat[g * n + j]` is row `j` of the `g`-th selector polynomial. Rows
    // are processed in parallel; each row writes disjoint addresses, so the
    // result is identical to the serial per-row loop.
    let mut flat = vec![unused; num_groups * n];
    let base = SendPtr(flat.as_mut_ptr());
    gate_index_of_row.par_iter().enumerate().for_each(|(j, &i)| {
        let gr = group(i);
        let val = F::from_canonical_usize(i);
        for g in 0..num_groups {
            // SAFETY: row `j` is handled by a single worker, so each address
            // `g * n + j` is written exactly once, and it is always in
            // `0..num_groups * n`.
            unsafe {
                *base.get().add(g * n + j) = if g == gr { val } else { unused };
            }
        }
    });
    let polynomials = (0..num_groups)
        .map(|g| PolynomialValues::new(flat[g * n..(g + 1) * n].to_vec()))
        .collect();

    (
        polynomials,
        SelectorsInfo {
            selector_indices,
            groups,
        },
    )
}

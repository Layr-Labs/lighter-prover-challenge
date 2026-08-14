//! Fail-closed early IFFT+LDE of final-block wire columns.
//!
//! Isolate marker: colfinal-ifft-1786690000.
//!
//! After the heavy-chain witness feed, any wire column whose every row is
//! already set is transformed (IFFT + coset LDE) on a side thread. At the
//! serial tail, each precomputed column is reused only if its degree-length
//! preimage still matches the finished witness. A later write (light-chain
//! feed, lookup padding) that changes a column forces a recompute. Wrong
//! reuse is therefore either a recompute (slow) or a Merkle-cap / FRI
//! mismatch (closed verify fail) — never a silent wrong proof.

use crate::field::fft::{ifft_borrowed, FftRootTable};
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::witness::PartitionWitness;
use crate::plonk::prover::precomputed;

/// Per-column early transform of the final-block wires commitment.
#[derive(Debug)]
pub struct EarlyWireLde<F: Field> {
    /// Degree-length values that were transformed. `None` = column was not
    /// complete at snapshot time.
    pub preimages: Vec<Option<Vec<F>>>,
    /// IFFT of those values (needed as `PolynomialBatch.polynomials`).
    pub coeffs: Vec<Option<PolynomialCoeffs<F>>>,
    /// Coset LDE (`degree << rate_bits`) of those coefficients.
    pub ldes: Vec<Option<Vec<F>>>,
    pub rate_bits: usize,
}

impl<F: Field> EarlyWireLde<F> {
    pub fn empty(num_wires: usize, rate_bits: usize) -> Self {
        Self {
            preimages: vec![None; num_wires],
            coeffs: vec![None; num_wires],
            ldes: vec![None; num_wires],
            rate_bits,
        }
    }

    pub fn ready_count(&self) -> usize {
        self.ldes.iter().filter(|c| c.is_some()).count()
    }
}

fn gather_complete_column<F: Field>(
    witness: &PartitionWitness<'_, F>,
    column: usize,
) -> Option<Vec<F>> {
    let degree = witness.degree;
    let num_wires = witness.num_wires;
    let mut values = Vec::with_capacity(degree);
    let mut wire_index = column;
    for _row in 0..degree {
        let rep = witness.representative_map[wire_index] as usize;
        if !witness.is_set_by_rep_index(rep) {
            return None;
        }
        values.push(witness.values[rep]);
        wire_index += num_wires;
    }
    Some(values)
}

/// One-column LDE, bit-identical to `PolynomialBatch::lde_values` for a
/// non-ZK column (no salt). Serial: intended for a side thread that must
/// not steal the process-global rayon pool from the light spine.
pub fn lde_one_column<F: Field>(
    coeffs: &PolynomialCoeffs<F>,
    rate_bits: usize,
    fft_root_table: Option<&FftRootTable<F>>,
) -> Vec<F> {
    use crate::field::batch_util::batch_multiply_into;
    use crate::field::fft::fft_in_place_with_options;

    let degree = coeffs.len();
    let lde_len = degree << rate_bits;
    let coset_powers = precomputed::coset_shift_powers::<F>(degree);
    let mut buffer = vec![F::ZERO; lde_len];
    batch_multiply_into(&mut buffer[..degree], &coeffs.coeffs, &coset_powers);
    if rate_bits == 0 || degree < 2 {
        buffer[degree..].fill(F::ZERO);
    }
    fft_in_place_with_options(&mut buffer, Some(rate_bits), fft_root_table);
    buffer
}

/// Snapshot every fully-set wire column and IFFT+LDE it. Incomplete
/// columns are left `None`. Must not run on the process-global rayon
/// pool (serial per column) so it can overlap the light spine.
pub fn precompute_complete_wire_ldes<F: RichField>(
    witness: &PartitionWitness<'_, F>,
    rate_bits: usize,
    fft_root_table: Option<&FftRootTable<F>>,
) -> EarlyWireLde<F> {
    let num_wires = witness.num_wires;
    let mut early = EarlyWireLde::empty(num_wires, rate_bits);
    for column in 0..num_wires {
        let Some(values) = gather_complete_column(witness, column) else {
            continue;
        };
        let coeffs = ifft_borrowed(&values);
        let lde = lde_one_column(&coeffs, rate_bits, fft_root_table);
        early.preimages[column] = Some(values);
        early.coeffs[column] = Some(coeffs);
        early.ldes[column] = Some(lde);
    }
    early
}

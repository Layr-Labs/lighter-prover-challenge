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

/// Snapshot one wire column. Unset representatives read as `F::ZERO`,
/// matching `PartitionWitness::full_witness`. Columns that a later feed
/// actually writes will fail the tail coefficient compare and
/// recompute; columns whose unset cells stay unset reuse this transform.
/// Marker: zero-pad-early-1786704000.
fn gather_column_zero_unset<F: Field>(
    witness: &PartitionWitness<'_, F>,
    column: usize,
) -> Vec<F> {
    let degree = witness.degree;
    let num_wires = witness.num_wires;
    let mut values = Vec::with_capacity(degree);
    let mut wire_index = column;
    for _row in 0..degree {
        let rep = witness.representative_map[wire_index] as usize;
        let value = if witness.is_set_by_rep_index(rep) {
            witness.values[rep]
        } else {
            F::ZERO
        };
        values.push(value);
        wire_index += num_wires;
    }
    values
}

/// One-column LDE. This is a literal extraction of the per-column body of
/// `PolynomialBatch::lde_values` (non-ZK, no salt): extend coeffs, skip the
/// tail memset when `rate_bits > 0`, `batch_multiply_inplace`, then
/// `fft_with_options(Some(rate_bits), table)`. The v1 isolate used
/// `vec![ZERO; lde_len]` + `batch_multiply_into` + `fft_in_place`; that is
/// *not* the production `lde_values` / `fill_lde_column_store` pair the
/// tail hash is compared against, and 8020574 verify-failed fixture 03.
pub fn lde_one_column<F: Field>(
    coeffs: &PolynomialCoeffs<F>,
    rate_bits: usize,
    fft_root_table: Option<&FftRootTable<F>>,
) -> Vec<F> {
    use crate::field::batch_util::batch_multiply_inplace;
    use crate::field::polynomial::PolynomialCoeffs as Coeffs;

    let degree = coeffs.len();
    let lde_len = degree << rate_bits;
    let coset_powers = precomputed::coset_shift_powers::<F>(degree);
    let mut buffer = Vec::with_capacity(lde_len);
    buffer.extend_from_slice(&coeffs.coeffs);
    if rate_bits == 0 || degree < 2 {
        buffer.resize(lde_len, F::ZERO);
    } else {
        // SAFETY: same contract as `PolynomialBatch::lde_values`.
        unsafe { buffer.set_len(lde_len) };
    }
    batch_multiply_inplace(&mut buffer[..degree], &coset_powers);
    Coeffs::new(buffer)
        .fft_with_options(Some(rate_bits), fft_root_table)
        .values
}

/// Snapshot every wire column (unset = ZERO) and IFFT+LDE it. Must not
/// run on the process-global rayon pool (serial per column) so it can
/// overlap the light spine. Later writes fail-close at the tail gate.
pub fn precompute_complete_wire_ldes<F: RichField>(
    witness: &PartitionWitness<'_, F>,
    rate_bits: usize,
    fft_root_table: Option<&FftRootTable<F>>,
) -> EarlyWireLde<F> {
    let num_wires = witness.num_wires;
    let mut early = EarlyWireLde::empty(num_wires, rate_bits);
    for column in 0..num_wires {
        let values = gather_column_zero_unset(witness, column);
        let coeffs = ifft_borrowed(&values);
        let lde = lde_one_column(&coeffs, rate_bits, fft_root_table);
        early.preimages[column] = Some(values);
        early.coeffs[column] = Some(coeffs);
        early.ldes[column] = Some(lde);
    }
    early
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::polynomial::PolynomialValues;
    use crate::field::types::Sample;

    type F = GoldilocksField;

    #[test]
    fn lde_one_column_matches_values_ifft_then_lde_values_body() {
        let degree = 1 << 8;
        let rate_bits = 3;
        let values: Vec<F> = (0..degree).map(|_| F::rand()).collect();
        let coeffs = ifft_borrowed(&values);
        let from_one = lde_one_column(&coeffs, rate_bits, None);
        let via_poly = PolynomialValues::new(values).ifft();
        assert_eq!(coeffs.coeffs, via_poly.coeffs);
        let lde_len = degree << rate_bits;
        assert_eq!(from_one.len(), lde_len);
        let again = lde_one_column(&coeffs, rate_bits, None);
        assert_eq!(from_one, again);
    }
}

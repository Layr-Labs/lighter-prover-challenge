#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::batch_util::batch_multiply_inplace;
use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::FriParams;
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{MerkleLeaves, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Route the whole commitment (NTT + hashing) through the GPU backend.
/// Official ranked A/B: submission 644c4257 (this on, over the 8.0011
/// frontier) scored 6.2323 despite a +4.6% controlled local win — the NTT
/// stages extend each tree's exclusive occupancy of the serialized GPU
/// stream, which is the ranked critical path. Keep off; hashing-only GPU
/// trees (`new_columns`) remain on.
const GPU_NTT_COMMITMENTS: bool = false;

/// Output layout for [`PolynomialBatch::fill_lde_batch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchLayout {
    /// `out[point * width + column]`
    PointMajor,
    /// `out[column * num_points + point]`
    PolyMajor,
}

#[inline]
fn resize_reusable_output<F: Field>(out: &mut Vec<F>, len: usize) {
    // Every caller overwrites the full requested range. Preserve an already
    // initialized prefix so reused scratch buffers do not get zeroed first.
    out.resize(len, F::ZERO);
}

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        if GPU_NTT_COMMITMENTS && !blinding {
            let value_columns: Vec<&[F]> =
                values.iter().map(|v| v.values.as_slice()).collect();
            if let Some((columns, digests, cap, coeff_columns)) = timed!(
                timing,
                "build Merkle tree",
                C::Hasher::try_build_commitment_from_values(
                    &value_columns,
                    rate_bits,
                    cap_height,
                )
            ) {
                let degree = values[0].len();
                let merkle_tree = MerkleTree::from_prebuilt_columns(columns, digests, cap);
                return Self {
                    polynomials: coeff_columns
                        .into_iter()
                        .map(PolynomialCoeffs::new)
                        .collect(),
                    merkle_tree,
                    degree_log: log2_strict(degree),
                    rate_bits,
                    blinding,
                };
            }
        }

        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();

        if GPU_NTT_COMMITMENTS && !blinding {
            let coeff_columns: Vec<&[F]> = polynomials
                .iter()
                .map(|p| p.coeffs.as_slice())
                .collect();
            if let Some((columns, digests, cap)) = timed!(
                timing,
                "build Merkle tree",
                C::Hasher::try_build_commitment_from_coeffs(
                    &coeff_columns,
                    rate_bits,
                    cap_height,
                )
            ) {
                let merkle_tree = MerkleTree::from_prebuilt_columns(columns, digests, cap);
                return Self {
                    polynomials,
                    merkle_tree,
                    degree_log: log2_strict(degree),
                    rate_bits,
                    blinding,
                };
            }
        }

        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_columns(lde_values, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();
        let coset_powers = F::coset_shift().powers().take(degree).collect::<Vec<_>>();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                // Fused zero-pad + shared-coset-powers multiply: one LDE-sized
                // buffer (instead of `lde()` + `coset_fft` each allocating),
                // with the packed batch multiply over the precomputed table.
                let lde_len = degree << rate_bits;
                let mut buffer = Vec::with_capacity(lde_len);
                buffer.extend_from_slice(&p.coeffs);
                buffer.resize(lde_len, F::ZERO);
                batch_multiply_inplace(&mut buffer[..degree], &coset_powers);
                PolynomialCoeffs::new(buffer)
                    .fft_with_options(Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// The number of value columns in this oracle, excluding any salt columns.
    pub(crate) fn lde_row_width(&self) -> usize {
        self.merkle_tree.leaf_width() - if self.blinding { SALT_SIZE } else { 0 }
    }

    /// Fetches LDE values at the `index * step`th point. Only available for
    /// row-major leaf storage; column-major oracles use [`Self::fill_lde_batch`].
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = self.merkle_tree.get(index);
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Gathers LDE values for a batch of points into `out`, in either layout,
    /// for both leaf storage modes. Point `k` (of `indices`) and column `c`
    /// (of `col_range`, indexing as in `get_lde_values(i, step)[c]`) land at
    /// `out[k * col_range.len() + (c - start)]` for `PointMajor` or
    /// `out[(c - start) * indices.len() + k]` for `PolyMajor`.
    pub(crate) fn fill_lde_batch(
        &self,
        indices: &[usize],
        step: usize,
        col_range: core::ops::Range<usize>,
        layout: BatchLayout,
        out: &mut Vec<F>,
    ) {
        let n = indices.len();
        let start = col_range.start;
        let w = col_range.len();
        resize_reusable_output(out, n * w);
        match &self.merkle_tree.leaves {
            MerkleLeaves::Columns { columns, .. } => {
                for (ci, c) in col_range.enumerate() {
                    let column = columns.col(c);
                    match layout {
                        BatchLayout::PolyMajor => {
                            let destination = &mut out[ci * n..(ci + 1) * n];
                            for (k, &i) in indices.iter().enumerate() {
                                destination[k] = column[i * step];
                            }
                        }
                        BatchLayout::PointMajor => {
                            for (k, &i) in indices.iter().enumerate() {
                                out[k * w + ci] = column[i * step];
                            }
                        }
                    }
                }
            }
            MerkleLeaves::Rows { .. } => {
                for (k, &i) in indices.iter().enumerate() {
                    let row = &self.get_lde_values(i, step)[start..start + w];
                    match layout {
                        BatchLayout::PointMajor => {
                            out[k * w..(k + 1) * w].copy_from_slice(row);
                        }
                        BatchLayout::PolyMajor => {
                            for (ci, &value) in row.iter().enumerate() {
                                out[ci * n + k] = value;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let leaf_size = self.lde_row_width();
        if let MerkleLeaves::Columns { columns, .. } = &self.merkle_tree.leaves {
            return (0..leaf_size)
                .map(|j| {
                    let column = columns.col(j);
                    let mut packed = P::ZEROS;
                    packed
                        .as_slice_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(l, packed_l)| *packed_l = column[(index_start + l) * step]);
                    packed
                })
                .collect_vec();
        }

        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            // In-place synthetic division reusing `composition_poly`'s buffer;
            // its final zeroed slot is the power-of-two pad.
            let quotient = composition_poly.divide_by_linear_padded_in_place(*point);
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        // `final_poly` is dead after this point, so pad it in place instead of
        // the clone-then-resize that `lde(&self)` performs.
        let mut lde_final_poly = final_poly;
        lde_final_poly
            .coeffs
            .resize(lde_final_poly.len() << fri_params.config.rate_bits, F::Extension::ZERO);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            // The top (1 - 1/2^rate_bits) of the padded coefficients are zero,
            // so the FFT's zero-run shortcut applies.
            lde_final_poly.coset_fft_with_options(
                F::coset_shift().into(),
                Some(fri_params.config.rate_bits),
                None,
            )
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        );

        fri_proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::hash::merkle_tree::{ColumnStore, MerkleLeaves};
    use crate::plonk::config::Poseidon2GoldilocksConfig;

    const D: usize = 2;
    type F = GoldilocksField;
    type C = Poseidon2GoldilocksConfig;

    #[test]
    fn shared_coset_powers_match_per_polynomial_shifts() {
        const RATE_BITS: usize = 3;

        let polynomials = (0..7)
            .map(|_| PolynomialCoeffs::new(F::rand_vec(1 << 8)))
            .collect::<Vec<_>>();
        let expected = polynomials
            .iter()
            .map(|polynomial| {
                polynomial
                    .lde(RATE_BITS)
                    .coset_fft_with_options(F::coset_shift(), Some(RATE_BITS), None)
                    .values
            })
            .collect::<Vec<_>>();
        let actual = PolynomialBatch::<F, C, D>::lde_values(&polynomials, RATE_BITS, false, None);

        assert_eq!(actual, expected);
    }

    #[test]
    fn reusable_output_resize_preserves_initialized_prefix() {
        let poison = F::NEG_ONE;

        let mut equal = vec![poison; 9];
        resize_reusable_output(&mut equal, 9);
        assert_eq!(equal, vec![poison; 9]);

        let mut oversized = vec![poison; 13];
        resize_reusable_output(&mut oversized, 9);
        assert_eq!(oversized, vec![poison; 9]);

        let mut undersized = vec![poison; 5];
        resize_reusable_output(&mut undersized, 9);
        assert_eq!(&undersized[..5], &[poison; 5]);
        assert_eq!(&undersized[5..], &[F::ZERO; 4]);
    }

    fn batch_test_columns() -> Vec<Vec<F>> {
        (0..5)
            .map(|column| {
                (0..8)
                    .map(|row| F::from_canonical_usize(column * 100 + row + 1))
                    .collect()
            })
            .collect()
    }

    fn batch_test_oracle(columns: &[Vec<F>], column_major: bool) -> PolynomialBatch<F, C, D> {
        let num_rows = columns[0].len();
        let log_rows = log2_strict(num_rows);
        let leaves = if column_major {
            MerkleLeaves::Columns {
                columns: ColumnStore::Owned(columns.to_vec()),
                log_rows,
            }
        } else {
            let mut data = Vec::with_capacity(num_rows * columns.len());
            for tree_row in 0..num_rows {
                let natural_row = reverse_bits(tree_row, log_rows);
                for column in columns {
                    data.push(column[natural_row]);
                }
            }
            MerkleLeaves::Rows {
                data,
                width: columns.len(),
            }
        };
        let mut merkle_tree =
            MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::default();
        merkle_tree.leaves = leaves;
        merkle_tree.num_leaves = num_rows;
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree,
            degree_log: log_rows,
            rate_bits: 0,
            blinding: false,
        }
    }

    fn expected_batch(
        columns: &[Vec<F>],
        indices: &[usize],
        col_range: core::ops::Range<usize>,
        layout: BatchLayout,
    ) -> Vec<F> {
        let width = col_range.len();
        let mut expected = vec![F::ZERO; indices.len() * width];
        for (column_offset, column) in columns[col_range].iter().enumerate() {
            for (point, &index) in indices.iter().enumerate() {
                let destination = match layout {
                    BatchLayout::PointMajor => point * width + column_offset,
                    BatchLayout::PolyMajor => column_offset * indices.len() + point,
                };
                expected[destination] = column[index];
            }
        }
        expected
    }

    #[test]
    fn fill_lde_batch_overwrites_poisoned_reusable_outputs_in_every_layout() {
        let columns = batch_test_columns();
        let indices = [0, 2, 5];
        let col_range = 1..4;
        let expected_len = indices.len() * col_range.len();

        for column_major in [false, true] {
            let oracle = batch_test_oracle(&columns, column_major);
            for layout in [BatchLayout::PointMajor, BatchLayout::PolyMajor] {
                let expected = expected_batch(&columns, &indices, col_range.clone(), layout);
                for initial_len in [expected_len - 4, expected_len, expected_len + 4] {
                    let mut actual = vec![F::NEG_ONE; initial_len];
                    oracle.fill_lde_batch(
                        &indices,
                        1,
                        col_range.clone(),
                        layout,
                        &mut actual,
                    );
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

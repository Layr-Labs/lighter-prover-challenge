#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{MerkleLeaves, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, transpose_to_bitrev_flat};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Output layout for [`PolynomialBatch::fill_lde_batch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchLayout {
    /// `out[point * width + column]`
    PointMajor,
    /// `out[column * num_points + point]`
    PolyMajor,
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

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
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
        out.clear();
        out.resize(n * w, F::ZERO);
        match &self.merkle_tree.leaves {
            MerkleLeaves::Columns { columns, .. } => {
                for (ci, c) in col_range.enumerate() {
                    let column = &columns[c];
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
                    let column = &columns[j];
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
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
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

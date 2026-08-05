#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
use core::slice;

use crate::field::fft::lde_coset_fft;

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
use crate::hash::merkle_tree::{LeafMatrix, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Target working-set size, in bytes, for one block of the fused transpose below. Sized to stay
/// comfortably inside a core's private L2 slice on the Apple Silicon hosts this prover targets.
const TRANSPOSE_BLOCK_BYTES: usize = 128 * 1024;

/// Raw destination pointer shared across the blocks of the fused transpose. Each block writes a
/// disjoint set of whole rows, so the aliasing obligation is discharged by the index algebra in
/// [`transpose_bit_reversed_into_leaves`].
struct RowScatterPtr<T>(*mut T);
// SAFETY: every write through this pointer targets a row index that is unique to the writing
// block, so no two threads ever touch the same element.
unsafe impl<T> Send for RowScatterPtr<T> {}
// SAFETY: see above.
unsafe impl<T> Sync for RowScatterPtr<T> {}

/// Transposes column-major `lde_values` into row-major Merkle leaves that are *already* in
/// bit-reversed row order, i.e. `out[i][j] == lde_values[j][reverse_bits(i)]`.
///
/// This fuses what used to be two separate cache-hostile passes — `transpose` into a
/// `Vec<Vec<F>>` (one heap allocation per LDE point) followed by `reverse_index_bits_in_place` —
/// into a single blocked pass over one contiguous allocation.
///
/// The blocking exploits the fact that bit reversal splits: writing `i = a * 2^(m-k) + b` with `a`
/// the high `k` bits, `reverse_bits_m(i) = reverse_bits_k(a) + reverse_bits_(m-k)(b) * 2^k`. So for
/// a *fixed* `b`, the `2^k` source rows needed form a contiguous run, which keeps the column reads
/// sequential and L2-resident; the corresponding destination rows are strided, and each is written
/// as one contiguous `width`-element run.
fn transpose_bit_reversed_into_leaves<F: Field>(lde_values: &[Vec<F>]) -> LeafMatrix<F> {
    let width = lde_values.len();
    assert!(width > 0, "cannot commit to an empty batch of polynomials");
    let rows = lde_values[0].len();
    debug_assert!(lde_values.iter().all(|column| column.len() == rows));
    let lg_rows = log2_strict(rows);

    let mut data: Vec<F> = Vec::with_capacity(rows * width);
    let base = RowScatterPtr(data.as_mut_ptr());

    // Rows per block, as a power of two, sized so one block's slice of every column fits in L2.
    let lg_block = {
        let target = (TRANSPOSE_BLOCK_BYTES / (width * size_of::<F>())).max(1);
        log2_strict(target.next_power_of_two().min(rows))
    };
    let block_rows = 1usize << lg_block;
    let num_blocks = rows >> lg_block;

    (0..num_blocks).into_par_iter().for_each(|b| {
        // Silence the unused-capture lint while keeping the pointer shared, not copied per row.
        let base = &base;
        let source_base = reverse_bits(b, lg_rows - lg_block) << lg_block;
        for a in 0..block_rows {
            let source_row = source_base + reverse_bits(a, lg_block);
            let destination_row = (a << (lg_rows - lg_block)) + b;
            // SAFETY: `destination_row` is `a * num_blocks + b` with `a < block_rows` and
            // `b < num_blocks`, so it is unique across the whole iteration space and in range.
            // The `width` elements written here are within the allocation reserved above.
            let destination =
                unsafe { slice::from_raw_parts_mut(base.0.add(destination_row * width), width) };
            for (destination, column) in destination.iter_mut().zip(lde_values) {
                *destination = column[source_row];
            }
        }
    });

    // SAFETY: the loop above writes every one of the `rows * width` reserved elements exactly once.
    unsafe { data.set_len(rows * width) };
    LeafMatrix::new(data, width)
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

        let leaves = timed!(
            timing,
            "transpose LDEs",
            transpose_bit_reversed_into_leaves(&lde_values)
        );
        drop(lde_values);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new(leaves, cap_height)
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
                lde_coset_fft(&p.coeffs, F::coset_shift(), rate_bits, fft_root_table)
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    /// Like [`Self::get_lde_values`], but indexes the Merkle leaves directly, i.e. takes the
    /// already bit-reversed row index. Callers that sweep the whole LDE can iterate leaf rows in
    /// storage order and recover the evaluation-point index with `reverse_bits`, which turns a
    /// random gather over the leaf matrix into a sequential scan.
    pub fn get_lde_values_at_leaf(&self, leaf_index: usize) -> &[F] {
        let slice = &self.merkle_tree.leaves[leaf_index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
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

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
        // Resize without clearing. For the usual equal-size reuse this is a
        // no-op; a shorter final batch truncates; growth zero-initializes only
        // the new suffix. Every branch below overwrites the entire `n * w`
        // logical range (and if either dimension is zero the range is empty),
        // so no stale logical element survives. The previous `clear()` forced
        // `resize` to rewrite the whole buffer with zeros before every gather,
        // only for every slot to be immediately overwritten with LDE data.
        out.resize(n * w, F::ZERO);
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
            // Fused (value-exact) form of:
            //   let quotient = composition_poly.divide_by_linear_padded_in_place(*point);
            //   alpha.shift_poly(&mut final_poly);
            //   final_poly += quotient;
            // (where the in-place division runs the classic `divide_by_linear`
            // Horner recurrence and leaves its top slot as the power-of-two
            // pad), writing straight into `final_poly`'s reusable buffer
            // instead of a division pass + shift pass + add pass.
            let shift = alpha.shift_factor();
            accumulate_linear_quotient(&mut final_poly, &composition_poly, *point, shift);
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

/// Folds one batch's quotient `(p(X) - p(z))/(X - z)` into the running FRI
/// `final_poly`, fusing the three passes of the previous code into one serial
/// sweep with no per-batch buffer traffic beyond it:
/// `divide_by_linear_padded_in_place` (the in-place Horner division whose top
/// slot is the power-of-two zero pad), `shift_poly`'s `final_poly *= shift`,
/// and `final_poly += quotient`.
///
/// Value-exact: the division's Horner recurrence (`acc = acc * z + c`)
/// produces the quotient's coefficients highest-first, so a descending sweep
/// can emit each coefficient in the same multiply-add order and combine it
/// with the shifted accumulator entry (`old * shift + q_i`) immediately. The
/// only dropped work is the recurrence's final step (the remainder `p(z)`,
/// which the division discards) and the reference's `+ ZERO` on the pad
/// slot / `ZERO * shift` on fresh slots, all of which leave values unchanged.
fn accumulate_linear_quotient<F: Field>(
    final_poly: &mut PolynomialCoeffs<F>,
    composition_poly: &PolynomialCoeffs<F>,
    z: F,
    shift: F,
) {
    let d = composition_poly.len();
    let coeffs = &composition_poly.coeffs;
    let buf = &mut final_poly.coeffs;
    // Entries past the padded quotient's length only see the shift.
    for l in buf.iter_mut().skip(d) {
        *l *= shift;
    }
    if buf.len() < d {
        buf.resize(d, F::ZERO);
    }
    if d == 0 {
        return;
    }
    // Highest slot: the quotient coefficient there is the explicit zero pad.
    buf[d - 1] *= shift;
    // Synthetic division, highest coefficient first: the quotient's
    // coefficient at `x^i` is the accumulator after absorbing `coeffs[i + 1]`.
    let mut acc = F::ZERO;
    for i in (0..d - 1).rev() {
        acc = acc * z + coeffs[i + 1];
        buf[i] = buf[i] * shift + acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::plonk::config::Poseidon2GoldilocksConfig;

    /// `fill_lde_batch` resizes its reusable output without clearing it, on
    /// the argument that every branch (both storage modes x both layouts)
    /// overwrites the entire logical range. Differential check: for every
    /// combination, filling a poisoned undersized / equal-sized / oversized
    /// starting buffer must produce exactly the same output as filling a
    /// fresh one (which is byte-for-byte the old clear-then-zero-fill
    /// behavior).
    #[test]
    fn fill_lde_batch_poisoned_reuse_matches_fresh() {
        const D: usize = 2;
        type F = GoldilocksField;
        type C = Poseidon2GoldilocksConfig;
        type H = <C as GenericConfig<D>>::Hasher;

        let width = 5usize;
        let n_leaves = 8usize; // log = 3 = degree_log + rate_bits below

        // Rows-mode oracle (flat row-major leaves).
        let rows: Vec<Vec<F>> = (0..n_leaves)
            .map(|i| {
                (0..width)
                    .map(|j| F::from_canonical_u64((i * width + j + 1) as u64))
                    .collect()
            })
            .collect();
        let rows_batch = PolynomialBatch::<F, C, D> {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::<F, H>::new(rows, 0),
            degree_log: 2,
            rate_bits: 1,
            blinding: false,
        };

        // Columns-mode oracle (natural-order poly-major columns).
        let columns: Vec<Vec<F>> = (0..width)
            .map(|j| {
                (0..n_leaves)
                    .map(|i| F::from_canonical_u64((j * n_leaves + i + 100) as u64))
                    .collect()
            })
            .collect();
        let columns_batch = PolynomialBatch::<F, C, D> {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::<F, H>::new_columns(columns, 0),
            degree_log: 2,
            rate_bits: 1,
            blinding: false,
        };

        let indices = [0usize, 2, 3];
        let step = 2usize;
        let poison = F::from_canonical_u64(0xDEAD_BEEF_DEAD_BEEF);

        for batch in [&rows_batch, &columns_batch] {
            for layout in [BatchLayout::PointMajor, BatchLayout::PolyMajor] {
                for col_range in [0..width, 1..4] {
                    // Filling a fresh buffer is exactly the old behavior
                    // (resize from empty zero-fills everything first).
                    let mut expected = Vec::new();
                    batch.fill_lde_batch(
                        &indices,
                        step,
                        col_range.clone(),
                        layout,
                        &mut expected,
                    );
                    let out_len = expected.len();
                    assert_eq!(out_len, indices.len() * col_range.len());

                    for start_len in [out_len.saturating_sub(3), out_len, out_len + 7] {
                        let mut out = vec![poison; start_len];
                        batch.fill_lde_batch(&indices, step, col_range.clone(), layout, &mut out);
                        assert_eq!(out, expected);
                    }
                }
            }
        }

        // An empty column range must leave an empty logical output even when
        // the reused buffer starts non-empty.
        let mut out = vec![poison; 9];
        rows_batch.fill_lde_batch(&indices, step, 2..2, BatchLayout::PointMajor, &mut out);
        assert!(out.is_empty());
    }

    /// Microbench for the gather-buffer zero-fill deletion: clear+resize vs
    /// resize alone, followed by the same full overwrite, on the five
    /// production quotient-scratch buffer lengths (constants 2x32, sigmas
    /// 80x32, wires 136x32, zs local/next 20x32). Run explicitly:
    /// `cargo test --release -- --ignored --nocapture microbench_gather`
    #[test]
    #[ignore = "microbench; run with --ignored --nocapture"]
    fn microbench_gather_resize_with_and_without_clear() {
        use core::hint::black_box;

        type F = GoldilocksField;

        let lengths = [64usize, 2560, 4352, 640, 640];
        let iterations = 250_000usize;
        let trials = 11usize;

        let mut medians = Vec::new();
        for clear_first in [true, false] {
            let mut samples = Vec::new();
            for _ in 0..trials {
                let mut buffers: Vec<Vec<F>> =
                    lengths.iter().map(|&len| vec![F::ZERO; len]).collect();
                let start = std::time::Instant::now();
                for it in 0..iterations {
                    let fill = F::from_canonical_u64(it as u64 + 1);
                    for (buffer, &len) in buffers.iter_mut().zip(&lengths) {
                        if clear_first {
                            buffer.clear();
                        }
                        buffer.resize(len, F::ZERO);
                        // The full overwrite every branch of fill_lde_batch
                        // performs.
                        for slot in buffer.iter_mut() {
                            *slot = fill;
                        }
                    }
                }
                samples.push(start.elapsed().as_secs_f64());
                black_box(&buffers);
            }
            samples.sort_by(f64::total_cmp);
            let median = samples[trials / 2];
            medians.push(median);
            eprintln!(
                "gather fill ({}): median {:.3} ms over {trials} trials",
                if clear_first {
                    "clear+resize"
                } else {
                    "resize only "
                },
                median * 1e3
            );
        }
        eprintln!(
            "gather fill ratio clear/no-clear: {:.4}x",
            medians[0] / medians[1]
        );
    }

    #[test]
    fn shared_coset_powers_match_per_polynomial_shifts() {
        const D: usize = 2;
        const RATE_BITS: usize = 3;
        type F = GoldilocksField;
        type C = Poseidon2GoldilocksConfig;

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

    /// The fused quotient accumulation must be bit-identical (raw u64
    /// representation) to the pre-fusion op sequences it replaces: both the
    /// classic reference (`divide_by_linear` + explicit zero pad +
    /// `shift_poly` + add) and this tree's in-place variant
    /// (`divide_by_linear_padded_in_place` + `shift_poly` + add), including
    /// the empty-accumulator first batch and mismatched lengths.
    #[test]
    fn fused_quotient_accumulation_matches_reference() {
        use crate::field::extension::FieldExtension;
        use crate::field::types::PrimeField64;

        type F = <GoldilocksField as Extendable<2>>::Extension;

        fn raw(values: &[F]) -> Vec<u64> {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
                .map(|c: GoldilocksField| c.to_noncanonical_u64())
                .collect()
        }

        for &(old_len, d) in &[
            (0usize, 1usize),
            (0, 8),
            (1, 1),
            (8, 8),
            (4, 8),
            (8, 4),
            (256, 256),
        ] {
            let initial = PolynomialCoeffs::new(F::rand_vec(old_len));
            let composition_poly = PolynomialCoeffs::new(F::rand_vec(d));
            let z = F::rand();
            let shift = F::rand();

            // Classic reference: the op sequence in `prove_openings` before
            // either in-place rewrite.
            let mut expected = initial.clone();
            let mut quotient = composition_poly.divide_by_linear(z);
            quotient.coeffs.push(F::ZERO); // pad back to power of two
            expected *= shift; // shift_poly
            expected += quotient;

            // This tree's exact pre-fusion sequence: the consuming in-place
            // division (top slot already the pad) + shift_poly + add.
            let mut expected_in_place = initial.clone();
            let quotient_in_place = composition_poly
                .clone()
                .divide_by_linear_padded_in_place(z);
            expected_in_place *= shift; // shift_poly
            expected_in_place += quotient_in_place;

            let mut actual = initial;
            accumulate_linear_quotient(&mut actual, &composition_poly, z, shift);

            assert_eq!(raw(&actual.coeffs), raw(&expected.coeffs));
            assert_eq!(raw(&actual.coeffs), raw(&expected_in_place.coeffs));
        }
    }
}

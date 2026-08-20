#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use anyhow::Result;
use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::{Gate, U32QuotientGate};
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate for checking that a particular element of a list matches a given value.
#[derive(Copy, Clone, Debug, Default)]
pub struct RandomAccessGate<F: RichField + Extendable<D>, const D: usize> {
    /// Number of bits in the index (log2 of the list size).
    pub bits: usize,

    /// How many separate copies are packed into one gate.
    pub num_copies: usize,

    /// Leftover wires are used as global scratch space to store constants.
    pub num_extra_constants: usize,

    _phantom: PhantomData<F>,
}

/// Evaluate one constraint row point-by-point and immediately fold it into
/// the shared filtered accumulator. This mirrors `batch_multiply_add_inplace`
/// exactly: the maximal prefix is reinterpreted as the field's preferred
/// packing and uses `Packing::multiply_accumulate`, while the ragged suffix
/// keeps the scalar `out += term * filter` operation.
///
/// Constraint terms are deliberately computed with scalar `F` operations and
/// only then placed in a single packed register. Computing the expressions in
/// `Packing` directly could produce a different noncanonical representative,
/// even when the field value is equal.
#[inline]
fn accumulate_constraint_direct<F: Field>(
    out: &mut [F],
    filters: &[F],
    mut term_at: impl FnMut(usize) -> F,
) {
    assert_eq!(out.len(), filters.len());

    type Packing<F> = <F as Packable>::Packing;
    let width = Packing::<F>::WIDTH;
    let packed_len = out.len() - out.len() % width;
    let (out_prefix, out_leftovers) = out.split_at_mut(packed_len);
    let (filter_prefix, filter_leftovers) = filters.split_at(packed_len);
    let out_packed = Packing::<F>::pack_slice_mut(out_prefix);
    let filter_packed = Packing::<F>::pack_slice(filter_prefix);

    for (group, (x_out, &x_filter)) in out_packed.iter_mut().zip(filter_packed).enumerate() {
        let mut terms = Packing::<F>::ZEROS;
        let base = group * width;
        for (lane, slot) in terms.as_slice_mut().iter_mut().enumerate() {
            *slot = term_at(base + lane);
        }
        *x_out = x_out.multiply_accumulate(terms, x_filter);
    }

    for (offset, (x_out, &x_filter)) in out_leftovers.iter_mut().zip(filter_leftovers).enumerate() {
        *x_out += term_at(packed_len + offset) * x_filter;
    }
}

/// One level of the selector fold, `out[p] = x[p] + b[p] * (y[p] - x[p])`,
/// evaluated in the field's preferred packing over the maximal prefix.
///
/// The fold is the whole cost of a wide `RandomAccessGate`: bits=6 spends 63 of
/// its 69 multiplies per point here, for a single constraint row. Scalar, it
/// ran at 1.22 ns per base-multiply-equivalent against 0.62-0.66 for every
/// other CPU-resident quotient gate.
///
/// The packed form is raw-limb identical, not merely field-equal.
/// `WideGoldilocksField` is two `NeonGoldilocksField` lane pairs whose add and
/// sub are the scalar per-lane operations verbatim, and whose multiply is
/// `mul_reduce_pair`, documented to compute bit-for-bit the same intermediates
/// -- and therefore the same non-canonical `u64` representative -- as the
/// scalar `reduce128`. The ragged tail keeps the scalar operation, matching
/// `accumulate_constraint_direct`'s own packed/leftover split.
#[inline]
fn fold_level<F: Field>(xs: &[F], ys: &[F], b: &[F], out: &mut [F]) {
    let n = out.len();
    debug_assert_eq!(xs.len(), n);
    debug_assert_eq!(ys.len(), n);
    debug_assert!(b.len() >= n);

    type Packing<F> = <F as Packable>::Packing;
    let width = Packing::<F>::WIDTH;
    let packed_len = n - n % width;
    if packed_len != 0 {
        let out_packed = Packing::<F>::pack_slice_mut(&mut out[..packed_len]);
        let xs_packed = Packing::<F>::pack_slice(&xs[..packed_len]);
        let ys_packed = Packing::<F>::pack_slice(&ys[..packed_len]);
        let b_packed = Packing::<F>::pack_slice(&b[..packed_len]);
        for (((o, &x), &y), &bit) in out_packed
            .iter_mut()
            .zip(xs_packed)
            .zip(ys_packed)
            .zip(b_packed)
        {
            *o = x + bit * (y - x);
        }
    }
    for p in packed_len..n {
        let x = xs[p];
        let y = ys[p];
        out[p] = x + b[p] * (y - x);
    }
}

impl<F: RichField + Extendable<D>, const D: usize> RandomAccessGate<F, D> {
    const fn new(num_copies: usize, bits: usize, num_extra_constants: usize) -> Self {
        Self {
            bits,
            num_copies,
            num_extra_constants,
            _phantom: PhantomData,
        }
    }

    pub fn new_from_config(config: &CircuitConfig, bits: usize) -> Self {
        // We can access a list of 2^bits elements.
        let vec_size = 1 << bits;

        // We need `(2 + vec_size) * num_copies` routed wires.
        let max_copies = (config.num_routed_wires / (2 + vec_size)).min(
            // We need `(2 + vec_size + bits) * num_copies` wires in total.
            config.num_wires / (2 + vec_size + bits),
        );

        // Any leftover wires can be used for constants.
        let max_extra_constants = config.num_routed_wires - (2 + vec_size) * max_copies;

        Self::new(
            max_copies,
            bits,
            max_extra_constants.min(config.num_constants),
        )
    }

    /// Length of the list being accessed.
    const fn vec_size(&self) -> usize {
        1 << self.bits
    }

    /// For each copy, a wire containing the claimed index of the element.
    pub(crate) const fn wire_access_index(&self, copy: usize) -> usize {
        debug_assert!(copy < self.num_copies);
        (2 + self.vec_size()) * copy
    }

    /// For each copy, a wire containing the element claimed to be at the index.
    pub(crate) const fn wire_claimed_element(&self, copy: usize) -> usize {
        debug_assert!(copy < self.num_copies);
        (2 + self.vec_size()) * copy + 1
    }

    /// For each copy, wires containing the entire list.
    pub(crate) const fn wire_list_item(&self, i: usize, copy: usize) -> usize {
        debug_assert!(i < self.vec_size());
        debug_assert!(copy < self.num_copies);
        (2 + self.vec_size()) * copy + 2 + i
    }

    const fn start_extra_constants(&self) -> usize {
        (2 + self.vec_size()) * self.num_copies
    }

    const fn wire_extra_constant(&self, i: usize) -> usize {
        debug_assert!(i < self.num_extra_constants);
        self.start_extra_constants() + i
    }

    /// All above wires are routed.
    pub const fn num_routed_wires(&self) -> usize {
        self.start_extra_constants() + self.num_extra_constants
    }

    /// An intermediate wire where the prover gives the (purported) binary decomposition of the
    /// index.
    pub(crate) const fn wire_bit(&self, i: usize, copy: usize) -> usize {
        debug_assert!(i < self.bits);
        debug_assert!(copy < self.num_copies);
        self.num_routed_wires() + copy * self.bits + i
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for RandomAccessGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.bits)?;
        dst.write_usize(self.num_copies)?;
        dst.write_usize(self.num_extra_constants)?;
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let bits = src.read_usize()?;
        let num_copies = src.read_usize()?;
        let num_extra_constants = src.read_usize()?;
        Ok(Self::new(num_copies, bits, num_extra_constants))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        for copy in 0..self.num_copies {
            let access_index = vars.local_wires[self.wire_access_index(copy)];
            let mut list_items = (0..self.vec_size())
                .map(|i| vars.local_wires[self.wire_list_item(i, copy)])
                .collect::<Vec<_>>();
            let claimed_element = vars.local_wires[self.wire_claimed_element(copy)];
            let bits = (0..self.bits)
                .map(|i| vars.local_wires[self.wire_bit(i, copy)])
                .collect::<Vec<_>>();

            // Assert that each bit wire value is indeed boolean.
            for &b in &bits {
                constraints.push(b * (b - F::Extension::ONE));
            }

            // Assert that the binary decomposition was correct.
            let reconstructed_index = bits
                .iter()
                .rev()
                .fold(F::Extension::ZERO, |acc, &b| acc.double() + b);
            constraints.push(reconstructed_index - access_index);

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.
            for b in bits {
                list_items = list_items
                    .iter()
                    .tuples()
                    .map(|(&x, &y)| x + b * (y - x))
                    .collect()
            }

            debug_assert_eq!(list_items.len(), 1);
            constraints.push(list_items[0] - claimed_element);
        }

        constraints.extend(
            (0..self.num_extra_constants)
                .map(|i| vars.local_constants[i] - vars.local_wires[self.wire_extra_constant(i)]),
        );

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let n = vars_base.len();
        let wires = vars_base.local_wires;
        let constants = vars_base.local_constants;
        let col = |w: usize| &wires[w * n..][..n];
        let vec_size = self.vec_size();
        let mut res = vec![F::ZERO; n * self.num_constraints()];
        let mut chunks = res.chunks_exact_mut(n);
        // `items` holds vec_size columns of n points, folded in place; the
        // write index k always trails the read indices 2k, 2k+1, which were
        // consumed at an earlier k of the same level.
        let mut items = vec![F::ZERO; vec_size * n];
        let mut acc = vec![F::ZERO; n];

        for copy in 0..self.num_copies {
            // Assert that each bit wire value is indeed boolean.
            for i in 0..self.bits {
                let b = col(self.wire_bit(i, copy));
                let out = chunks.next().unwrap();
                for p in 0..n {
                    out[p] = b[p] * (b[p] - F::ONE);
                }
            }

            // Assert that the binary decomposition was correct.
            acc.fill(F::ZERO);
            for i in (0..self.bits).rev() {
                let b = col(self.wire_bit(i, copy));
                for p in 0..n {
                    acc[p] = acc[p].double() + b[p];
                }
            }
            let access_index = col(self.wire_access_index(copy));
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = acc[p] - access_index[p];
            }

            // Repeatedly fold the list, selecting the left or right item from
            // each pair based on the corresponding bit.
            for i in 0..vec_size {
                items[i * n..][..n].copy_from_slice(col(self.wire_list_item(i, copy)));
            }
            let mut level_size = vec_size;
            for i in 0..self.bits {
                let b = col(self.wire_bit(i, copy));
                for k in 0..level_size / 2 {
                    for p in 0..n {
                        let x = items[2 * k * n + p];
                        let y = items[(2 * k + 1) * n + p];
                        items[k * n + p] = x + b[p] * (y - x);
                    }
                }
                level_size /= 2;
            }
            let claimed_element = col(self.wire_claimed_element(copy));
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = items[p] - claimed_element[p];
            }
        }

        for i in 0..self.num_extra_constants {
            let constant = &constants[i * n..][..n];
            let wire = col(self.wire_extra_constant(i));
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = constant[p] - wire[p];
            }
        }
        res
    }

    /// Same contiguous-column evaluation as `eval_unfiltered_base_batch`, but
    /// multiply-adds each filtered constraint row straight into the shared
    /// buffer instead of materializing the full constraint matrix first.
    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_constraints() * n);

        let wires = vars_base.local_wires;
        let constants = vars_base.local_constants;
        let col = |w: usize| &wires[w * n..][..n];
        let vec_size = self.vec_size();
        let mut row = 0;
        // The first selector fold reads the immutable wire columns directly,
        // so only its `vec_size / 2` output columns need scratch storage. The
        // former path zero-filled and copied all `vec_size` input columns here,
        // then immediately consumed and discarded that mirror.
        //
        // The remaining levels alternate between the two halves of this buffer
        // instead of shrinking in place. In-place, the store to column `k` and
        // the loads at columns `2k` and `2k + 1` come from one `&mut [F]`, so
        // the point loop cannot be expressed in the field's packing at all --
        // `pack_slice` and `pack_slice_mut` cannot both borrow it. Splitting
        // the buffer once makes each level's source and destination provably
        // disjoint, which is what lets `fold_level` below run packed.
        //
        // `hi` takes a level's `vec_size / 2` outputs and `lo` the next
        // level's, so the pair is `3 * vec_size / 4` columns wide. Each is
        // exactly the width its producing level fills -- the first level writes
        // `vec_size / 2` columns and the second `vec_size / 4` -- so neither is
        // ever a slice over memory no level has written. Both are empty for
        // `bits == 0`, and `lo` is also empty for `bits == 1`, where the loop
        // that would fill it never runs and nothing reads it.
        let hi_count = (vec_size / 2) * n;
        let lo_count = (vec_size / 4) * n;
        // The ranked offload leaves bits=6 here (48 columns at a 32-point
        // batch), which exceeds any sane stack reservation and took a heap
        // buffer before this change too -- the split costs no extra
        // allocation, only a wider one. bits=4 with four copies, the next
        // shape down, still fits.
        let mut items_stack = [MaybeUninit::<F>::uninit(); 12 * 32];
        let mut items_heap;
        let items_uninit: &mut [MaybeUninit<F>] = if hi_count + lo_count <= items_stack.len() {
            &mut items_stack[..hi_count + lo_count]
        } else {
            items_heap = vec![MaybeUninit::uninit(); hi_count + lo_count];
            &mut items_heap
        };

        for copy in 0..self.num_copies {
            // Assert that each bit wire value is indeed boolean.
            for i in 0..self.bits {
                let b = col(self.wire_bit(i, copy));
                accumulate_constraint_direct(
                    &mut combined_gate_constraints[row * n..][..n],
                    filters,
                    |p| b[p] * (b[p] - F::ONE),
                );
                row += 1;
            }

            // Assert that the binary decomposition was correct.
            let access_index = col(self.wire_access_index(copy));
            accumulate_constraint_direct(
                &mut combined_gate_constraints[row * n..][..n],
                filters,
                |p| {
                    let mut reconstructed_index = F::ZERO;
                    for i in (0..self.bits).rev() {
                        reconstructed_index =
                            reconstructed_index.double() + col(self.wire_bit(i, copy))[p];
                    }
                    reconstructed_index - access_index[p]
                },
            );
            row += 1;

            // Repeatedly fold the list, selecting the left or right item from
            // each pair based on the corresponding bit. Build the first level
            // straight from the wire columns; this performs the same field
            // expression in the same order as the mirror-backed reference.
            //
            // SAFETY: `MaybeUninit<F>` has the same layout and alignment as
            // `F`. Both halves are sized to exactly the number of columns the
            // level that produces them writes, so each is fully initialized
            // before it is first read: the first level writes all of `hi`
            // whenever `bits != 0`, the second writes all of `lo` whenever
            // `bits > 1`, and both are empty in the cases where those levels
            // do not run.
            let (hi_uninit, lo_uninit) = items_uninit.split_at_mut(hi_count);
            let hi = unsafe {
                core::slice::from_raw_parts_mut(hi_uninit.as_mut_ptr().cast::<F>(), hi_count)
            };
            let lo = unsafe {
                core::slice::from_raw_parts_mut(lo_uninit.as_mut_ptr().cast::<F>(), lo_count)
            };
            if self.bits != 0 {
                let b = col(self.wire_bit(0, copy));
                for k in 0..vec_size / 2 {
                    let xs = col(self.wire_list_item(2 * k, copy));
                    let ys = col(self.wire_list_item(2 * k + 1, copy));
                    fold_level(xs, ys, b, &mut hi[k * n..][..n]);
                }
            }
            let mut level_size = vec_size / 2;
            let mut source_is_hi = true;
            for i in 1..self.bits {
                let b = col(self.wire_bit(i, copy));
                let (source, destination): (&[F], &mut [F]) = if source_is_hi {
                    (hi, lo)
                } else {
                    (lo, hi)
                };
                for k in 0..level_size / 2 {
                    // Cut the two source columns out of one slice so the
                    // borrow checker sees them as reads while `destination`
                    // stays a disjoint write.
                    let (head, tail) = source.split_at((2 * k + 1) * n);
                    let xs = &head[(2 * k) * n..][..n];
                    let ys = &tail[..n];
                    fold_level(xs, ys, b, &mut destination[k * n..][..n]);
                }
                level_size /= 2;
                source_is_hi = !source_is_hi;
            }
            let selected: &[F] = if self.bits == 0 {
                col(self.wire_list_item(0, copy))
            } else if source_is_hi {
                &hi[..n]
            } else {
                &lo[..n]
            };
            let claimed_element = col(self.wire_claimed_element(copy));
            accumulate_constraint_direct(
                &mut combined_gate_constraints[row * n..][..n],
                filters,
                |p| selected[p] - claimed_element[p],
            );
            row += 1;
        }

        for i in 0..self.num_extra_constants {
            let constant = &constants[i * n..][..n];
            let wire = col(self.wire_extra_constant(i));
            accumulate_constraint_direct(
                &mut combined_gate_constraints[row * n..][..n],
                filters,
                |p| constant[p] - wire[p],
            );
            row += 1;
        }
        debug_assert_eq!(row, self.num_constraints());
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let zero = builder.zero_extension();
        let two = builder.two_extension();
        let mut constraints = Vec::with_capacity(self.num_constraints());

        for copy in 0..self.num_copies {
            let access_index = vars.local_wires[self.wire_access_index(copy)];
            let mut list_items = (0..self.vec_size())
                .map(|i| vars.local_wires[self.wire_list_item(i, copy)])
                .collect::<Vec<_>>();
            let claimed_element = vars.local_wires[self.wire_claimed_element(copy)];
            let bits = (0..self.bits)
                .map(|i| vars.local_wires[self.wire_bit(i, copy)])
                .collect::<Vec<_>>();

            // Assert that each bit wire value is indeed boolean.
            for &b in &bits {
                constraints.push(builder.mul_sub_extension(b, b, b));
            }

            // Assert that the binary decomposition was correct.
            let reconstructed_index = bits
                .iter()
                .rev()
                .fold(zero, |acc, &b| builder.mul_add_extension(acc, two, b));
            constraints.push(builder.sub_extension(reconstructed_index, access_index));

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.
            for b in bits {
                list_items = list_items
                    .iter()
                    .tuples()
                    .map(|(&x, &y)| builder.select_ext_generalized(b, y, x))
                    .collect()
            }

            // Check that the one remaining element after the folding is the claimed element.
            debug_assert_eq!(list_items.len(), 1);
            constraints.push(builder.sub_extension(list_items[0], claimed_element));
        }

        // Check the constant values.
        constraints.extend((0..self.num_extra_constants).map(|i| {
            builder.sub_extension(
                vars.local_constants[i],
                vars.local_wires[self.wire_extra_constant(i)],
            )
        }));

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_copies)
            .map(|copy| {
                WitnessGeneratorRef::new(
                    RandomAccessGenerator {
                        row,
                        gate: *self,
                        copy,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_routed_wires() + self.num_copies * self.bits
    }

    fn num_constants(&self) -> usize {
        self.num_extra_constants
    }

    fn degree(&self) -> usize {
        self.bits + 1
    }

    fn num_constraints(&self) -> usize {
        let constraints_per_copy = self.bits + 2;
        self.num_copies * constraints_per_copy + self.num_extra_constants
    }

    fn u32_quotient_gate(&self) -> Option<U32QuotientGate> {
        Some(U32QuotientGate::RandomAccess {
            bits: self.bits,
            num_ops: self.num_copies,
            num_extra_constants: self.num_extra_constants,
        })
    }

    fn extra_constant_wires(&self) -> Vec<(usize, usize)> {
        (0..self.num_extra_constants)
            .map(|i| (i, self.wire_extra_constant(i)))
            .collect()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for RandomAccessGate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for copy in 0..self.num_copies {
            let access_index = vars.local_wires[self.wire_access_index(copy)];
            let mut list_items = (0..self.vec_size())
                .map(|i| vars.local_wires[self.wire_list_item(i, copy)])
                .collect::<Vec<_>>();
            let claimed_element = vars.local_wires[self.wire_claimed_element(copy)];
            let bits = (0..self.bits)
                .map(|i| vars.local_wires[self.wire_bit(i, copy)])
                .collect::<Vec<_>>();

            // Assert that each bit wire value is indeed boolean.
            for &b in &bits {
                yield_constr.one(b * (b - F::ONE));
            }

            // Assert that the binary decomposition was correct.
            let reconstructed_index = bits.iter().rev().fold(P::ZEROS, |acc, &b| acc + acc + b);
            yield_constr.one(reconstructed_index - access_index);

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.
            for b in bits {
                list_items = list_items
                    .iter()
                    .tuples()
                    .map(|(&x, &y)| x + b * (y - x))
                    .collect()
            }

            debug_assert_eq!(list_items.len(), 1);
            yield_constr.one(list_items[0] - claimed_element);
        }
        yield_constr.many(
            (0..self.num_extra_constants)
                .map(|i| vars.local_constants[i] - vars.local_wires[self.wire_extra_constant(i)]),
        );
    }
}

#[derive(Debug, Default)]
pub struct RandomAccessGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    gate: RandomAccessGate<F, D>,
    copy: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for RandomAccessGenerator<F, D>
{
    fn id(&self) -> String {
        "RandomAccessGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        let mut deps = vec![local_target(self.gate.wire_access_index(self.copy))];
        for i in 0..self.gate.vec_size() {
            deps.push(local_target(self.gate.wire_list_item(i, self.copy)));
        }
        deps
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let get_local_wire = |column| witness.get_wire(local_wire(column));
        let mut set_local_wire = |column, value| out_buffer.set_wire(local_wire(column), value);

        let copy = self.copy;
        let vec_size = self.gate.vec_size();

        let access_index_f = get_local_wire(self.gate.wire_access_index(copy));
        let access_index = access_index_f.to_canonical_u64() as usize;
        debug_assert!(
            access_index < vec_size,
            "Access index {} is larger than the vector size {}",
            access_index,
            vec_size
        );

        set_local_wire(
            self.gate.wire_claimed_element(copy),
            get_local_wire(self.gate.wire_list_item(access_index, copy)),
        )?;

        for i in 0..self.gate.bits {
            let bit = F::from_bool(((access_index >> i) & 1) != 0);
            set_local_wire(self.gate.wire_bit(i, copy), bit)?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.copy)?;
        self.gate.serialize(dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let copy = src.read_usize()?;
        let gate = RandomAccessGate::<F, D>::deserialize(src, _common_data)?;
        Ok(Self { row, gate, copy })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::field::batch_util::batch_multiply_add_inplace;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field64, PrimeField64, Sample};
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(RandomAccessGate::new(4, 4, 1));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(RandomAccessGate::new(4, 4, 1))
    }

    /// Companion to `CosetInterpolationGate`'s microbenchmark, on the shape the
    /// ranked offload leaves on the CPU (bits=6, one copy, two extra
    /// constants). Both arms run in one process over one set of buffers; the
    /// materialized reference is the control, because no change to the
    /// accumulate path touches it.
    #[test]
    #[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
    fn random_access_accumulate_microbench() {
        use core::time::Duration;
        use std::time::Instant;

        const D: usize = 2;
        type F = GoldilocksField;

        let n = 32;
        let iters = 5_000;
        let gate = RandomAccessGate::<F, D>::new(1, 6, 2);
        let num_wires = <RandomAccessGate<F, D> as Gate<F, D>>::num_wires(&gate);
        let num_constants = <RandomAccessGate<F, D> as Gate<F, D>>::num_constants(&gate);
        let num_constraints = <RandomAccessGate<F, D> as Gate<F, D>>::num_constraints(&gate);

        let wires_batch: Vec<F> = (0..num_wires * n).map(|_| F::rand()).collect();
        let constants_batch: Vec<F> = (0..num_constants.max(1) * n).map(|_| F::rand()).collect();
        let filters: Vec<F> = (0..n).map(|_| F::rand()).collect();
        let public_inputs_hash = HashOut::<F>::ZERO;
        let vars_batch = EvaluationVarsBaseBatch::new(
            n,
            &constants_batch,
            &wires_batch,
            &public_inputs_hash,
        );

        let mut combined = vec![F::ZERO; num_constraints * n];
        let start = Instant::now();
        for _ in 0..iters {
            let res = gate.eval_unfiltered_base_batch(vars_batch);
            for (acc, row) in combined.chunks_exact_mut(n).zip(res.chunks_exact(n)) {
                batch_multiply_add_inplace(acc, row, &filters);
            }
        }
        let reference: Duration = start.elapsed();
        let reference_sink = combined[0];

        let mut combined = vec![F::ZERO; num_constraints * n];
        let start = Instant::now();
        for _ in 0..iters {
            gate.eval_unfiltered_base_batch_accumulate(vars_batch, &filters, &mut combined);
        }
        let fused: Duration = start.elapsed();
        assert_eq!(reference_sink, combined[0], "paths diverged");

        println!(
            "{:>10.3}us/iter -> {:>10.3}us/iter ({:.2}x)  RandomAccessGate(bits=6, copies=1)",
            reference.as_secs_f64() * 1e6 / iters as f64,
            fused.as_secs_f64() * 1e6 / iters as f64,
            reference.as_secs_f64() / fused.as_secs_f64(),
        );
    }

    #[test]
    fn direct_accumulation_matches_materialized_mirror_raw_words() {
        const D: usize = 2;
        type F = GoldilocksField;

        // Include valid noncanonical Goldilocks representatives so this test
        // detects changes hidden by field equality. Small canonical values
        // and ORDER + small represent the same field elements with different
        // raw words.
        fn value(i: usize) -> F {
            let small = ((i as u64).wrapping_mul(0x9e37_79b9) ^ 0x5a5a_a5a5) & 0xffff;
            if i % 3 == 0 {
                GoldilocksField(F::ORDER + small)
            } else {
                F::from_canonical_u64(small)
            }
        }

        // Exercise both sides of every preferred-packing boundary so the
        // direct path must match the reference's packed prefix and scalar
        // leftovers on any target, not just WIDTH=4 AArch64.
        let packing_width = <<F as Packable>::Packing as PackedField>::WIDTH;
        let mut batch_sizes = vec![
            1,
            3,
            5,
            7,
            11,
            31,
            32,
            33,
            packing_width.saturating_sub(1).max(1),
            packing_width,
            packing_width + 1,
            packing_width + 2,
            2 * packing_width - 1,
            2 * packing_width,
            2 * packing_width + 1,
        ];
        // Hit every possible scalar-tail length after one and two complete
        // packed groups (notably remainder 2 when WIDTH=4).
        batch_sizes.extend((0..packing_width).map(|remainder| packing_width + remainder));
        batch_sizes.extend((0..packing_width).map(|remainder| 2 * packing_width + remainder));
        batch_sizes.sort_unstable();
        batch_sizes.dedup();
        for bits in [0, 1, 2, 3, 4, 6] {
            for &n in &batch_sizes {
                let gate = RandomAccessGate::<F, D>::new(3, bits, 2);
                let wires = (0..gate.num_wires() * n)
                    .map(|i| value(i + 1))
                    .collect::<Vec<_>>();
                let constants = (0..gate.num_constants() * n)
                    .map(|i| value(i + 10_001))
                    .collect::<Vec<_>>();
                let filters = (0..n)
                    .map(|i| match i % 7 {
                        0 => F::ZERO,
                        1 => GoldilocksField(F::ORDER), // noncanonical zero
                        _ => value(i + 20_001),
                    })
                    .collect::<Vec<_>>();
                let hash = HashOut::ZERO;
                let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

                // `eval_unfiltered_base_batch` intentionally retains the old
                // zero-fill + full list mirror + in-place fold and is the
                // independent reference for the production accumulate path.
                let materialized = gate.eval_unfiltered_base_batch(vars);
                let initial = (0..gate.num_constraints() * n)
                    .map(|i| match i % 11 {
                        0 => F::ZERO,
                        1 => GoldilocksField(F::ORDER), // noncanonical zero
                        _ => value(i + 30_001),
                    })
                    .collect::<Vec<_>>();
                let mut expected = initial.clone();
                for (acc, constraints) in expected
                    .chunks_exact_mut(n)
                    .zip(materialized.chunks_exact(n))
                {
                    batch_multiply_add_inplace(acc, constraints, &filters);
                }

                let mut actual = initial;
                gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);

                for (i, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
                    assert_eq!(
                        actual.to_noncanonical_u64(),
                        expected.to_noncanonical_u64(),
                        "bits={bits}, n={n}, output={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gate_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        /// Returns the local wires for a random access gate given the vectors, elements to compare,
        /// and indices.
        fn get_wires(
            bits: usize,
            lists: Vec<Vec<F>>,
            access_indices: Vec<usize>,
            claimed_elements: Vec<F>,
            constants: &[F],
        ) -> Vec<FF> {
            let num_copies = lists.len();
            let vec_size = lists[0].len();

            let mut v = Vec::new();
            let mut bit_vals = Vec::new();
            for copy in 0..num_copies {
                let access_index = access_indices[copy];
                v.push(F::from_canonical_usize(access_index));
                v.push(claimed_elements[copy]);
                for j in 0..vec_size {
                    v.push(lists[copy][j]);
                }

                for i in 0..bits {
                    bit_vals.push(F::from_bool(((access_index >> i) & 1) != 0));
                }
            }
            v.extend(constants);
            v.extend(bit_vals);

            v.iter().map(|&x| x.into()).collect()
        }

        let bits = 3;
        let vec_size = 1 << bits;
        let num_copies = 4;
        let lists = (0..num_copies)
            .map(|_| F::rand_vec(vec_size))
            .collect::<Vec<_>>();
        let access_indices = (0..num_copies)
            .map(|_| OsRng.gen_range(0..vec_size))
            .collect::<Vec<_>>();
        let gate = RandomAccessGate::<F, D> {
            bits,
            num_copies,
            num_extra_constants: 1,
            _phantom: PhantomData,
        };
        let constants = F::rand_vec(gate.num_constants());

        let good_claimed_elements = lists
            .iter()
            .zip(&access_indices)
            .map(|(l, &i)| l[i])
            .collect();
        let good_vars = EvaluationVars {
            local_constants: &constants.iter().map(|&x| x.into()).collect::<Vec<_>>(),
            local_wires: &get_wires(
                bits,
                lists.clone(),
                access_indices.clone(),
                good_claimed_elements,
                &constants,
            ),
            public_inputs_hash: &HashOut::rand(),
        };
        let bad_claimed_elements = F::rand_vec(4);
        let bad_vars = EvaluationVars {
            local_constants: &constants.iter().map(|&x| x.into()).collect::<Vec<_>>(),
            local_wires: &get_wires(
                bits,
                lists,
                access_indices,
                bad_claimed_elements,
                &constants,
            ),
            public_inputs_hash: &HashOut::rand(),
        };

        assert!(
            gate.eval_unfiltered(good_vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
        assert!(
            !gate.eval_unfiltered(bad_vars).iter().all(|x| x.is_zero()),
            "Gate constraints are satisfied but should not be."
        );
    }
}

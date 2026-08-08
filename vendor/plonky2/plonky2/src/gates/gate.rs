#[cfg(not(feature = "std"))]
use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::any::Any;
use core::fmt::{Debug, Error, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::Range;
#[cfg(feature = "std")]
use std::sync::Arc;

use hashbrown::HashMap;
use serde::{Serialize, Serializer};

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::selectors::UNUSED_SELECTOR;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::WitnessGeneratorRef;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
};
use crate::util::serialization::{Buffer, IoResult};

/// Static wire-layout metadata for the base-4 range-check gate quotient
/// specialization. This lives in the core gate trait so downstream custom
/// gate crates can opt in without making `plonky2` depend on those crates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeCheckQuotientGate {
    pub num_ops: usize,
    pub bit_size: usize,
}

/// Static wire-layout metadata for gates supported by the optional combined
/// quotient backend. Keeping only layout values here avoids a dependency from
/// `plonky2` back to downstream circuit crates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum U32QuotientGate {
    Arithmetic {
        num_ops: usize,
    },
    /// Borrowing subtraction of two `base_bits`-wide words. The layout is
    /// identical at every width: five routed words per operation followed by
    /// `base_bits / 2` base-4 result limbs.
    Subtraction {
        num_ops: usize,
        base_bits: usize,
    },
    /// Addition of `num_addends` `base_bits`-wide words plus a carry, with
    /// `base_bits / 2` result limbs and `num_carry_limbs` carry limbs.
    AddMany {
        num_ops: usize,
        num_addends: usize,
        base_bits: usize,
        num_carry_limbs: usize,
    },
    /// Byte decomposition: `1 + num_limbs` routed words (sum then bytes)
    /// plus `4 * num_limbs` base-4 aux limbs per operation,
    /// `1 + 5 * num_limbs` constraint rows per operation.
    ByteDecomposition { num_ops: usize, num_limbs: usize },
    /// Degree-5 extension-field multiplication over the base field: fifteen
    /// routed words per operation (five limbs each for the two inputs and
    /// the output), five constraint rows per operation.
    QuinticMultiplication { num_ops: usize },
    /// Degree-5 extension-field squaring over the base field: ten routed
    /// words (input and output limbs) plus ten temporary wires per
    /// operation, fifteen constraint rows per operation.
    QuinticSquaring { num_ops: usize },
    /// Random access with a little-endian binary index, `2^bits` list items
    /// per copy, and optional routed local constants.
    RandomAccess {
        bits: usize,
        num_ops: usize,
        num_extra_constants: usize,
    },
}

/// A custom gate.
///
/// Vanilla Plonk arithmetization only supports basic fan-in 2 / fan-out 1 arithmetic gates,
/// each of the form
///
/// $$ a.b \cdot q_M + a \cdot q_L + b \cdot q_R + c \cdot q_O + q_C = 0 $$
///
/// where:
/// - $q_M$, $q_L$, $q_R$ and $q_O$ are boolean selectors,
/// - $a$, $b$ and $c$ are values used as inputs and output respectively,
/// - $q_C$ is a constant (possibly 0).
///
/// This allows expressing simple operations like multiplication, addition, etc. For
/// instance, to define a multiplication, one can set $q_M=1$, $q_L=q_R=0$, $q_O = -1$ and $q_C = 0$.
///
/// Hence, the gate equation simplifies to $a.b - c = 0$, or equivalently to $a.b = c$.
///
/// However, such a gate is fairly limited for more complex computations. Hence, when a computation may
/// require too many of these "vanilla" gates, or when a computation arises often within the same circuit,
/// one may want to construct a tailored custom gate. These custom gates can use more selectors and are
/// not necessarily limited to 2 inputs + 1 output = 3 wires.
/// For instance, plonky2 supports natively a custom Poseidon hash gate that uses 135 wires.
///
/// Note however that extending the number of wires necessary for a custom gate comes at a price, and may
/// impact the overall performances when generating proofs for a circuit containing them.
pub trait Gate<F: RichField + Extendable<D>, const D: usize>: 'static + Send + Sync {
    /// Defines a unique identifier for this custom gate.
    ///
    /// This is used as differentiating tag in gate serializers.
    fn id(&self) -> String;

    /// Serializes this custom gate to the targeted byte buffer, with the provided [`CommonCircuitData`].
    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    /// Deserializes the bytes in the provided buffer into this custom gate, given some [`CommonCircuitData`].
    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;

    /// Defines and evaluates the constraints that enforce the statement represented by this gate.
    /// Constraints must be defined in the extension of this custom gate base field.
    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension>;

    /// Like `eval_unfiltered`, but specialized for points in the base field.
    ///
    ///
    /// `eval_unfiltered_base_batch` calls this method by default. If `eval_unfiltered_base_batch`
    /// is overridden, then `eval_unfiltered_base_one` is not necessary.
    ///
    /// By default, this just calls `eval_unfiltered`, which treats the point as an extension field
    /// element. This isn't very efficient.
    fn eval_unfiltered_base_one(
        &self,
        vars_base: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Note that this method uses `yield_constr` instead of returning its constraints.
        // `yield_constr` abstracts out the underlying memory layout.
        let local_constants = &vars_base
            .local_constants
            .iter()
            .map(|c| F::Extension::from_basefield(*c))
            .collect::<Vec<_>>();
        let local_wires = &vars_base
            .local_wires
            .iter()
            .map(|w| F::Extension::from_basefield(*w))
            .collect::<Vec<_>>();
        let public_inputs_hash = &vars_base.public_inputs_hash;
        let vars = EvaluationVars {
            local_constants,
            local_wires,
            public_inputs_hash,
        };
        let values = self.eval_unfiltered(vars);

        // Each value should be in the base field, i.e. only the degree-zero part should be nonzero.
        values.into_iter().for_each(|value| {
            debug_assert!(F::Extension::is_in_basefield(&value));
            yield_constr.one(value.to_basefield_array()[0])
        })
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let mut res = vec![F::ZERO; vars_base.len() * self.num_constraints()];
        for (i, vars_base_one) in vars_base.iter().enumerate() {
            self.eval_unfiltered_base_one(
                vars_base_one,
                StridedConstraintConsumer::new(&mut res, vars_base.len(), i),
            );
        }
        res
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let batch_size = vars_base.len();
        assert_eq!(filters.len(), batch_size);
        let res_batch = self.eval_unfiltered_base_batch(vars_base);
        for (combined, res) in combined_gate_constraints
            .chunks_exact_mut(batch_size)
            .zip(res_batch.chunks_exact(batch_size))
        {
            batch_multiply_add_inplace(combined, res, filters);
        }
    }

    /// Defines the recursive constraints that enforce the statement represented by this custom gate.
    /// This is necessary to recursively verify proofs generated from a circuit containing such gates.
    ///
    /// **Note**: The order of the recursive constraints output by this method should match exactly the order
    /// of the constraints obtained by the non-recursive [`Gate::eval_unfiltered`] method, otherwise the
    /// prover won't be able to generate proofs.
    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>>;

    fn eval_filtered(
        &self,
        mut vars: EvaluationVars<F, D>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
    ) -> Vec<F::Extension> {
        let filter = compute_filter(
            row,
            group_range,
            vars.local_constants[selector_index],
            num_selectors > 1,
        );
        vars.remove_prefix(num_selectors);
        vars.remove_prefix(num_lookup_selectors);
        self.eval_unfiltered(vars)
            .into_iter()
            .map(|c| filter * c)
            .collect()
    }

    /// Adds this gate's filtered base-field constraints directly to the shared constraint buffer.
    ///
    /// Constraint `j` for point `i` is at index `j * batch_size + i`.
    fn eval_filtered_base_batch(
        &self,
        mut vars_batch: EvaluationVarsBaseBatch<F>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
        filters: &mut Vec<F>,
        combined_gate_constraints: &mut [F],
    ) {
        let batch_size = vars_batch.len();
        debug_assert!(self.num_constraints() * batch_size <= combined_gate_constraints.len());
        let selector_col = &vars_batch.local_constants[selector_index * batch_size..][..batch_size];
        let mut factors = group_range
            .filter(|&i| i != row)
            .chain((num_selectors > 1).then_some(UNUSED_SELECTOR));
        filters.clear();
        if let Some(i) = factors.next() {
            let factor = F::from_canonical_usize(i);
            filters.extend(selector_col.iter().map(|&s| factor - s));
        } else {
            filters.resize(batch_size, F::ONE);
        }
        for i in factors {
            let factor = F::from_canonical_usize(i);
            for (filter, &s) in filters.iter_mut().zip(selector_col) {
                *filter *= factor - s;
            }
        }
        vars_batch.remove_prefix(num_selectors + num_lookup_selectors);
        self.eval_unfiltered_base_batch_accumulate(vars_batch, filters, combined_gate_constraints);
    }

    /// Internal batch-filter entry point with a shared left prefix.
    ///
    /// `filter_prefix` is empty for the first gate in a selector group. For a
    /// later gate at row `r`, it contains exactly the left-to-right product of
    /// `(k - selector)` for every group row `k < r`, with one value per batch
    /// point. The default implementation appends rows after `r` and the
    /// optional unused-selector factor in the scalar filter's original order.
    #[doc(hidden)]
    fn eval_filtered_base_batch_with_prefix(
        &self,
        mut vars_batch: EvaluationVarsBaseBatch<F>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
        filter_prefix: &[F],
        filters: &mut Vec<F>,
        combined_gate_constraints: &mut [F],
    ) {
        let batch_size = vars_batch.len();
        debug_assert!(self.num_constraints() * batch_size <= combined_gate_constraints.len());
        // Continue after the caller's shared left prefix, accumulating the
        // remaining product terms in the same order as `compute_filter`.
        // This retains identical raw field values while avoiding repeated
        // prefix work across gates in the same selector group.
        let selector_col = &vars_batch.local_constants[selector_index * batch_size..][..batch_size];
        compute_filter_base_batch_from_prefix(
            row,
            group_range,
            selector_col,
            num_selectors > 1,
            filter_prefix,
            filters,
        );
        vars_batch.remove_prefix(num_selectors + num_lookup_selectors);
        self.eval_unfiltered_base_batch_accumulate(vars_batch, filters, combined_gate_constraints);
    }

    /// Adds this gate's filtered constraints into the `combined_gate_constraints` buffer.
    fn eval_filtered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        mut vars: EvaluationTargets<D>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
        combined_gate_constraints: &mut [ExtensionTarget<D>],
    ) {
        let filter = compute_filter_circuit(
            builder,
            row,
            group_range,
            vars.local_constants[selector_index],
            num_selectors > 1,
        );
        vars.remove_prefix(num_selectors);
        vars.remove_prefix(num_lookup_selectors);
        let my_constraints = self.eval_unfiltered_circuit(builder, vars);
        for (acc, c) in combined_gate_constraints.iter_mut().zip(my_constraints) {
            *acc = builder.mul_add_extension(filter, c, *acc);
        }
    }

    /// The generators used to populate the witness.
    ///
    /// **Note**: This should return exactly 1 generator per operation in the gate.
    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>>;

    /// The number of wires used by this gate.
    ///
    /// While vanilla Plonk can only evaluate one addition/multiplication at a time, a wider
    /// configuration may be able to accommodate several identical gates at once. This is
    /// particularly helpful for tiny custom gates that are being used extensively in circuits.
    ///
    /// For instance, the [crate::gates::multiplication_extension::MulExtensionGate] takes `3*D`
    /// wires per multiplication (where `D`` is the degree of the extension), hence for a usual
    /// configuration of 80 routed wires with D=2, one can evaluate 13 multiplications within a
    /// single gate.
    fn num_wires(&self) -> usize;

    /// The number of constants used by this gate.
    fn num_constants(&self) -> usize;

    /// The maximum degree among this gate's constraint polynomials.
    fn degree(&self) -> usize;

    /// The number of constraints defined by this sole custom gate.
    fn num_constraints(&self) -> usize;

    /// Number of operations performed by the gate.
    fn num_ops(&self) -> usize {
        self.generators(0, &vec![F::ZERO; self.num_constants()])
            .len()
    }

    /// Advertises the exact base-4 range-check layout to optional quotient
    /// backends. The default keeps every other gate backend-agnostic.
    fn range_check_quotient_gate(&self) -> Option<RangeCheckQuotientGate> {
        None
    }

    /// Advertises one of the exact promoted gate layouts to optional quotient
    /// backends. The default leaves unrelated gates on the CPU.
    fn u32_quotient_gate(&self) -> Option<U32QuotientGate> {
        None
    }

    /// Enables gates to store some "routed constants", if they have both unused constants and
    /// unused routed wires.
    ///
    /// Each entry in the returned `Vec` has the form `(constant_index, wire_index)`. `wire_index`
    /// must correspond to a *routed* wire.
    fn extra_constant_wires(&self) -> Vec<(usize, usize)> {
        vec![]
    }

    // In the case of multiple operations that use non-trivial generators
    // the user must provide defaults to the input wires
    fn input_wires_defaults(&self, _index: usize) -> Vec<(usize, F)> {
        vec![]
    }
}

/// A wrapper trait over a `Gate`, to allow for gate serialization.
pub trait AnyGate<F: RichField + Extendable<D>, const D: usize>: Gate<F, D> {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Gate<F, D>, F: RichField + Extendable<D>, const D: usize> AnyGate<F, D> for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A wrapper around an `Arc<AnyGate>` which implements `PartialEq`, `Eq` and `Hash` based on gate IDs.
#[derive(Clone)]
pub struct GateRef<F: RichField + Extendable<D>, const D: usize>(pub Arc<dyn AnyGate<F, D>>);

impl<F: RichField + Extendable<D>, const D: usize> GateRef<F, D> {
    pub fn new<G: Gate<F, D>>(gate: G) -> GateRef<F, D> {
        GateRef(Arc::new(gate))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PartialEq for GateRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Hash for GateRef<F, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Eq for GateRef<F, D> {}

impl<F: RichField + Extendable<D>, const D: usize> Debug for GateRef<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.0.id())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Serialize for GateRef<F, D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.id())
    }
}

/// Map between gate parameters and available slots.
/// An available slot is of the form `(row, op)`, meaning the current available slot
/// is at gate index `row` in the `op`-th operation.
#[derive(Clone, Debug, Default)]
pub struct CurrentSlot<F: RichField + Extendable<D>, const D: usize> {
    pub current_slot: HashMap<Vec<F>, (usize, usize)>,
    /// Memoized [`Gate::num_ops`] for the gate this entry is keyed by.
    ///
    /// The default `Gate::num_ops` *materializes* the gate's generator list and returns its
    /// length, so every call allocates one `WitnessGeneratorRef` (an `Arc`) per operation plus
    /// the `Vec` holding them, and drops all of it immediately. `find_slot` calls it once per
    /// packed operation, which is where the overwhelming majority of the circuit's operations
    /// are placed. Caching it here evaluates it once per distinct gate value instead.
    ///
    /// Keying on the entry is exact: `CurrentSlot` entries are keyed by `GateRef`, whose `Eq`
    /// is `Gate::id()` equality, and every gate's `id()` is a `Debug` rendering of its complete
    /// configuration. Gates that compare equal therefore have identical fields, and `num_ops`
    /// is a pure function of those fields.
    pub num_ops: Option<usize>,
}

/// A gate along with any constants used to configure it.
#[derive(Clone, Debug)]
pub struct GateInstance<F: RichField + Extendable<D>, const D: usize> {
    pub gate_ref: GateRef<F, D>,
    pub constants: Vec<F>,
}

/// Map each gate to a boolean prefix used to construct the gate's selector polynomial.
#[derive(Debug, Clone)]
pub struct PrefixedGate<F: RichField + Extendable<D>, const D: usize> {
    pub gate: GateRef<F, D>,
    pub prefix: Vec<bool>,
}

/// A gate's filter designed so that it is non-zero if `s = row`.
fn compute_filter<K: Field>(row: usize, group_range: Range<usize>, s: K, many_selector: bool) -> K {
    debug_assert!(group_range.contains(&row));
    group_range
        .filter(|&i| i != row)
        .chain(many_selector.then_some(UNUSED_SELECTOR))
        .map(|i| K::from_canonical_usize(i) - s)
        .product()
}

/// Reusable state for selector-filter prefixes within one base-field batch.
#[derive(Debug, Default)]
pub(crate) struct SelectorFilterPrefix<F> {
    selector_index: Option<usize>,
    prefix_end: usize,
    values: Vec<F>,
}

impl<F: Field> SelectorFilterPrefix<F> {
    pub(crate) fn reset(&mut self) {
        self.selector_index = None;
        self.prefix_end = 0;
        self.values.clear();
    }

    pub(crate) fn prepare(
        &mut self,
        selector_index: usize,
        group_range: Range<usize>,
        row: usize,
        selector_col: &[F],
    ) -> &[F] {
        debug_assert!(group_range.contains(&row));
        if self.selector_index != Some(selector_index) || row < self.prefix_end {
            self.values.clear();
            self.selector_index = Some(selector_index);
            self.prefix_end = group_range.start;
        }
        debug_assert!(
            (self.prefix_end == group_range.start && self.values.is_empty())
                || (self.prefix_end > group_range.start && self.values.len() == selector_col.len())
        );
        while self.prefix_end < row {
            extend_filter_prefix(selector_col, self.prefix_end, &mut self.values);
            self.prefix_end += 1;
        }
        &self.values
    }
}

/// Extends a batch of selector-filter prefixes by one factor, retaining the
/// scalar filter's exact left-to-right multiplication order.
fn extend_filter_prefix<F: Field>(selector_col: &[F], factor_row: usize, prefix: &mut Vec<F>) {
    let factor = F::from_canonical_usize(factor_row);
    if prefix.is_empty() {
        prefix.extend(selector_col.iter().map(|&s| factor - s));
    } else {
        debug_assert_eq!(prefix.len(), selector_col.len());
        for (value, &s) in prefix.iter_mut().zip(selector_col) {
            *value *= factor - s;
        }
    }
}

fn compute_filter_base_batch_from_prefix<F: Field>(
    row: usize,
    group_range: Range<usize>,
    selector_col: &[F],
    many_selectors: bool,
    prefix: &[F],
    filters: &mut Vec<F>,
) {
    debug_assert!(group_range.contains(&row));
    debug_assert!(prefix.is_empty() || prefix.len() == selector_col.len());
    filters.clear();
    if selector_col.is_empty() {
        return;
    }
    filters.extend_from_slice(prefix);
    let mut factors = (row + 1..group_range.end).chain(many_selectors.then_some(UNUSED_SELECTOR));
    if filters.is_empty() {
        if let Some(i) = factors.next() {
            let factor = F::from_canonical_usize(i);
            filters.extend(selector_col.iter().map(|&s| factor - s));
        } else {
            filters.resize(selector_col.len(), F::ONE);
        }
    }
    for i in factors {
        let factor = F::from_canonical_usize(i);
        for (filter, &s) in filters.iter_mut().zip(selector_col) {
            *filter *= factor - s;
        }
    }
}

fn compute_filter_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    row: usize,
    group_range: Range<usize>,
    s: ExtensionTarget<D>,
    many_selectors: bool,
) -> ExtensionTarget<D> {
    debug_assert!(group_range.contains(&row));
    let v = group_range
        .filter(|&i| i != row)
        .chain(many_selectors.then_some(UNUSED_SELECTOR))
        .map(|i| {
            let c = builder.constant_extension(F::Extension::from_canonical_usize(i));
            builder.sub_extension(c, s)
        })
        .collect::<Vec<_>>();
    builder.mul_many_extension(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field64;

    type F = GoldilocksField;

    #[test]
    fn shared_filter_prefix_matches_scalar_raw_words() {
        let mut selector_col = vec![
            GoldilocksField(0),
            GoldilocksField(1),
            GoldilocksField(F::ORDER - 1),
            GoldilocksField(F::ORDER),
            GoldilocksField(F::ORDER + 1),
            GoldilocksField(u32::MAX as u64),
            GoldilocksField(1 << 32),
            GoldilocksField(u64::MAX),
        ];
        let mut state = 0x5345_4c45_4354_4f52u64;
        for _ in 0..128 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            selector_col.push(GoldilocksField(state));
        }

        for group in [0..1, 2..5, 3..9] {
            for many_selectors in [false, true] {
                let mut prefix = Vec::new();
                let mut filters = Vec::new();
                let mut prefix_end = group.start;
                for row in group.clone() {
                    while prefix_end < row {
                        extend_filter_prefix(&selector_col, prefix_end, &mut prefix);
                        prefix_end += 1;
                    }
                    compute_filter_base_batch_from_prefix(
                        row,
                        group.clone(),
                        &selector_col,
                        many_selectors,
                        &prefix,
                        &mut filters,
                    );
                    let expected = selector_col
                        .iter()
                        .map(|&s| compute_filter(row, group.clone(), s, many_selectors).0)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        filters.iter().map(|value| value.0).collect::<Vec<_>>(),
                        expected,
                        "group={group:?}, row={row}, many_selectors={many_selectors}",
                    );
                }
            }
        }
    }

    #[test]
    fn filter_prefix_state_handles_sparse_groups_and_rewinds() {
        let selector_col = [
            GoldilocksField(0),
            GoldilocksField(1),
            GoldilocksField(F::ORDER + 1),
            GoldilocksField(u64::MAX),
        ];
        let cases = [
            (0, 2..7, 2),
            (0, 2..7, 5),
            (1, 9..12, 11),
            (1, 9..12, 9),
            (0, 2..7, 4),
        ];
        let mut state = SelectorFilterPrefix::default();
        let mut filters = Vec::new();
        for (selector_index, group, row) in cases {
            let prefix = state.prepare(selector_index, group.clone(), row, &selector_col);
            compute_filter_base_batch_from_prefix(
                row,
                group.clone(),
                &selector_col,
                true,
                prefix,
                &mut filters,
            );
            let expected = selector_col
                .iter()
                .map(|&s| compute_filter(row, group.clone(), s, true).0)
                .collect::<Vec<_>>();
            assert_eq!(
                filters.iter().map(|value| value.0).collect::<Vec<_>>(),
                expected,
                "selector_index={selector_index}, group={group:?}, row={row}",
            );
        }
    }
}

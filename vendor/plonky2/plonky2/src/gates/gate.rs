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

    /// Whether this gate honours the `store_from` argument of
    /// [`Self::eval_unfiltered_base_batch_accumulate_store`].
    ///
    /// Gates opt in one at a time; the caller zero-fills the constraint rows
    /// first written by any gate that has not, so correctness never depends on
    /// how many gates have been converted.
    fn supports_store_from(&self) -> bool {
        false
    }

    /// Like [`Self::eval_unfiltered_base_batch_accumulate`], but the caller
    /// guarantees that constraint rows `store_from..num_constraints()` of
    /// `combined_gate_constraints` are *raw* `F::ZERO` — this gate is the first
    /// to write them for this batch — so those rows may be **stored** rather
    /// than accumulated into.
    ///
    /// Storing is bit-identical to accumulating into a raw zero, not merely
    /// field-value identical, which is what lets the caller drop the zero-fill
    /// entirely once every first-writing gate has opted in. Both
    /// `batch_multiply_into` and `batch_multiply_add_inplace` use the same
    /// packed/leftover split, and on each half:
    ///   - packed: `x_out.multiply_accumulate(a, b)` is, per lane,
    ///     `reduce128((0 as u128) + a * b)`, and the store is `a * b`, i.e.
    ///     `reduce128(a * b)` — the same bits;
    ///   - leftover: the accumulate is `F(0) + a * b`, and `Add` with a zero
    ///     left operand takes neither overflow branch and returns its right
    ///     operand unchanged.
    /// This is the same argument `reduce_gate_constraints_base_batch` already
    /// uses for its `res_out_is_zero_seed` fast path.
    ///
    /// The default implementation ignores `store_from` and accumulates, which
    /// is always correct: it just requires the caller to have zeroed the rows.
    fn eval_unfiltered_base_batch_accumulate_store(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
        _store_from: usize,
    ) {
        self.eval_unfiltered_base_batch_accumulate(vars_base, filters, combined_gate_constraints)
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
    ///
    /// `cached_filter`, when present, is this gate's already-computed filter
    /// values for exactly this batch of points, produced by
    /// [`fill_filter_column`] — the same function this method would otherwise
    /// call — so it is raw-limb identical to recomputing them here.
    #[allow(clippy::too_many_arguments)]
    fn eval_filtered_base_batch(
        &self,
        mut vars_batch: EvaluationVarsBaseBatch<F>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
        cached_filter: Option<&[F]>,
        filters: &mut Vec<F>,
        combined_gate_constraints: &mut [F],
        store_from: usize,
    ) {
        let batch_size = vars_batch.len();
        debug_assert!(self.num_constraints() * batch_size <= combined_gate_constraints.len());
        if let Some(cached) = cached_filter {
            debug_assert_eq!(cached.len(), batch_size);
            vars_batch.remove_prefix(num_selectors + num_lookup_selectors);
            // MUST honour `store_from` exactly as the uncached path below does.
            // Store-on-first-write deliberately leaves rows
            // `[store_from, num_constraints)` UN-zeroed, on the promise that
            // their first writer STORES rather than accumulates. Calling the
            // plain accumulate here breaks that contract and accumulates into
            // uninitialized memory. Both mechanisms' own differentials pass in
            // isolation; the bug appears only when they are composed, and it
            // surfaced as a SIGABRT in the end-to-end run, not in 182 unit
            // tests.
            self.eval_unfiltered_base_batch_accumulate_store(
                vars_batch,
                cached,
                combined_gate_constraints,
                store_from,
            );
            return;
        }
        // Contiguous-column filter computation: read the selector constant
        // column once and accumulate the same product terms, in the same
        // order, as the per-point `compute_filter` — identical field values
        // without the per-point strided views.
        let selector_col = &vars_batch.local_constants[selector_index * batch_size..][..batch_size];
        fill_filter_column(row, group_range, num_selectors, selector_col, filters);
        vars_batch.remove_prefix(num_selectors + num_lookup_selectors);
        self.eval_unfiltered_base_batch_accumulate_store(
            vars_batch,
            filters,
            combined_gate_constraints,
            store_from,
        );
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
pub(crate) fn compute_filter<K: Field>(
    row: usize,
    group_range: Range<usize>,
    s: K,
    many_selector: bool,
) -> K {
    debug_assert!(group_range.contains(&row));
    group_range
        .filter(|&i| i != row)
        .chain(many_selector.then_some(UNUSED_SELECTOR))
        .map(|i| K::from_canonical_usize(i) - s)
        .product()
}

/// Writes the filter values of the gate at `row` for a whole run of points into
/// `filters`, given that run's values of the gate's selector constant column.
///
/// This is the batched form of [`compute_filter`], factored out so that the
/// per-batch evaluator and the per-circuit column cache below run *the same
/// code*: the factors are consumed in the same order and each output element is
/// an independent product chain over its own `s`, so the values produced are
/// raw-limb identical no matter how the domain is cut into runs. (Goldilocks
/// addition is the lazy variant and `reduce128` does not fully canonicalize, so
/// "same value" would not be enough here — the operation order has to match.)
pub(crate) fn fill_filter_column<F: Field>(
    row: usize,
    group_range: Range<usize>,
    num_selectors: usize,
    selector_col: &[F],
    filters: &mut Vec<F>,
) {
    debug_assert!(group_range.contains(&row));
    let mut factors = group_range
        .filter(|&i| i != row)
        .chain((num_selectors > 1).then_some(UNUSED_SELECTOR));
    filters.clear();
    if let Some(i) = factors.next() {
        let k = F::from_canonical_usize(i);
        filters.extend(selector_col.iter().map(|&s| k - s));
    } else {
        filters.resize(selector_col.len(), F::ONE);
    }
    for i in factors {
        let k = F::from_canonical_usize(i);
        for (filter, &s) in filters.iter_mut().zip(selector_col) {
            *filter *= k - s;
        }
    }
}

/// True when the gate at `row` has no filter factors at all, i.e. its filter is
/// the constant `F::ONE` over the whole domain. Such a gate is never worth
/// caching: `fill_filter_column` would store a domain-sized run of ones.
pub(crate) fn filter_is_trivial(row: usize, group_range: Range<usize>, num_selectors: usize) -> bool {
    num_selectors <= 1 && !group_range.into_iter().any(|i| i != row)
}

/// Per-circuit cache of the gate filter columns over the quotient domain.
///
/// Every input to a gate's filter is circuit-fixed: the selector constant
/// column is a column of `constants_sigmas_commitment`, and `row`,
/// `group_range` and `num_selectors` are circuit constants. So the whole filter
/// column is a deterministic function of the circuit alone, identical in every
/// proof of that circuit — yet it is recomputed from scratch on every proof.
/// Caching it here converts that per-proof work into per-circuit work.
///
/// The cache hangs off `ProverOnlyCircuitData` rather than a keyed map so there
/// is no key to get wrong: a cache reached through a circuit's own prover data
/// can only belong to that circuit.
///
/// Storage is opt-in per circuit ([`Self::set_enabled`]) because a column costs
/// `8 * quotient_domain_size` bytes per non-trivial gate, which is worth paying
/// only for a circuit that is proved many times in one process.
///
/// Entry `i` corresponds to `common_data.gates[i]`; an *empty* entry means "not
/// cached, compute this gate's filters per batch as before".
pub struct GateFilterCache<F> {
    enabled: bool,
    #[cfg(feature = "std")]
    columns: std::sync::OnceLock<Vec<Vec<F>>>,
    #[cfg(not(feature = "std"))]
    _marker: core::marker::PhantomData<F>,
}

impl<F> Default for GateFilterCache<F> {
    fn default() -> Self {
        Self {
            enabled: false,
            #[cfg(feature = "std")]
            columns: std::sync::OnceLock::new(),
            #[cfg(not(feature = "std"))]
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F> Debug for GateFilterCache<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "GateFilterCache {{ enabled: {} }}", self.enabled)
    }
}

/// A derived cache is not part of a circuit's identity: two `ProverOnlyCircuitData`
/// that differ only in whether the filter columns happen to be materialized
/// describe the same circuit and prove the same statements.
impl<F> PartialEq for GateFilterCache<F> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<F> Eq for GateFilterCache<F> {}

impl<F: Field> GateFilterCache<F> {
    /// Turns caching on or off for this circuit. Off by default; has no effect
    /// once the columns have been materialized.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Drops any materialized columns. Called when a circuit will not be proved
    /// again in this process.
    pub fn clear(&mut self) {
        #[cfg(feature = "std")]
        {
            self.columns.take();
        }
    }

    /// Returns the cached columns, materializing them with `build` on first use.
    /// Returns `None` when caching is disabled for this circuit.
    #[cfg(feature = "std")]
    pub(crate) fn get_or_init(
        &self,
        build: impl FnOnce() -> Vec<Vec<F>>,
    ) -> Option<&[Vec<F>]> {
        self.enabled
            .then(|| self.columns.get_or_init(build).as_slice())
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn get_or_init(
        &self,
        _build: impl FnOnce() -> Vec<Vec<F>>,
    ) -> Option<&[Vec<F>]> {
        None
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
mod filter_column_tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field64, PrimeField64};

    type F = GoldilocksField;

    /// Verbatim copy of the per-batch filter loop that `fill_filter_column`
    /// factored out, kept here as the differential reference. Any change to the
    /// order or association of the product terms shows up as a raw-limb
    /// mismatch against this.
    fn reference_batch_filters(
        row: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        selector_col: &[F],
    ) -> Vec<F> {
        let batch_size = selector_col.len();
        let mut factors = group_range
            .filter(|&i| i != row)
            .chain((num_selectors > 1).then_some(UNUSED_SELECTOR));
        let mut filters: Vec<F> = Vec::new();
        if let Some(i) = factors.next() {
            let k = F::from_canonical_usize(i);
            filters.extend(selector_col.iter().map(|&s| k - s));
        } else {
            filters.resize(batch_size, F::ONE);
        }
        for i in factors {
            let k = F::from_canonical_usize(i);
            for (filter, &s) in filters.iter_mut().zip(selector_col) {
                *filter *= k - s;
            }
        }
        filters
    }

    /// A selector column standing in for one gathered off the constants
    /// commitment. It deliberately mixes:
    /// - the canonical group indices `0..8` and `UNUSED_SELECTOR`, which make
    ///   the filter of some gate vanish identically on those points;
    /// - `0`, `1`, `ORDER - 1` and raw limbs at and above the field order, which
    ///   is where Goldilocks' lazy `Sub`/`Mul` can hand back non-canonical
    ///   representatives — the case in which "same field value" and "same bits"
    ///   come apart;
    /// - LCG pseudo-random limbs, standing in for generic LDE values.
    fn synthetic_selector_col(len: usize) -> Vec<F> {
        let specials: Vec<u64> = (0..8u64)
            .chain([
                UNUSED_SELECTOR as u64,
                0,
                1,
                F::ORDER - 1,
                F::ORDER,
                F::ORDER + 1,
                u64::MAX,
                1 << 63,
            ])
            .collect();
        let mut state = 0x1234_5678_9abc_def0u64;
        (0..len)
            .map(|i| {
                if i % 5 == 0 {
                    GoldilocksField(specials[(i / 5) % specials.len()])
                } else {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    GoldilocksField(state)
                }
            })
            .collect()
    }

    /// Every shape below is `(row, group_range, num_selectors)`: a singleton
    /// group with one selector (no factors at all), a singleton group inside a
    /// multi-selector circuit (the `UNUSED_SELECTOR` factor only), and groups of
    /// four to six with and without the unused factor.
    fn shapes() -> Vec<(usize, Range<usize>, usize)> {
        vec![
            (0, 0..1, 1),
            (0, 0..1, 3),
            (5, 5..6, 2),
            (2, 0..5, 1),
            (2, 0..5, 4),
            (3, 1..7, 4),
            (7, 2..8, 6),
        ]
    }

    /// The cached column must be raw-limb identical to the per-batch loop it
    /// replaces, for every batch size — including sizes that are not multiples
    /// of the quotient path's 32-point batch, and a short final batch.
    #[test]
    fn cached_filter_column_matches_per_batch_reference() {
        // 1000 is deliberately not a multiple of 32, so the 32-point walk below
        // ends in a short final batch of 8.
        let len = 1000;
        let selector_col = synthetic_selector_col(len);
        let mut saw_zero = false;
        let mut saw_nonzero = false;

        for (row, group_range, num_selectors) in shapes() {
            let mut column = Vec::new();
            fill_filter_column(
                row,
                group_range.clone(),
                num_selectors,
                &selector_col,
                &mut column,
            );
            assert_eq!(column.len(), len);

            for batch_size in [32usize, 17, 1, 999, len] {
                for (batch_i, batch) in selector_col.chunks(batch_size).enumerate() {
                    let expected =
                        reference_batch_filters(row, group_range.clone(), num_selectors, batch);
                    let offset = batch_i * batch_size;
                    let cached = &column[offset..offset + batch.len()];
                    for (k, (got, want)) in cached.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            got.0,
                            want.0,
                            "raw limb mismatch: row {row}, group {group_range:?}, \
                             num_selectors {num_selectors}, batch_size {batch_size}, point {}",
                            offset + k
                        );
                    }
                }
            }

            // The cached column must also agree with the untouched per-point
            // `compute_filter`, which is the definition the batched loop was
            // itself derived from.
            for (k, (&got, &s)) in column.iter().zip(&selector_col).enumerate() {
                let want = compute_filter(row, group_range.clone(), s, num_selectors > 1);
                assert_eq!(got.0, want.0, "raw limb mismatch vs compute_filter at {k}");
                if got.to_canonical_u64() == 0 {
                    saw_zero = true;
                } else {
                    saw_nonzero = true;
                }
            }
        }

        // Coverage guard: at least one shape's filter really does vanish on part
        // of the domain (a selector value landing on another group member's
        // index), and the columns are not trivially all zero.
        assert!(saw_zero, "no shape produced a vanishing filter");
        assert!(saw_nonzero);
    }

    /// `filter_is_trivial` must agree with the column actually produced: it is
    /// the predicate deciding which gates get no cache entry at all, so a false
    /// positive would silently substitute ones for real filter values.
    #[test]
    fn trivial_filters_are_exactly_the_all_ones_columns() {
        let selector_col = synthetic_selector_col(137);
        for (row, group_range, num_selectors) in shapes() {
            let mut column = Vec::new();
            fill_filter_column(
                row,
                group_range.clone(),
                num_selectors,
                &selector_col,
                &mut column,
            );
            let all_ones = column.iter().all(|f| f.0 == F::ONE.0);
            assert_eq!(
                filter_is_trivial(row, group_range.clone(), num_selectors),
                all_ones,
                "row {row}, group {group_range:?}, num_selectors {num_selectors}"
            );
        }
    }
}

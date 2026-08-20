#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::any::TypeId;
use core::marker::PhantomData;
use core::ops::Range;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::{Extendable, FieldExtension, OEF};
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::interpolation::barycentric_weights;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// One of the instantiations of `InterpolationGate`: allows constraints of variable
/// degree, up to `1<<subgroup_bits`.
///
/// This gate has as routed wires
/// - the coset shift from subgroup H
/// - the values that the interpolated polynomial takes on the coset
/// - the evaluation point
///
/// The evaluation strategy is based on the observation that if $P(X)$ is the interpolant of some
/// values over a coset and $P'(X)$ is the interpolant of those values over the subgroup, then
/// $P(X) = P'(X \cdot \mathrm{shift}^{-1})$. Interpolating $P'(X)$ is preferable because when subgroup is fixed
/// then so are the Barycentric weights and both can be hardcoded into the constraint polynomials.
///
/// A full interpolation of N values corresponds to the evaluation of a degree-N polynomial. This
/// gate can however be configured with a bounded degree of at least 2 by introducing more
/// non-routed wires. Let $x[]$ be the domain points, $v[]$ be the values, $w[]$ be the Barycentric
/// weights and $z$ be the evaluation point. Define the sequences
///
/// $p\[0\] = 1,$
///
/// $p\[i\] = p[i - 1] \cdot (z - x[i - 1]),$
///
/// $e\[0\] = 0,$
///
/// $e\[i\] = e[i - 1] ] \cdot (z - x[i - 1]) + w[i - 1] \cdot v[i - 1] \cdot p[i - 1]$
///
/// Then $e\[N\]$ is the final interpolated value. The non-routed wires hold every $(d - 1)$'th
/// intermediate value of $p$ and $e$, starting at $p\[d\]$ and $e\[d\]$, where $d$ is the gate degree.
#[derive(Clone, Debug, Default)]
pub struct CosetInterpolationGate<F: RichField + Extendable<D>, const D: usize> {
    pub subgroup_bits: usize,
    pub degree: usize,
    pub barycentric_weights: Vec<F>,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> CosetInterpolationGate<F, D> {
    pub fn new(subgroup_bits: usize) -> Self {
        Self::with_max_degree(subgroup_bits, 1 << subgroup_bits)
    }

    pub(crate) fn with_max_degree(subgroup_bits: usize, max_degree: usize) -> Self {
        assert!(max_degree > 1, "need at least quadratic constraints");

        let n_points = 1 << subgroup_bits;

        // Number of intermediate values required to compute interpolation with degree bound
        let n_intermediates = (n_points - 2) / (max_degree - 1);

        // Find minimum degree such that (n_points - 2) / (degree - 1) < n_intermediates + 1
        // Minimizing the degree this way allows the gate to be in a larger selector group
        let degree = (n_points - 2) / (n_intermediates + 1) + 2;

        let barycentric_weights = barycentric_weights(
            &F::two_adic_subgroup(subgroup_bits)
                .into_iter()
                .map(|x| (x, F::ZERO))
                .collect::<Vec<_>>(),
        );

        Self {
            subgroup_bits,
            degree,
            barycentric_weights,
            _phantom: PhantomData,
        }
    }

    const fn num_points(&self) -> usize {
        1 << self.subgroup_bits
    }

    /// Wire index of the coset shift.
    pub(crate) const fn wire_shift(&self) -> usize {
        0
    }

    const fn start_values(&self) -> usize {
        1
    }

    /// Wire indices of the `i`th interpolant value.
    pub(crate) fn wires_value(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_points());
        let start = self.start_values() + i * D;
        start..start + D
    }

    const fn start_evaluation_point(&self) -> usize {
        self.start_values() + self.num_points() * D
    }

    /// Wire indices of the point to evaluate the interpolant at.
    pub(crate) const fn wires_evaluation_point(&self) -> Range<usize> {
        let start = self.start_evaluation_point();
        start..start + D
    }

    const fn start_evaluation_value(&self) -> usize {
        self.start_evaluation_point() + D
    }

    /// Wire indices of the interpolated value.
    pub(crate) const fn wires_evaluation_value(&self) -> Range<usize> {
        let start = self.start_evaluation_value();
        start..start + D
    }

    const fn start_intermediates(&self) -> usize {
        self.start_evaluation_value() + D
    }

    pub const fn num_routed_wires(&self) -> usize {
        self.start_intermediates()
    }

    const fn num_intermediates(&self) -> usize {
        (self.num_points() - 2) / (self.degree - 1)
    }

    /// The wires corresponding to the i'th intermediate evaluation.
    const fn wires_intermediate_eval(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_intermediates());
        let start = self.start_intermediates() + D * i;
        start..start + D
    }

    /// The wires corresponding to the i'th intermediate product.
    const fn wires_intermediate_prod(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_intermediates());
        let start = self.start_intermediates() + D * (self.num_intermediates() + i);
        start..start + D
    }

    /// End of wire indices, exclusive.
    const fn end(&self) -> usize {
        self.start_intermediates() + D * (2 * self.num_intermediates() + 1)
    }

    /// Wire indices of the shifted point to evaluate the interpolant at.
    const fn wires_shifted_evaluation_point(&self) -> Range<usize> {
        let start = self.start_intermediates() + D * 2 * self.num_intermediates();
        start..start + D
    }
}

type GoldilocksExt2 = QuadraticExtension<GoldilocksField>;

/// Caller-owned initial state. Passing one immutable aggregate prevents the
/// Rust aggregate ABI from using the individual eval/prod arguments as
/// recurrent write-back slots inside the release loop.
#[derive(Copy, Clone)]
struct PartialInterpolation2Input {
    x_0: GoldilocksExt2,
    initial_eval_0: GoldilocksExt2,
    initial_partial_prod_0: GoldilocksExt2,
    x_1: GoldilocksExt2,
    initial_eval_1: GoldilocksExt2,
    initial_partial_prod_1: GoldilocksExt2,
}

/// Advance two independent Goldilocks quadratic-extension interpolations over
/// the same domain and weights. This deliberately spells each lane as the
/// established scalar recurrence. In particular, it does not pack lanes,
/// fuse extension operations, delay reductions, or reassociate either sum.
#[inline(never)]
fn partial_interpolate_2(
    domain: &[GoldilocksField],
    values_0: &[GoldilocksExt2],
    values_1: &[GoldilocksExt2],
    barycentric_weights: &[GoldilocksField],
    initial: &PartialInterpolation2Input,
) -> (
    (GoldilocksExt2, GoldilocksExt2),
    (GoldilocksExt2, GoldilocksExt2),
) {
    let n = domain.len();
    assert_ne!(n, 0);
    assert_eq!(n, values_0.len());
    assert_eq!(n, values_1.len());
    assert_eq!(n, barycentric_weights.len());

    let x_0 = initial.x_0;
    let mut eval_0 = initial.initial_eval_0;
    let mut partial_prod_0 = initial.initial_partial_prod_0;
    let x_1 = initial.x_1;
    let mut eval_1 = initial.initial_eval_1;
    let mut partial_prod_1 = initial.initial_partial_prod_1;

    for i in 0..n {
        // SAFETY: all four slices were checked to have the same `n` above and
        // `i` is drawn from `0..n`. Unchecked indexing keeps the release loop
        // free of recurrent bounds-check state and cold panic edges.
        let x_i = unsafe { *domain.get_unchecked(i) };
        let weight = unsafe { *barycentric_weights.get_unchecked(i) };
        let value_0 = unsafe { *values_0.get_unchecked(i) };
        let value_1 = unsafe { *values_1.get_unchecked(i) };

        // Lane 0 is one complete, expression-identical scalar step before
        // lane 1 begins. Keep `partial_prod_0` unchanged until both next
        // values have been computed, exactly as the tuple returned by `fold`.
        let val_0 = FieldExtension::<2>::scalar_mul(&value_0, weight);
        let term_0 = x_0 - x_i.into();
        let next_eval_0 = eval_0 * term_0 + val_0 * partial_prod_0;
        let next_partial_prod_0 = partial_prod_0 * term_0;
        eval_0 = next_eval_0;
        partial_prod_0 = next_partial_prod_0;

        // Lane 1 repeats the same complete scalar step with no cross-lane
        // arithmetic or changed operation order.
        let val_1 = FieldExtension::<2>::scalar_mul(&value_1, weight);
        let term_1 = x_1 - x_i.into();
        let next_eval_1 = eval_1 * term_1 + val_1 * partial_prod_1;
        let next_partial_prod_1 = partial_prod_1 * term_1;
        eval_1 = next_eval_1;
        partial_prod_1 = next_partial_prod_1;
    }

    ((eval_0, partial_prod_0), (eval_1, partial_prod_1))
}

#[inline(always)]
fn load_goldilocks_ext2(
    wires: &[GoldilocksField],
    batch_size: usize,
    start_wire: usize,
    point: usize,
) -> GoldilocksExt2 {
    QuadraticExtension([
        wires[start_wire * batch_size + point],
        wires[(start_wire + 1) * batch_size + point],
    ])
}

#[inline(always)]
fn store_goldilocks_ext2(
    scratch: &mut [GoldilocksField],
    batch_size: usize,
    row: usize,
    point: usize,
    value: GoldilocksExt2,
) {
    scratch[row * batch_size + point] = value.0[0];
    scratch[(row + 1) * batch_size + point] = value.0[1];
}

#[cfg(test)]
static COSET_PAIR_BATCH_DISPATCHES: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static COSET_PAIR_POINT_PAIRS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static COSET_PAIR_SCALAR_TAILS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// Established scalar point path for the defensive odd tail of the paired
/// traversal. This is intentionally the same three-segment recurrence and
/// scratch layout as the generic fallback.
fn eval_goldilocks_quadratic_scalar_point(
    gate: &CosetInterpolationGate<GoldilocksField, 2>,
    domain: &[GoldilocksField],
    wires: &[GoldilocksField],
    batch_size: usize,
    point: usize,
    values: &mut [GoldilocksExt2],
    scratch: &mut [GoldilocksField],
) {
    let shift = wires[gate.wire_shift() * batch_size + point];
    let evaluation_point = load_goldilocks_ext2(
        wires,
        batch_size,
        gate.wires_evaluation_point().start,
        point,
    );
    let shifted_point = load_goldilocks_ext2(
        wires,
        batch_size,
        gate.wires_shifted_evaluation_point().start,
        point,
    );
    store_goldilocks_ext2(
        scratch,
        batch_size,
        0,
        point,
        evaluation_point - FieldExtension::<2>::scalar_mul(&shifted_point, shift),
    );

    for (i, value) in values.iter_mut().enumerate() {
        *value = load_goldilocks_ext2(wires, batch_size, gate.wires_value(i).start, point);
    }
    let weights = &gate.barycentric_weights;
    let (mut computed_eval, mut computed_prod) = partial_interpolate::<GoldilocksField, 2>(
        &domain[..6],
        &values[..6],
        &weights[..6],
        shifted_point,
        GoldilocksExt2::ZERO,
        GoldilocksExt2::ONE,
    );

    let mut row = 2;
    for i in 0..2 {
        let intermediate_eval = load_goldilocks_ext2(
            wires,
            batch_size,
            gate.wires_intermediate_eval(i).start,
            point,
        );
        let intermediate_prod = load_goldilocks_ext2(
            wires,
            batch_size,
            gate.wires_intermediate_prod(i).start,
            point,
        );
        store_goldilocks_ext2(
            scratch,
            batch_size,
            row,
            point,
            intermediate_eval - computed_eval,
        );
        row += 2;
        store_goldilocks_ext2(
            scratch,
            batch_size,
            row,
            point,
            intermediate_prod - computed_prod,
        );
        row += 2;

        let start = 6 + 5 * i;
        let end = start + 5;
        (computed_eval, computed_prod) = partial_interpolate::<GoldilocksField, 2>(
            &domain[start..end],
            &values[start..end],
            &weights[start..end],
            shifted_point,
            intermediate_eval,
            intermediate_prod,
        );
    }
    let evaluation_value = load_goldilocks_ext2(
        wires,
        batch_size,
        gate.wires_evaluation_value().start,
        point,
    );
    store_goldilocks_ext2(
        scratch,
        batch_size,
        row,
        point,
        evaluation_value - computed_eval,
    );
}

/// Exact production-shape evaluator: D=2, subgroup bits 4, degree 6,
/// 32 points, and a four-wide Goldilocks packing target. Arithmetic here is
/// scalar; the packing width is only part of the narrow dispatch fingerprint.
fn eval_goldilocks_quadratic_pair_n32(
    gate: &CosetInterpolationGate<GoldilocksField, 2>,
    domain: &[GoldilocksField],
    wires: &[GoldilocksField],
    filters: &[GoldilocksField],
    combined_gate_constraints: &mut [GoldilocksField],
) {
    const N: usize = 32;
    const NUM_POINTS: usize = 16;
    const NUM_CONSTRAINTS: usize = 12;

    debug_assert_eq!(gate.subgroup_bits, 4);
    debug_assert_eq!(gate.degree, 6);
    debug_assert_eq!(domain.len(), NUM_POINTS);
    debug_assert_eq!(gate.barycentric_weights.len(), NUM_POINTS);
    debug_assert_eq!(filters.len(), N);
    debug_assert!(wires.len() >= gate.end() * N);
    debug_assert!(combined_gate_constraints.len() >= NUM_CONSTRAINTS * N);

    #[cfg(test)]
    COSET_PAIR_BATCH_DISPATCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

    // One allocation, like the scalar batch path's `values` buffer, with one
    // 16-value slice per adjacent point lane. Each pair overwrites all slots.
    let mut pair_values = vec![GoldilocksExt2::ZERO; 2 * NUM_POINTS];
    let mut scratch = vec![GoldilocksField::ZERO; NUM_CONSTRAINTS * N];
    let weights = &gate.barycentric_weights;

    let paired_end = N & !1;
    let mut p = 0;
    while p < paired_end {
        let p_0 = p;
        let p_1 = p + 1;
        let (values_0, values_1) = pair_values.split_at_mut(NUM_POINTS);
        for i in 0..NUM_POINTS {
            let start = gate.wires_value(i).start;
            values_0[i] = load_goldilocks_ext2(wires, N, start, p_0);
            values_1[i] = load_goldilocks_ext2(wires, N, start, p_1);
        }

        // Shift constraint, preserving the scalar expression and each lane's
        // original row/point scratch position.
        let shift_0 = wires[gate.wire_shift() * N + p_0];
        let evaluation_point_0 =
            load_goldilocks_ext2(wires, N, gate.wires_evaluation_point().start, p_0);
        let shifted_point_0 =
            load_goldilocks_ext2(wires, N, gate.wires_shifted_evaluation_point().start, p_0);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            0,
            p_0,
            evaluation_point_0 - FieldExtension::<2>::scalar_mul(&shifted_point_0, shift_0),
        );

        let shift_1 = wires[gate.wire_shift() * N + p_1];
        let evaluation_point_1 =
            load_goldilocks_ext2(wires, N, gate.wires_evaluation_point().start, p_1);
        let shifted_point_1 =
            load_goldilocks_ext2(wires, N, gate.wires_shifted_evaluation_point().start, p_1);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            0,
            p_1,
            evaluation_point_1 - FieldExtension::<2>::scalar_mul(&shifted_point_1, shift_1),
        );

        // Production segmentation is exactly 0..6, 6..11, 11..16.
        let ((computed_eval_0, computed_prod_0), (computed_eval_1, computed_prod_1)) =
            partial_interpolate_2(
                &domain[..6],
                &values_0[..6],
                &values_1[..6],
                &weights[..6],
                &PartialInterpolation2Input {
                    x_0: shifted_point_0,
                    initial_eval_0: GoldilocksExt2::ZERO,
                    initial_partial_prod_0: GoldilocksExt2::ONE,
                    x_1: shifted_point_1,
                    initial_eval_1: GoldilocksExt2::ZERO,
                    initial_partial_prod_1: GoldilocksExt2::ONE,
                },
            );

        let intermediate_eval_0 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_eval(0).start, p_0);
        let intermediate_prod_0 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_prod(0).start, p_0);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            2,
            p_0,
            intermediate_eval_0 - computed_eval_0,
        );
        store_goldilocks_ext2(
            &mut scratch,
            N,
            4,
            p_0,
            intermediate_prod_0 - computed_prod_0,
        );
        let intermediate_eval_1 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_eval(0).start, p_1);
        let intermediate_prod_1 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_prod(0).start, p_1);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            2,
            p_1,
            intermediate_eval_1 - computed_eval_1,
        );
        store_goldilocks_ext2(
            &mut scratch,
            N,
            4,
            p_1,
            intermediate_prod_1 - computed_prod_1,
        );

        let ((computed_eval_0, computed_prod_0), (computed_eval_1, computed_prod_1)) =
            partial_interpolate_2(
                &domain[6..11],
                &values_0[6..11],
                &values_1[6..11],
                &weights[6..11],
                &PartialInterpolation2Input {
                    x_0: shifted_point_0,
                    initial_eval_0: intermediate_eval_0,
                    initial_partial_prod_0: intermediate_prod_0,
                    x_1: shifted_point_1,
                    initial_eval_1: intermediate_eval_1,
                    initial_partial_prod_1: intermediate_prod_1,
                },
            );

        let intermediate_eval_0 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_eval(1).start, p_0);
        let intermediate_prod_0 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_prod(1).start, p_0);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            6,
            p_0,
            intermediate_eval_0 - computed_eval_0,
        );
        store_goldilocks_ext2(
            &mut scratch,
            N,
            8,
            p_0,
            intermediate_prod_0 - computed_prod_0,
        );
        let intermediate_eval_1 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_eval(1).start, p_1);
        let intermediate_prod_1 =
            load_goldilocks_ext2(wires, N, gate.wires_intermediate_prod(1).start, p_1);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            6,
            p_1,
            intermediate_eval_1 - computed_eval_1,
        );
        store_goldilocks_ext2(
            &mut scratch,
            N,
            8,
            p_1,
            intermediate_prod_1 - computed_prod_1,
        );

        let ((computed_eval_0, _), (computed_eval_1, _)) = partial_interpolate_2(
            &domain[11..16],
            &values_0[11..16],
            &values_1[11..16],
            &weights[11..16],
            &PartialInterpolation2Input {
                x_0: shifted_point_0,
                initial_eval_0: intermediate_eval_0,
                initial_partial_prod_0: intermediate_prod_0,
                x_1: shifted_point_1,
                initial_eval_1: intermediate_eval_1,
                initial_partial_prod_1: intermediate_prod_1,
            },
        );
        let evaluation_value_0 =
            load_goldilocks_ext2(wires, N, gate.wires_evaluation_value().start, p_0);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            10,
            p_0,
            evaluation_value_0 - computed_eval_0,
        );
        let evaluation_value_1 =
            load_goldilocks_ext2(wires, N, gate.wires_evaluation_value().start, p_1);
        store_goldilocks_ext2(
            &mut scratch,
            N,
            10,
            p_1,
            evaluation_value_1 - computed_eval_1,
        );

        p += 2;
    }

    #[cfg(test)]
    COSET_PAIR_POINT_PAIRS.fetch_add(paired_end / 2, core::sync::atomic::Ordering::Relaxed);

    // Defensive scalar tail. The exact n=32 dispatch has no tail, but keeping
    // the evaluator whole for odd lengths prevents any future shape widening
    // from silently dropping the final point.
    if paired_end < N {
        #[cfg(test)]
        COSET_PAIR_SCALAR_TAILS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        eval_goldilocks_quadratic_scalar_point(
            gate,
            domain,
            wires,
            N,
            paired_end,
            &mut pair_values[..NUM_POINTS],
            &mut scratch,
        );
    }

    // Filtering and the shared sink are unchanged from the scalar fused path.
    for (j, row_slice) in scratch.chunks_exact(N).enumerate() {
        batch_multiply_add_inplace(
            &mut combined_gate_constraints[j * N..][..N],
            row_slice,
            filters,
        );
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for CosetInterpolationGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.subgroup_bits)?;
        dst.write_usize(self.degree)?;
        dst.write_usize(self.barycentric_weights.len())?;
        dst.write_field_vec(&self.barycentric_weights)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let subgroup_bits = src.read_usize()?;
        let degree = src.read_usize()?;
        let length = src.read_usize()?;
        let barycentric_weights: Vec<F> = src.read_field_vec(length)?;
        Ok(Self {
            subgroup_bits,
            degree,
            barycentric_weights,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        let shift = vars.local_wires[self.wire_shift()];
        let evaluation_point = vars.get_local_ext_algebra(self.wires_evaluation_point());
        let shifted_evaluation_point =
            vars.get_local_ext_algebra(self.wires_shifted_evaluation_point());
        constraints.extend(
            (evaluation_point - shifted_evaluation_point.scalar_mul(shift)).to_basefield_array(),
        );

        let domain = F::two_adic_subgroup(self.subgroup_bits);
        let values = (0..self.num_points())
            .map(|i| vars.get_local_ext_algebra(self.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.barycentric_weights;

        let (mut computed_eval, mut computed_prod) = partial_interpolate_ext_algebra(
            &domain[..self.degree()],
            &values[..self.degree()],
            &weights[..self.degree()],
            shifted_evaluation_point,
            ExtensionAlgebra::ZERO,
            ExtensionAlgebra::one(),
        );

        for i in 0..self.num_intermediates() {
            let intermediate_eval = vars.get_local_ext_algebra(self.wires_intermediate_eval(i));
            let intermediate_prod = vars.get_local_ext_algebra(self.wires_intermediate_prod(i));
            constraints.extend((intermediate_eval - computed_eval).to_basefield_array());
            constraints.extend((intermediate_prod - computed_prod).to_basefield_array());

            let start_index = 1 + (self.degree() - 1) * (i + 1);
            let end_index = (start_index + self.degree() - 1).min(self.num_points());
            (computed_eval, computed_prod) = partial_interpolate_ext_algebra(
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                intermediate_eval,
                intermediate_prod,
            );
        }

        let evaluation_value = vars.get_local_ext_algebra(self.wires_evaluation_value());
        constraints.extend((evaluation_value - computed_eval).to_basefield_array());

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let shift = vars.local_wires[self.wire_shift()];
        let evaluation_point = vars.get_local_ext(self.wires_evaluation_point());
        let shifted_evaluation_point = vars.get_local_ext(self.wires_shifted_evaluation_point());
        yield_constr.many(
            (evaluation_point - shifted_evaluation_point.scalar_mul(shift)).to_basefield_array(),
        );

        let domain = crate::field::fft::cached_two_adic_subgroup::<F>(self.subgroup_bits);
        let values = (0..self.num_points())
            .map(|i| vars.get_local_ext(self.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.barycentric_weights;

        let (mut computed_eval, mut computed_prod) = partial_interpolate(
            &domain[..self.degree()],
            &values[..self.degree()],
            &weights[..self.degree()],
            shifted_evaluation_point,
            F::Extension::ZERO,
            F::Extension::ONE,
        );

        for i in 0..self.num_intermediates() {
            let intermediate_eval = vars.get_local_ext(self.wires_intermediate_eval(i));
            let intermediate_prod = vars.get_local_ext(self.wires_intermediate_prod(i));
            yield_constr.many((intermediate_eval - computed_eval).to_basefield_array());
            yield_constr.many((intermediate_prod - computed_prod).to_basefield_array());

            let start_index = 1 + (self.degree() - 1) * (i + 1);
            let end_index = (start_index + self.degree() - 1).min(self.num_points());
            (computed_eval, computed_prod) = partial_interpolate(
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                intermediate_eval,
                intermediate_prod,
            );
        }

        let evaluation_value = vars.get_local_ext(self.wires_evaluation_value());
        yield_constr.many((evaluation_value - computed_eval).to_basefield_array());
    }

    /// Batched fused evaluation. The interpolation itself is inherently
    /// per-point, but this override hoists the subgroup computation and the
    /// values buffer out of the point loop (the default path recomputes the
    /// two-adic subgroup and collects the values `Vec` once per point) and
    /// multiply-adds the filtered constraint rows straight into the shared
    /// buffer.
    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        let num_constraints = <Self as Gate<F, D>>::num_constraints(self);
        assert!(combined_gate_constraints.len() >= num_constraints * n);

        // Cached process-wide: value-identical to `F::two_adic_subgroup`, but
        // skips the primitive-root exponentiation and power chain that would
        // otherwise run once per 32-point batch call.
        let domain = crate::field::fft::cached_two_adic_subgroup::<F>(self.subgroup_bits);
        let weights = &self.barycentric_weights;

        // Keep the optimization pinned to the one ranked production tuple.
        // Every other dimension, gate shape, batch length, packing width, or
        // field type runs the whole established scalar path below.
        if D == 2
            && self.subgroup_bits == 4
            && self.degree == 6
            && self.num_points() == 16
            && n == 32
            && <<F as Packable>::Packing as PackedField>::WIDTH == 4
            && TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
            && TypeId::of::<F::Extension>() == TypeId::of::<GoldilocksExt2>()
            && domain.len() == 16
            && weights.len() == 16
            && vars_base.local_wires.len() >= self.end() * n
        {
            // SAFETY: the two TypeId equalities prove `F` and its D=2
            // extension are exactly the concrete types used below. Therefore
            // each cast preserves element layout, length, and alignment; the
            // remaining shape guards prove the specialized evaluator's fixed
            // wire and constraint layout.
            let gate = unsafe {
                &*(self as *const Self).cast::<CosetInterpolationGate<GoldilocksField, 2>>()
            };
            let domain = unsafe {
                core::slice::from_raw_parts(domain.as_ptr().cast::<GoldilocksField>(), domain.len())
            };
            let wires = unsafe {
                core::slice::from_raw_parts(
                    vars_base.local_wires.as_ptr().cast::<GoldilocksField>(),
                    vars_base.local_wires.len(),
                )
            };
            let filters = unsafe {
                core::slice::from_raw_parts(
                    filters.as_ptr().cast::<GoldilocksField>(),
                    filters.len(),
                )
            };
            let combined = unsafe {
                core::slice::from_raw_parts_mut(
                    combined_gate_constraints
                        .as_mut_ptr()
                        .cast::<GoldilocksField>(),
                    combined_gate_constraints.len(),
                )
            };
            eval_goldilocks_quadratic_pair_n32(gate, domain, wires, filters, combined);
            return;
        }

        let mut values = vec![F::Extension::ZERO; self.num_points()];
        let mut scratch = vec![F::ZERO; num_constraints * n];

        for (p, vars) in vars_base.iter().enumerate() {
            let shift = vars.local_wires[self.wire_shift()];
            let evaluation_point = vars.get_local_ext(self.wires_evaluation_point());
            let shifted_evaluation_point =
                vars.get_local_ext(self.wires_shifted_evaluation_point());
            let arr = (evaluation_point - shifted_evaluation_point.scalar_mul(shift))
                .to_basefield_array();
            for (d, a) in arr.iter().enumerate() {
                scratch[d * n + p] = *a;
            }

            for (i, value) in values.iter_mut().enumerate() {
                *value = vars.get_local_ext(self.wires_value(i));
            }

            let (mut computed_eval, mut computed_prod) = partial_interpolate(
                &domain[..self.degree()],
                &values[..self.degree()],
                &weights[..self.degree()],
                shifted_evaluation_point,
                F::Extension::ZERO,
                F::Extension::ONE,
            );

            let mut row = D;
            for i in 0..self.num_intermediates() {
                let intermediate_eval = vars.get_local_ext(self.wires_intermediate_eval(i));
                let intermediate_prod = vars.get_local_ext(self.wires_intermediate_prod(i));
                let arr = (intermediate_eval - computed_eval).to_basefield_array();
                for (d, a) in arr.iter().enumerate() {
                    scratch[(row + d) * n + p] = *a;
                }
                row += D;
                let arr = (intermediate_prod - computed_prod).to_basefield_array();
                for (d, a) in arr.iter().enumerate() {
                    scratch[(row + d) * n + p] = *a;
                }
                row += D;

                let start_index = 1 + (self.degree() - 1) * (i + 1);
                let end_index = (start_index + self.degree() - 1).min(self.num_points());
                (computed_eval, computed_prod) = partial_interpolate(
                    &domain[start_index..end_index],
                    &values[start_index..end_index],
                    &weights[start_index..end_index],
                    shifted_evaluation_point,
                    intermediate_eval,
                    intermediate_prod,
                );
            }

            let evaluation_value = vars.get_local_ext(self.wires_evaluation_value());
            let arr = (evaluation_value - computed_eval).to_basefield_array();
            for (d, a) in arr.iter().enumerate() {
                scratch[(row + d) * n + p] = *a;
            }
        }

        for (j, row_slice) in scratch.chunks_exact(n).enumerate() {
            batch_multiply_add_inplace(
                &mut combined_gate_constraints[j * n..][..n],
                row_slice,
                filters,
            );
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        let shift = vars.local_wires[self.wire_shift()];
        let evaluation_point = vars.get_local_ext_algebra(self.wires_evaluation_point());
        let shifted_evaluation_point =
            vars.get_local_ext_algebra(self.wires_shifted_evaluation_point());

        let neg_one = builder.neg_one();
        let neg_shift = builder.scalar_mul_ext(neg_one, shift);
        constraints.extend(
            builder
                .scalar_mul_add_ext_algebra(neg_shift, shifted_evaluation_point, evaluation_point)
                .to_ext_target_array(),
        );

        let domain = F::two_adic_subgroup(self.subgroup_bits);
        let values = (0..self.num_points())
            .map(|i| vars.get_local_ext_algebra(self.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.barycentric_weights;

        let initial_eval = builder.zero_ext_algebra();
        let initial_prod = builder.constant_ext_algebra(F::Extension::ONE.into());
        let (mut computed_eval, mut computed_prod) = partial_interpolate_ext_algebra_target(
            builder,
            &domain[..self.degree()],
            &values[..self.degree()],
            &weights[..self.degree()],
            shifted_evaluation_point,
            initial_eval,
            initial_prod,
        );

        for i in 0..self.num_intermediates() {
            let intermediate_eval = vars.get_local_ext_algebra(self.wires_intermediate_eval(i));
            let intermediate_prod = vars.get_local_ext_algebra(self.wires_intermediate_prod(i));
            constraints.extend(
                builder
                    .sub_ext_algebra(intermediate_eval, computed_eval)
                    .to_ext_target_array(),
            );
            constraints.extend(
                builder
                    .sub_ext_algebra(intermediate_prod, computed_prod)
                    .to_ext_target_array(),
            );

            let start_index = 1 + (self.degree() - 1) * (i + 1);
            let end_index = (start_index + self.degree() - 1).min(self.num_points());
            (computed_eval, computed_prod) = partial_interpolate_ext_algebra_target(
                builder,
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                intermediate_eval,
                intermediate_prod,
            );
        }

        let evaluation_value = vars.get_local_ext_algebra(self.wires_evaluation_value());
        constraints.extend(
            builder
                .sub_ext_algebra(evaluation_value, computed_eval)
                .to_ext_target_array(),
        );

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = InterpolationGenerator::<F, D>::new(row, self.clone());
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        self.end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn num_constraints(&self) -> usize {
        // D constraints to check for consistency of the shifted evaluation point, plus D
        // constraints for the evaluation value.
        D + D + 2 * D * self.num_intermediates()
    }
}

#[derive(Debug, Default)]
pub struct InterpolationGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    gate: CosetInterpolationGate<F, D>,
    interpolation_domain: Vec<F>,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> InterpolationGenerator<F, D> {
    fn new(row: usize, gate: CosetInterpolationGate<F, D>) -> Self {
        let interpolation_domain =
            crate::field::fft::cached_two_adic_subgroup::<F>(gate.subgroup_bits).to_vec();
        InterpolationGenerator {
            row,
            gate,
            interpolation_domain,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for InterpolationGenerator<F, D>
{
    fn id(&self) -> String {
        "InterpolationGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| {
            Target::Wire(Wire {
                row: self.row,
                column,
            })
        };

        let local_targets = |columns: Range<usize>| columns.map(local_target);

        let num_points = self.gate.num_points();
        let mut deps = Vec::with_capacity(1 + D + num_points * D);

        deps.push(local_target(self.gate.wire_shift()));
        deps.extend(local_targets(self.gate.wires_evaluation_point()));
        for i in 0..num_points {
            deps.extend(local_targets(self.gate.wires_value(i)));
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

        let get_local_ext = |wire_range: Range<usize>| {
            debug_assert_eq!(wire_range.len(), D);
            let values = wire_range.map(get_local_wire).collect::<Vec<_>>();
            let arr = values.try_into().unwrap();
            F::Extension::from_basefield_array(arr)
        };

        let evaluation_point = get_local_ext(self.gate.wires_evaluation_point());
        let shift = get_local_wire(self.gate.wire_shift());
        let shifted_evaluation_point = evaluation_point.scalar_mul(shift.inverse());
        let degree = self.gate.degree();

        out_buffer.set_ext_wires(
            self.gate.wires_shifted_evaluation_point().map(local_wire),
            shifted_evaluation_point,
        )?;

        let domain = &self.interpolation_domain;
        let values = (0..self.gate.num_points())
            .map(|i| get_local_ext(self.gate.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.gate.barycentric_weights;

        let (mut computed_eval, mut computed_prod) = partial_interpolate(
            &domain[..degree],
            &values[..degree],
            &weights[..degree],
            shifted_evaluation_point,
            F::Extension::ZERO,
            F::Extension::ONE,
        );

        for i in 0..self.gate.num_intermediates() {
            let intermediate_eval_wires = self.gate.wires_intermediate_eval(i).map(local_wire);
            let intermediate_prod_wires = self.gate.wires_intermediate_prod(i).map(local_wire);
            out_buffer.set_ext_wires(intermediate_eval_wires, computed_eval)?;
            out_buffer.set_ext_wires(intermediate_prod_wires, computed_prod)?;

            let start_index = 1 + (degree - 1) * (i + 1);
            let end_index = (start_index + degree - 1).min(self.gate.num_points());
            (computed_eval, computed_prod) = partial_interpolate(
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                computed_eval,
                computed_prod,
            );
        }

        let evaluation_value_wires = self.gate.wires_evaluation_value().map(local_wire);
        out_buffer.set_ext_wires(evaluation_value_wires, computed_eval)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        self.gate.serialize(dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = CosetInterpolationGate::deserialize(src, _common_data)?;
        Ok(Self::new(row, gate))
    }
}

/// Interpolate the polynomial defined by its values on an arbitrary domain at the given point `x`.
///
/// The domain lies in a base field while the values and evaluation point may be from an extension
/// field. The Barycentric weights are precomputed and taken as arguments.
pub fn interpolate_over_base_domain<F: Field + Extendable<D>, const D: usize>(
    domain: &[F],
    values: &[F::Extension],
    barycentric_weights: &[F],
    x: F::Extension,
) -> F::Extension {
    let (result, _) = partial_interpolate(
        domain,
        values,
        barycentric_weights,
        x,
        F::Extension::ZERO,
        F::Extension::ONE,
    );
    result
}

/// Perform a partial interpolation of the polynomial defined by its values on an arbitrary domain.
///
/// The Barycentric algorithm to interpolate a polynomial at a given point `x` is a linear pass
/// over the sequence of domain points, values, and Barycentric weights which maintains two
/// accumulated values, a partial evaluation and a partial product. This partially updates the
/// accumulated values, so that starting with an initial evaluation of 0 and a partial evaluation
/// of 1 and running over the whole domain is a full interpolation.
fn partial_interpolate<F: Field + Extendable<D>, const D: usize>(
    domain: &[F],
    values: &[F::Extension],
    barycentric_weights: &[F],
    x: F::Extension,
    initial_eval: F::Extension,
    initial_partial_prod: F::Extension,
) -> (F::Extension, F::Extension) {
    let n = domain.len();
    assert_ne!(n, 0);
    assert_eq!(n, values.len());
    assert_eq!(n, barycentric_weights.len());

    let weighted_values = values
        .iter()
        .zip(barycentric_weights.iter())
        .map(|(&value, &weight)| value.scalar_mul(weight));

    weighted_values.zip(domain.iter()).fold(
        (initial_eval, initial_partial_prod),
        |(eval, terms_partial_prod), (val, &x_i)| {
            let term = x - x_i.into();
            let next_eval = eval * term + val * terms_partial_prod;
            let next_terms_partial_prod = terms_partial_prod * term;
            (next_eval, next_terms_partial_prod)
        },
    )
}

fn partial_interpolate_ext_algebra<F: OEF<D>, const D: usize>(
    domain: &[F::BaseField],
    values: &[ExtensionAlgebra<F, D>],
    barycentric_weights: &[F::BaseField],
    x: ExtensionAlgebra<F, D>,
    initial_eval: ExtensionAlgebra<F, D>,
    initial_partial_prod: ExtensionAlgebra<F, D>,
) -> (ExtensionAlgebra<F, D>, ExtensionAlgebra<F, D>) {
    let n = domain.len();
    assert_ne!(n, 0);
    assert_eq!(n, values.len());
    assert_eq!(n, barycentric_weights.len());

    let weighted_values = values
        .iter()
        .zip(barycentric_weights.iter())
        .map(|(&value, &weight)| value.scalar_mul(F::from_basefield(weight)));

    weighted_values.zip(domain.iter()).fold(
        (initial_eval, initial_partial_prod),
        |(eval, terms_partial_prod), (val, &x_i)| {
            let term = x - F::from_basefield(x_i).into();
            let next_eval = eval * term + val * terms_partial_prod;
            let next_terms_partial_prod = terms_partial_prod * term;
            (next_eval, next_terms_partial_prod)
        },
    )
}

fn partial_interpolate_ext_algebra_target<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    domain: &[F],
    values: &[ExtensionAlgebraTarget<D>],
    barycentric_weights: &[F],
    point: ExtensionAlgebraTarget<D>,
    initial_eval: ExtensionAlgebraTarget<D>,
    initial_partial_prod: ExtensionAlgebraTarget<D>,
) -> (ExtensionAlgebraTarget<D>, ExtensionAlgebraTarget<D>) {
    let n = values.len();
    debug_assert!(n != 0);
    debug_assert!(domain.len() == n);
    debug_assert!(barycentric_weights.len() == n);

    values
        .iter()
        .cloned()
        .zip(domain.iter().cloned())
        .zip(barycentric_weights.iter().cloned())
        .fold(
            (initial_eval, initial_partial_prod),
            |(eval, partial_prod), ((val, x), weight)| {
                let x_target = builder.constant_ext_algebra(F::Extension::from(x).into());
                let weight_target = builder.constant_extension(F::Extension::from(weight));
                let term = builder.sub_ext_algebra(point, x_target);
                let weighted_val = builder.scalar_mul_ext_algebra(weight_target, val);
                let new_eval = builder.mul_ext_algebra(eval, term);
                let new_eval = builder.mul_add_ext_algebra(weighted_val, partial_prod, new_eval);
                let new_partial_prod = builder.mul_ext_algebra(partial_prod, term);
                (new_eval, new_partial_prod)
            },
        )
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2_field::polynomial::PolynomialValues;
    use plonky2_util::log2_strict;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field64, Sample};
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    fn scalar_fused_accumulate_reference(
        gate: &CosetInterpolationGate<GoldilocksField, 2>,
        vars_base: EvaluationVarsBaseBatch<GoldilocksField>,
        filters: &[GoldilocksField],
        combined_gate_constraints: &mut [GoldilocksField],
    ) {
        let n = vars_base.len();
        let num_constraints = <CosetInterpolationGate<GoldilocksField, 2> as Gate<
            GoldilocksField,
            2,
        >>::num_constraints(gate);
        let domain =
            crate::field::fft::cached_two_adic_subgroup::<GoldilocksField>(gate.subgroup_bits);
        let weights = &gate.barycentric_weights;
        let mut values = vec![GoldilocksExt2::ZERO; gate.num_points()];
        let mut scratch = vec![GoldilocksField::ZERO; num_constraints * n];

        for (p, vars) in vars_base.iter().enumerate() {
            let shift = vars.local_wires[gate.wire_shift()];
            let evaluation_point: GoldilocksExt2 =
                vars.get_local_ext::<2>(gate.wires_evaluation_point());
            let shifted_evaluation_point: GoldilocksExt2 =
                vars.get_local_ext::<2>(gate.wires_shifted_evaluation_point());
            let arr = (evaluation_point
                - FieldExtension::<2>::scalar_mul(&shifted_evaluation_point, shift))
            .0;
            for (d, a) in arr.iter().enumerate() {
                scratch[d * n + p] = *a;
            }

            for (i, value) in values.iter_mut().enumerate() {
                *value = vars.get_local_ext::<2>(gate.wires_value(i));
            }
            let (mut computed_eval, mut computed_prod) = partial_interpolate::<_, 2>(
                &domain[..gate.degree()],
                &values[..gate.degree()],
                &weights[..gate.degree()],
                shifted_evaluation_point,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
            );

            let mut row = 2;
            for i in 0..gate.num_intermediates() {
                let intermediate_eval: GoldilocksExt2 =
                    vars.get_local_ext::<2>(gate.wires_intermediate_eval(i));
                let intermediate_prod: GoldilocksExt2 =
                    vars.get_local_ext::<2>(gate.wires_intermediate_prod(i));
                let arr = (intermediate_eval - computed_eval).0;
                for (d, a) in arr.iter().enumerate() {
                    scratch[(row + d) * n + p] = *a;
                }
                row += 2;
                let arr = (intermediate_prod - computed_prod).0;
                for (d, a) in arr.iter().enumerate() {
                    scratch[(row + d) * n + p] = *a;
                }
                row += 2;

                let start = 1 + (gate.degree() - 1) * (i + 1);
                let end = (start + gate.degree() - 1).min(gate.num_points());
                (computed_eval, computed_prod) = partial_interpolate::<_, 2>(
                    &domain[start..end],
                    &values[start..end],
                    &weights[start..end],
                    shifted_evaluation_point,
                    intermediate_eval,
                    intermediate_prod,
                );
            }
            let evaluation_value: GoldilocksExt2 =
                vars.get_local_ext::<2>(gate.wires_evaluation_value());
            let arr = (evaluation_value - computed_eval).0;
            for (d, a) in arr.iter().enumerate() {
                scratch[(row + d) * n + p] = *a;
            }
        }

        for (j, row_slice) in scratch.chunks_exact(n).enumerate() {
            batch_multiply_add_inplace(
                &mut combined_gate_constraints[j * n..][..n],
                row_slice,
                filters,
            );
        }
    }

    fn raw_ext(value: GoldilocksExt2) -> [u64; 2] {
        [value.0[0].0, value.0[1].0]
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_pair_raw_matches_two_scalar_calls(
        domain: &[GoldilocksField],
        values_0: &[GoldilocksExt2],
        values_1: &[GoldilocksExt2],
        weights: &[GoldilocksField],
        x_0: GoldilocksExt2,
        eval_0: GoldilocksExt2,
        prod_0: GoldilocksExt2,
        x_1: GoldilocksExt2,
        eval_1: GoldilocksExt2,
        prod_1: GoldilocksExt2,
    ) -> (
        (GoldilocksExt2, GoldilocksExt2),
        (GoldilocksExt2, GoldilocksExt2),
    ) {
        let actual = partial_interpolate_2(
            domain,
            values_0,
            values_1,
            weights,
            &PartialInterpolation2Input {
                x_0,
                initial_eval_0: eval_0,
                initial_partial_prod_0: prod_0,
                x_1,
                initial_eval_1: eval_1,
                initial_partial_prod_1: prod_1,
            },
        );
        let expected_0 = partial_interpolate::<GoldilocksField, 2>(
            domain, values_0, weights, x_0, eval_0, prod_0,
        );
        let expected_1 = partial_interpolate::<GoldilocksField, 2>(
            domain, values_1, weights, x_1, eval_1, prod_1,
        );
        assert_eq!(raw_ext(actual.0 .0), raw_ext(expected_0.0));
        assert_eq!(raw_ext(actual.0 .1), raw_ext(expected_0.1));
        assert_eq!(raw_ext(actual.1 .0), raw_ext(expected_1.0));
        assert_eq!(raw_ext(actual.1 .1), raw_ext(expected_1.1));
        actual
    }

    fn raw_edge_ext(edges: &[u64], offset: usize) -> GoldilocksExt2 {
        QuadraticExtension([
            GoldilocksField(edges[offset % edges.len()]),
            GoldilocksField(edges[(offset + 1) % edges.len()]),
        ])
    }

    /// Checks the fused `eval_unfiltered_base_batch_accumulate` against the
    /// (unchanged) per-point default batch evaluation across a multi-point
    /// batch, including the bounded-degree production shape (subgroup bits 4,
    /// degree 6) used by the recursion circuits, which the public constructor
    /// cannot reach.
    #[test]
    fn test_accumulate_matches_default_across_batch() {
        const D: usize = 2;
        type F = GoldilocksField;

        let n = 32;
        for max_degree in [2, 3, 6, 16] {
            let gate = <CosetInterpolationGate<F, D>>::with_max_degree(4, max_degree);
            let num_wires = <CosetInterpolationGate<F, D> as Gate<F, D>>::num_wires(&gate);
            let num_constraints =
                <CosetInterpolationGate<F, D> as Gate<F, D>>::num_constraints(&gate);

            let wires_batch: Vec<F> = (0..num_wires * n).map(|_| F::rand()).collect();
            let filters: Vec<F> = (0..n).map(|_| F::rand()).collect();
            let initial: Vec<F> = (0..num_constraints * n).map(|_| F::rand()).collect();
            let public_inputs_hash = HashOut::<F>::ZERO;
            let vars_batch =
                EvaluationVarsBaseBatch::new(n, &[], &wires_batch, &public_inputs_hash);

            let reference = gate.eval_unfiltered_base_batch(vars_batch);
            let mut expected = initial.clone();
            for (combined, row) in expected.chunks_exact_mut(n).zip(reference.chunks_exact(n)) {
                for p in 0..n {
                    combined[p] += row[p] * filters[p];
                }
            }

            let mut actual = initial;
            gate.eval_unfiltered_base_batch_accumulate(vars_batch, &filters, &mut actual);
            assert_eq!(actual, expected, "max_degree {max_degree}");
        }
    }

    #[test]
    fn partial_interpolate_pair_matches_raw_scalar_steps_and_segments() {
        type F = GoldilocksField;
        let edges = [0, 1, F::ORDER, F::ORDER + 1, u64::MAX];
        let domain = (0..16)
            .map(|i| GoldilocksField(edges[(3 * i) % edges.len()]))
            .collect::<Vec<_>>();
        let weights = (0..16)
            .map(|i| GoldilocksField(edges[(3 * i + 1) % edges.len()]))
            .collect::<Vec<_>>();
        let values_0 = (0..16)
            .map(|i| raw_edge_ext(&edges, 2 * i))
            .collect::<Vec<_>>();
        let values_1 = (0..16)
            .map(|i| raw_edge_ext(&edges, 2 * i + 3))
            .collect::<Vec<_>>();
        let x_0 = raw_edge_ext(&edges, 0);
        let eval_0 = raw_edge_ext(&edges, 1);
        let prod_0 = raw_edge_ext(&edges, 2);
        let x_1 = raw_edge_ext(&edges, 3);
        let eval_1 = raw_edge_ext(&edges, 4);
        let prod_1 = raw_edge_ext(&edges, 5);

        // Every prefix checks the state after every individual recurrence
        // step; the requested 1/5/6/16 lengths are included explicitly.
        for len in 1..=16 {
            assert_pair_raw_matches_two_scalar_calls(
                &domain[..len],
                &values_0[..len],
                &values_1[..len],
                &weights[..len],
                x_0,
                eval_0,
                prod_0,
                x_1,
                eval_1,
                prod_1,
            );
        }

        // And check the actual production carry chain across 6/5/5.
        let first = assert_pair_raw_matches_two_scalar_calls(
            &domain[..6],
            &values_0[..6],
            &values_1[..6],
            &weights[..6],
            x_0,
            eval_0,
            prod_0,
            x_1,
            eval_1,
            prod_1,
        );
        let second = assert_pair_raw_matches_two_scalar_calls(
            &domain[6..11],
            &values_0[6..11],
            &values_1[6..11],
            &weights[6..11],
            x_0,
            first.0 .0,
            first.0 .1,
            x_1,
            first.1 .0,
            first.1 .1,
        );
        assert_pair_raw_matches_two_scalar_calls(
            &domain[11..16],
            &values_0[11..16],
            &values_1[11..16],
            &weights[11..16],
            x_0,
            second.0 .0,
            second.0 .1,
            x_1,
            second.1 .0,
            second.1 .1,
        );

        // A ragged tuple must be rejected as a whole, never partially paired.
        assert!(std::panic::catch_unwind(|| {
            partial_interpolate_2(
                &domain[..6],
                &values_0[..5],
                &values_1[..6],
                &weights[..6],
                &PartialInterpolation2Input {
                    x_0,
                    initial_eval_0: eval_0,
                    initial_partial_prod_0: prod_0,
                    x_1,
                    initial_eval_1: eval_1,
                    initial_partial_prod_1: prod_1,
                },
            )
        })
        .is_err());
    }

    #[test]
    fn partial_interpolate_pair_one_million_random_raw_steps() {
        let mut state = 0xA59C_05E7_1A7E_2D31u64;
        let mut next = || {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            state
        };

        for _ in 0..1_000_000 {
            let domain = [GoldilocksField(next())];
            let weights = [GoldilocksField(next())];
            let values_0 = [QuadraticExtension([
                GoldilocksField(next()),
                GoldilocksField(next()),
            ])];
            let values_1 = [QuadraticExtension([
                GoldilocksField(next()),
                GoldilocksField(next()),
            ])];
            let mut ext = || QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]);
            let x_0 = ext();
            let eval_0 = ext();
            let prod_0 = ext();
            let x_1 = ext();
            let eval_1 = ext();
            let prod_1 = ext();
            assert_pair_raw_matches_two_scalar_calls(
                &domain, &values_0, &values_1, &weights, x_0, eval_0, prod_0, x_1, eval_1, prod_1,
            );
        }
    }

    #[test]
    fn production_pair_constraints_are_raw_identical_and_dispatches_16_pairs() {
        type F = GoldilocksField;
        const N: usize = 32;
        let gate = <CosetInterpolationGate<F, 2>>::with_max_degree(4, 6);
        assert_eq!(gate.degree, 6);
        assert_eq!(<<F as Packable>::Packing as PackedField>::WIDTH, 4);
        let num_wires = <CosetInterpolationGate<F, 2> as Gate<F, 2>>::num_wires(&gate);
        let num_constraints = <CosetInterpolationGate<F, 2> as Gate<F, 2>>::num_constraints(&gate);
        let edges = [0, 1, F::ORDER, F::ORDER + 1, u64::MAX];
        let mut state = 0xD1FF_E2E2_5A59_0042u64;
        let mut draw = |i: usize| {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            if i % 11 < edges.len() {
                GoldilocksField(edges[i % edges.len()])
            } else {
                GoldilocksField(state)
            }
        };
        let wires = (0..num_wires * N).map(&mut draw).collect::<Vec<_>>();
        let filters = (0..N).map(|i| draw(num_wires * N + i)).collect::<Vec<_>>();
        let initial = (0..num_constraints * N)
            .map(|i| draw(num_wires * N + N + i))
            .collect::<Vec<_>>();
        let public_inputs_hash = HashOut::<F>::ZERO;
        let vars = EvaluationVarsBaseBatch::new(N, &[], &wires, &public_inputs_hash);

        let mut expected = initial.clone();
        scalar_fused_accumulate_reference(&gate, vars, &filters, &mut expected);
        let dispatches = COSET_PAIR_BATCH_DISPATCHES.load(core::sync::atomic::Ordering::Relaxed);
        let pairs = COSET_PAIR_POINT_PAIRS.load(core::sync::atomic::Ordering::Relaxed);
        let tails = COSET_PAIR_SCALAR_TAILS.load(core::sync::atomic::Ordering::Relaxed);
        let mut actual = initial;
        gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);

        assert_eq!(
            actual.iter().map(|x| x.0).collect::<Vec<_>>(),
            expected.iter().map(|x| x.0).collect::<Vec<_>>()
        );
        assert!(
            COSET_PAIR_BATCH_DISPATCHES.load(core::sync::atomic::Ordering::Relaxed)
                >= dispatches + 1
        );
        assert!(COSET_PAIR_POINT_PAIRS.load(core::sync::atomic::Ordering::Relaxed) >= pairs + 16);
        assert_eq!(
            COSET_PAIR_SCALAR_TAILS.load(core::sync::atomic::Ordering::Relaxed),
            tails
        );
    }

    #[test]
    fn nonproduction_batches_and_dimensions_take_scalar_whole_fallback() {
        type F = GoldilocksField;
        let gate = <CosetInterpolationGate<F, 2>>::with_max_degree(4, 6);
        let num_wires = <CosetInterpolationGate<F, 2> as Gate<F, 2>>::num_wires(&gate);
        let num_constraints = <CosetInterpolationGate<F, 2> as Gate<F, 2>>::num_constraints(&gate);
        let public_inputs_hash = HashOut::<F>::ZERO;

        // Odd and otherwise ragged point counts stay wholly scalar.
        for n in [1, 5, 6, 16, 31, 33] {
            let wires = (0..num_wires * n).map(|_| F::rand()).collect::<Vec<_>>();
            let filters = (0..n).map(|_| F::rand()).collect::<Vec<_>>();
            let initial = (0..num_constraints * n)
                .map(|_| F::rand())
                .collect::<Vec<_>>();
            let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &public_inputs_hash);
            let mut expected = initial.clone();
            scalar_fused_accumulate_reference(&gate, vars, &filters, &mut expected);
            let mut actual = initial;
            gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
            assert_eq!(actual, expected, "batch length {n}");
        }

        // D=4 with the same subgroup/max-degree/n tuple must also stay on the
        // generic scalar implementation.
        let gate_d4 = <CosetInterpolationGate<F, 4>>::with_max_degree(4, 6);
        let n = 32;
        let nw = <CosetInterpolationGate<F, 4> as Gate<F, 4>>::num_wires(&gate_d4);
        let nc = <CosetInterpolationGate<F, 4> as Gate<F, 4>>::num_constraints(&gate_d4);
        let wires = (0..nw * n).map(|_| F::rand()).collect::<Vec<_>>();
        let filters = (0..n).map(|_| F::rand()).collect::<Vec<_>>();
        let initial = (0..nc * n).map(|_| F::rand()).collect::<Vec<_>>();
        let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &public_inputs_hash);
        let reference = gate_d4.eval_unfiltered_base_batch(vars);
        let mut expected = initial.clone();
        for (combined, row) in expected.chunks_exact_mut(n).zip(reference.chunks_exact(n)) {
            for p in 0..n {
                combined[p] += row[p] * filters[p];
            }
        }
        let mut actual = initial;
        gate_d4.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
        assert_eq!(actual, expected);
    }

    /// Exact production-batch microbenchmark of the scalar fused path against
    /// the adjacent-pair recurrence. Samples alternate AB/BA. Run with:
    /// `cargo test --manifest-path vendor/plonky2/Cargo.toml --release -p plonky2 \
    ///    --lib -- --ignored --exact gates::coset_interpolation::tests::coset_interpolation_accumulate_microbench --nocapture`
    #[test]
    #[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
    fn coset_interpolation_accumulate_microbench() {
        use core::hint::black_box;
        use std::time::Instant;

        const D: usize = 2;
        type F = GoldilocksField;

        let n = 32;
        let iters = 10_000;
        let gate = <CosetInterpolationGate<F, D>>::with_max_degree(4, 6);
        let num_wires = <CosetInterpolationGate<F, D> as Gate<F, D>>::num_wires(&gate);
        let num_constraints = <CosetInterpolationGate<F, D> as Gate<F, D>>::num_constraints(&gate);
        let wires_batch: Vec<F> = (0..num_wires * n).map(|_| F::rand()).collect();
        let filters: Vec<F> = (0..n).map(|_| F::rand()).collect();
        let public_inputs_hash = HashOut::<F>::ZERO;
        let vars_batch = EvaluationVarsBaseBatch::new(n, &[], &wires_batch, &public_inputs_hash);

        let mut scalar_samples = Vec::new();
        let mut pair_samples = Vec::new();
        for order in [[false, true], [true, false], [false, true], [true, false]] {
            for paired in order {
                let mut combined = vec![F::ZERO; num_constraints * n];
                let start = Instant::now();
                if paired {
                    for _ in 0..iters {
                        gate.eval_unfiltered_base_batch_accumulate(
                            black_box(vars_batch),
                            black_box(&filters),
                            black_box(&mut combined),
                        );
                    }
                } else {
                    for _ in 0..iters {
                        scalar_fused_accumulate_reference(
                            black_box(&gate),
                            black_box(vars_batch),
                            black_box(&filters),
                            black_box(&mut combined),
                        );
                    }
                }
                let ns = start.elapsed().as_nanos() as f64 / iters as f64;
                black_box(&combined);
                if paired {
                    pair_samples.push(ns);
                } else {
                    scalar_samples.push(ns);
                }
            }
        }
        scalar_samples.sort_by(f64::total_cmp);
        pair_samples.sort_by(f64::total_cmp);
        let scalar = (scalar_samples[1] + scalar_samples[2]) / 2.0;
        let pair = (pair_samples[1] + pair_samples[2]) / 2.0;
        println!("scalar ns/call: {scalar_samples:?}");
        println!("paired ns/call: {pair_samples:?}");
        println!(
            "median {:>10.3}us -> {:>10.3}us ({:+.3}%) CosetInterpolationGate(subgroup_bits=4, degree=6, n=32)",
            scalar / 1e3,
            pair / 1e3,
            100.0 * (scalar - pair) / scalar,
        );
    }

    #[test]
    fn test_degree_and_wires_minimized() {
        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 2);
        assert_eq!(gate.num_intermediates(), 6);
        assert_eq!(gate.degree(), 2);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 3);
        assert_eq!(gate.num_intermediates(), 3);
        assert_eq!(gate.degree(), 3);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 4);
        assert_eq!(gate.num_intermediates(), 2);
        assert_eq!(gate.degree(), 4);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 5);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 5);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 6);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 5);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 7);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 5);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 3);
        assert_eq!(gate.num_intermediates(), 7);
        assert_eq!(gate.degree(), 3);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 6);
        assert_eq!(gate.num_intermediates(), 2);
        assert_eq!(gate.degree(), 6);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 8);
        assert_eq!(gate.num_intermediates(), 2);
        assert_eq!(gate.degree(), 6);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 9);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 9);
    }

    #[test]
    fn wire_indices_degree2() {
        let gate = CosetInterpolationGate::<GoldilocksField, 4> {
            subgroup_bits: 2,
            degree: 2,
            barycentric_weights: barycentric_weights(
                &GoldilocksField::two_adic_subgroup(2)
                    .into_iter()
                    .map(|x| (x, GoldilocksField::ZERO))
                    .collect::<Vec<_>>(),
            ),
            _phantom: PhantomData,
        };

        // The exact indices aren't really important, but we want to make sure we don't have any
        // overlaps or gaps.
        assert_eq!(gate.wire_shift(), 0);
        assert_eq!(gate.wires_value(0), 1..5);
        assert_eq!(gate.wires_value(1), 5..9);
        assert_eq!(gate.wires_value(2), 9..13);
        assert_eq!(gate.wires_value(3), 13..17);
        assert_eq!(gate.wires_evaluation_point(), 17..21);
        assert_eq!(gate.wires_evaluation_value(), 21..25);
        assert_eq!(gate.wires_intermediate_eval(0), 25..29);
        assert_eq!(gate.wires_intermediate_eval(1), 29..33);
        assert_eq!(gate.wires_intermediate_prod(0), 33..37);
        assert_eq!(gate.wires_intermediate_prod(1), 37..41);
        assert_eq!(gate.wires_shifted_evaluation_point(), 41..45);
        assert_eq!(gate.num_wires(), 45);
    }

    #[test]
    fn wire_indices_degree_3() {
        let gate = CosetInterpolationGate::<GoldilocksField, 4> {
            subgroup_bits: 2,
            degree: 3,
            barycentric_weights: barycentric_weights(
                &GoldilocksField::two_adic_subgroup(2)
                    .into_iter()
                    .map(|x| (x, GoldilocksField::ZERO))
                    .collect::<Vec<_>>(),
            ),
            _phantom: PhantomData,
        };

        // The exact indices aren't really important, but we want to make sure we don't have any
        // overlaps or gaps.
        assert_eq!(gate.wire_shift(), 0);
        assert_eq!(gate.wires_value(0), 1..5);
        assert_eq!(gate.wires_value(1), 5..9);
        assert_eq!(gate.wires_value(2), 9..13);
        assert_eq!(gate.wires_value(3), 13..17);
        assert_eq!(gate.wires_evaluation_point(), 17..21);
        assert_eq!(gate.wires_evaluation_value(), 21..25);
        assert_eq!(gate.wires_intermediate_eval(0), 25..29);
        assert_eq!(gate.wires_intermediate_prod(0), 29..33);
        assert_eq!(gate.wires_shifted_evaluation_point(), 33..37);
        assert_eq!(gate.num_wires(), 37);
    }

    #[test]
    fn wire_indices_degree_n() {
        let gate = CosetInterpolationGate::<GoldilocksField, 4> {
            subgroup_bits: 2,
            degree: 4,
            barycentric_weights: barycentric_weights(
                &GoldilocksField::two_adic_subgroup(2)
                    .into_iter()
                    .map(|x| (x, GoldilocksField::ZERO))
                    .collect::<Vec<_>>(),
            ),
            _phantom: PhantomData,
        };

        // The exact indices aren't really important, but we want to make sure we don't have any
        // overlaps or gaps.
        assert_eq!(gate.wire_shift(), 0);
        assert_eq!(gate.wires_value(0), 1..5);
        assert_eq!(gate.wires_value(1), 5..9);
        assert_eq!(gate.wires_value(2), 9..13);
        assert_eq!(gate.wires_value(3), 13..17);
        assert_eq!(gate.wires_evaluation_point(), 17..21);
        assert_eq!(gate.wires_evaluation_value(), 21..25);
        assert_eq!(gate.wires_shifted_evaluation_point(), 25..29);
        assert_eq!(gate.num_wires(), 29);
    }

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(CosetInterpolationGate::new(2));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        for degree in 2..=4 {
            test_eval_fns::<F, C, _, D>(CosetInterpolationGate::with_max_degree(2, degree))?;
        }
        Ok(())
    }

    #[test]
    fn test_gate_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        /// Returns the local wires for an interpolation gate for given coeffs, points and eval point.
        fn get_wires(shift: F, values: PolynomialValues<FF>, eval_point: FF) -> Vec<FF> {
            let domain = F::two_adic_subgroup(log2_strict(values.len()));
            let shifted_eval_point =
                <FF as FieldExtension<2>>::scalar_mul(&eval_point, shift.inverse());
            let weights =
                barycentric_weights(&domain.iter().map(|&x| (x, F::ZERO)).collect::<Vec<_>>());
            let (intermediate_eval, intermediate_prod) = partial_interpolate::<_, D>(
                &domain[..3],
                &values.values[..3],
                &weights[..3],
                shifted_eval_point,
                FF::ZERO,
                FF::ONE,
            );
            let eval = interpolate_over_base_domain::<_, D>(
                &domain,
                &values.values,
                &weights,
                shifted_eval_point,
            );
            let mut v = vec![shift];
            for val in values.values.iter() {
                v.extend(val.0);
            }
            v.extend(eval_point.0);
            v.extend(eval.0);
            v.extend(intermediate_eval.0);
            v.extend(intermediate_prod.0);
            v.extend(shifted_eval_point.0);
            v.iter().map(|&x| x.into()).collect()
        }

        // Get a working row for InterpolationGate.
        let shift = F::rand();
        let values = PolynomialValues::new(core::iter::repeat_with(FF::rand).take(4).collect());
        let eval_point = FF::rand();
        let gate = CosetInterpolationGate::<F, D>::with_max_degree(2, 3);
        let vars = EvaluationVars {
            local_constants: &[],
            local_wires: &get_wires(shift, values, eval_point),
            public_inputs_hash: &HashOut::rand(),
        };

        assert!(
            gate.eval_unfiltered(vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
    }

    #[test]
    fn test_num_wires_constraints() {
        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 8);
        assert_eq!(gate.num_wires(), 47);
        assert_eq!(gate.num_constraints(), 12);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 8);
        assert_eq!(gate.num_wires(), 23);
        assert_eq!(gate.num_constraints(), 4);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 16);
        assert_eq!(gate.num_wires(), 39);
        assert_eq!(gate.num_constraints(), 4);
    }
}

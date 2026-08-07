// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Shared batch-accumulation primitives for the custom gates'
//! `eval_unfiltered_base_batch_accumulate` paths.

use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;

/// Accumulate `filters[p] * limb[p](limb[p]-1)(limb[p]-2)(limb[p]-3)` into
/// `combined[p]`, evaluated as `y(y+2)` with `y = x(x-3)`.
///
/// Every uint and byte gate previously materialized this base-4 range product
/// into a scratch row and then handed that row to `batch_multiply_add_inplace`,
/// which read it straight back. This runs the same packed
/// `multiply_accumulate` over the same operands in the same order, so it is
/// exact on raw representatives rather than merely ring-identical -- there is
/// no reassociation. What disappears is one store-and-reload of the scratch row
/// per limb, and these loops run once per limb per op (16 limbs for a 32-bit
/// output, 24 for a 48-bit one).
///
/// Splitting at a multiple of `Packing::WIDTH` and casting the prefix is
/// exactly what `batch_multiply_add_inplace` already does to these same
/// `combined` subslices, so the packing carries no new alignment assumption.
/// The remainder is handled scalar, as before.
// `#[inline]` alone left this outlined as a real call per limb -- it showed up
// as its own 13k-sample symbol in the profile. Force it into the gate loops.
#[inline(always)]
pub fn accumulate_base4_range_product<F: Field + Packable>(
    combined: &mut [F],
    limb: &[F],
    filters: &[F],
    three: F,
) {
    let n = combined.len();
    debug_assert_eq!(limb.len(), n);
    debug_assert_eq!(filters.len(), n);

    let split = n - n % <<F as Packable>::Packing as PackedField>::WIDTH;
    let (combined_head, combined_tail) = combined.split_at_mut(split);
    let (limb_head, limb_tail) = limb.split_at(split);
    let (filters_head, filters_tail) = filters.split_at(split);

    for ((acc, &x), &f) in <F as Packable>::Packing::pack_slice_mut(combined_head)
        .iter_mut()
        .zip(<F as Packable>::Packing::pack_slice(limb_head))
        .zip(<F as Packable>::Packing::pack_slice(filters_head))
    {
        let y = x * (x - three);
        *acc = acc.multiply_accumulate(y * (y + F::TWO), f);
    }
    for ((acc, &x), &f) in combined_tail.iter_mut().zip(limb_tail).zip(filters_tail) {
        let y = x * (x - three);
        *acc += y * (y + F::TWO) * f;
    }
}

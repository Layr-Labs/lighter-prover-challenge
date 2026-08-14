// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Montgomery batch inversion for non-native (secp256k1) witness generators.
//! Marker: batch-inv-1786692400.
//!
//! Each `NonNativeInverseGenerator` still writes the unique inverse the
//! original per-site `try_inverse` produced. Ready denominators from one
//! circuit are inverted together (prefix products, one field inverse, unwind).
//! A wrong inverse fails the existing `x * inv = 1 + k·p` constraint, so the
//! path is fail-closed.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use num::{BigUint, Integer, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::extension::FieldExtension;
use plonky2::field::types::{Field, PrimeField};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness};

/// One `NonNativeInverseGenerator` site: the denominator limbs and the two
/// output buffers (`inv`, `div`) the constraint system already allocated.
#[derive(Clone, Debug)]
pub struct InvSite {
    pub x_limbs: Vec<Target>,
    pub inv_limbs: Vec<Target>,
    pub div_limbs: Vec<Target>,
}

pub struct InvGroup {
    sites: Vec<InvSite>,
    /// Per-site cache of `(x, inv, div)`. Stale when `x` changes between proofs.
    cache: Mutex<Vec<Option<(BigUint, BigUint, BigUint)>>>,
}

impl InvGroup {
    fn new(sites: Vec<InvSite>) -> Self {
        let cache = Mutex::new(vec![None; sites.len()]);
        Self { sites, cache }
    }
}

/// One `QuinticQuotientGenerator` site. Cached values are the five Goldilocks
/// limbs of the denominator and of the quotient (canonical `u64`s), so the
/// registry is field-type-erased.
#[derive(Clone, Debug)]
pub struct QuinticSite {
    pub numerator: [Target; 5],
    pub denominator: [Target; 5],
    pub quotient: [Target; 5],
}

pub struct QuinticGroup {
    sites: Vec<QuinticSite>,
    /// `(den_limbs, quot_limbs)` per site.
    cache: Mutex<Vec<Option<([u64; 5], [u64; 5])>>>,
}

impl QuinticGroup {
    fn new(sites: Vec<QuinticSite>) -> Self {
        let cache = Mutex::new(vec![None; sites.len()]);
        Self { sites, cache }
    }
}

thread_local! {
    static BUILDING_INV: RefCell<Vec<InvSite>> = const { RefCell::new(Vec::new()) };
    static BUILDING_Q: RefCell<Vec<QuinticSite>> = const { RefCell::new(Vec::new()) };
}

static INV_GROUPS: Mutex<Vec<Arc<InvGroup>>> = Mutex::new(Vec::new());
static Q_GROUPS: Mutex<Vec<Arc<QuinticGroup>>> = Mutex::new(Vec::new());

/// Start a new inverse-site group on this thread (one circuit's generators).
pub fn begin_inv_group() {
    BUILDING_INV.with(|b| b.borrow_mut().clear());
    BUILDING_Q.with(|b| b.borrow_mut().clear());
}

pub fn push_inv_site(site: InvSite) {
    BUILDING_INV.with(|b| b.borrow_mut().push(site));
}

pub fn push_quintic_site(site: QuinticSite) {
    BUILDING_Q.with(|b| b.borrow_mut().push(site));
}

/// Seal the thread-local sites into a process-wide group. Safe to call if
/// nothing was pushed (no-op).
pub fn end_inv_group() {
    BUILDING_INV.with(|b| {
        let sites = std::mem::take(&mut *b.borrow_mut());
        if !sites.is_empty() {
            INV_GROUPS
                .lock()
                .expect("inv-group registry")
                .push(Arc::new(InvGroup::new(sites)));
        }
    });
    BUILDING_Q.with(|b| {
        let sites = std::mem::take(&mut *b.borrow_mut());
        if !sites.is_empty() {
            Q_GROUPS
                .lock()
                .expect("quintic-group registry")
                .push(Arc::new(QuinticGroup::new(sites)));
        }
    });
}

fn find_inv_group(x_limbs: &[Target]) -> Option<Arc<InvGroup>> {
    INV_GROUPS
        .lock()
        .ok()?
        .iter()
        .find(|g| g.sites.iter().any(|s| s.x_limbs == x_limbs))
        .cloned()
}

fn find_quintic_group(denom: &[Target; 5], quot: &[Target; 5]) -> Option<Arc<QuinticGroup>> {
    Q_GROUPS
        .lock()
        .ok()?
        .iter()
        .find(|g| {
            g.sites
                .iter()
                .any(|s| &s.denominator == denom && &s.quotient == quot)
        })
        .cloned()
}

fn try_read_biguint<F: RichField>(
    witness: &PartitionWitness<F>,
    limbs: &[Target],
) -> Option<BigUint> {
    let mut acc = BigUint::zero();
    for &limb in limbs.iter().rev() {
        let v = witness.try_get_target(limb)?;
        acc = (acc << 32) + BigUint::from(v.to_canonical_u64());
    }
    Some(acc)
}

fn try_read_u64x5<F: RichField>(
    witness: &PartitionWitness<F>,
    limbs: [Target; 5],
) -> Option<[u64; 5]> {
    Some([
        witness.try_get_target(limbs[0])?.to_canonical_u64(),
        witness.try_get_target(limbs[1])?.to_canonical_u64(),
        witness.try_get_target(limbs[2])?.to_canonical_u64(),
        witness.try_get_target(limbs[3])?.to_canonical_u64(),
        witness.try_get_target(limbs[4])?.to_canonical_u64(),
    ])
}

/// Invert `x` (and every other ready site in its circuit group) via Montgomery
/// batch inversion. Returns `(inv, div)` where `x * inv = div * |FF| + 1`.
pub fn invert_nonnative_batched<F: RichField, FF: PrimeField>(
    witness: &PartitionWitness<F>,
    x_limbs: &[Target],
    x: FF,
) -> (BigUint, BigUint) {
    if x.is_zero() {
        return (BigUint::zero(), BigUint::zero());
    }

    let Some(group) = find_inv_group(x_limbs) else {
        return single_inv_div(x);
    };

    let my_idx = match group.sites.iter().position(|s| s.x_limbs == x_limbs) {
        Some(i) => i,
        None => return single_inv_div(x),
    };

    let mut cache = group.cache.lock().expect("inv-group cache");
    if cache.len() != group.sites.len() {
        cache.resize(group.sites.len(), None);
    }

    let x_big = x.to_canonical_biguint();
    if let Some((cached_x, inv, div)) = &cache[my_idx] {
        if cached_x == &x_big {
            return (inv.clone(), div.clone());
        }
    }

    let mut ready_idx = Vec::new();
    let mut ready_x: Vec<FF> = Vec::new();
    for (i, site) in group.sites.iter().enumerate() {
        let Some(xb) = try_read_biguint(witness, &site.x_limbs) else {
            continue;
        };
        if let Some((cached_x, _, _)) = &cache[i] {
            if cached_x == &xb {
                continue;
            }
        }
        let xi = FF::from_noncanonical_biguint(xb);
        if xi.is_zero() {
            cache[i] = Some((BigUint::zero(), BigUint::zero(), BigUint::zero()));
            continue;
        }
        ready_idx.push(i);
        ready_x.push(xi);
    }

    if ready_x.is_empty() {
        return single_inv_div(x);
    }

    let invs = FF::batch_multiplicative_inverse(&ready_x);
    let modulus = FF::order();
    for (i, (xi, invi)) in ready_idx.into_iter().zip(ready_x.iter().zip(invs.iter())) {
        let xb = xi.to_canonical_biguint();
        let ib = invi.to_canonical_biguint();
        let (div, _) = (&xb * &ib).div_rem(&modulus);
        cache[i] = Some((xb, ib, div));
    }

    match &cache[my_idx] {
        Some((_, inv, div)) => (inv.clone(), div.clone()),
        None => single_inv_div(x),
    }
}

fn single_inv_div<FF: PrimeField>(x: FF) -> (BigUint, BigUint) {
    match x.try_inverse() {
        None => (BigUint::zero(), BigUint::zero()),
        Some(inv) => {
            let xb = x.to_canonical_biguint();
            let ib = inv.to_canonical_biguint();
            let (div, _) = (&xb * &ib).div_rem(&FF::order());
            (ib, div)
        }
    }
}

/// Invert the denominator of this quintic quotient (and every other ready
/// site in its group) and return `numerator * inv`.
pub fn invert_quintic_batched<F: RichField + Extendable<5>>(
    witness: &PartitionWitness<F>,
    _numerator: [Target; 5],
    denominator: [Target; 5],
    quotient: [Target; 5],
    num: QuinticExtension<F>,
    den: QuinticExtension<F>,
) -> QuinticExtension<F> {
    if den.is_zero() {
        return QuinticExtension::<F>::ZERO;
    }

    let Some(group) = find_quintic_group(&denominator, &quotient) else {
        return match den.try_inverse() {
            None => QuinticExtension::<F>::ZERO,
            Some(inv) => num * inv,
        };
    };

    let my_idx = match group
        .sites
        .iter()
        .position(|s| s.denominator == denominator && s.quotient == quotient)
    {
        Some(i) => i,
        None => {
            return match den.try_inverse() {
                None => QuinticExtension::<F>::ZERO,
                Some(inv) => num * inv,
            };
        }
    };

    let den_arr: [F; 5] = den.to_basefield_array();
    let den_limbs = [
        den_arr[0].to_canonical_u64(),
        den_arr[1].to_canonical_u64(),
        den_arr[2].to_canonical_u64(),
        den_arr[3].to_canonical_u64(),
        den_arr[4].to_canonical_u64(),
    ];

    let mut cache = group.cache.lock().expect("quintic-group cache");
    if cache.len() != group.sites.len() {
        cache.resize(group.sites.len(), None);
    }

    if let Some((cached_den, cached_q)) = &cache[my_idx] {
        if cached_den == &den_limbs {
            return QuinticExtension::<F>::from_basefield_array([
                F::from_canonical_u64(cached_q[0]),
                F::from_canonical_u64(cached_q[1]),
                F::from_canonical_u64(cached_q[2]),
                F::from_canonical_u64(cached_q[3]),
                F::from_canonical_u64(cached_q[4]),
            ]);
        }
    }

    let mut ready_idx = Vec::new();
    let mut ready_den = Vec::new();
    let mut ready_num = Vec::new();
    for (i, site) in group.sites.iter().enumerate() {
        let Some(d_limbs) = try_read_u64x5(witness, site.denominator) else {
            continue;
        };
        let Some(n_limbs) = try_read_u64x5(witness, site.numerator) else {
            continue;
        };
        if let Some((cached_d, _)) = &cache[i] {
            if cached_d == &d_limbs {
                continue;
            }
        }
        let d = QuinticExtension::<F>::from_basefield_array([
            F::from_canonical_u64(d_limbs[0]),
            F::from_canonical_u64(d_limbs[1]),
            F::from_canonical_u64(d_limbs[2]),
            F::from_canonical_u64(d_limbs[3]),
            F::from_canonical_u64(d_limbs[4]),
        ]);
        let n = QuinticExtension::<F>::from_basefield_array([
            F::from_canonical_u64(n_limbs[0]),
            F::from_canonical_u64(n_limbs[1]),
            F::from_canonical_u64(n_limbs[2]),
            F::from_canonical_u64(n_limbs[3]),
            F::from_canonical_u64(n_limbs[4]),
        ]);
        if d.is_zero() {
            cache[i] = Some((d_limbs, [0; 5]));
            continue;
        }
        ready_idx.push(i);
        ready_den.push(d);
        ready_num.push(n);
    }

    if ready_den.is_empty() {
        return match den.try_inverse() {
            None => QuinticExtension::<F>::ZERO,
            Some(inv) => num * inv,
        };
    }

    let invs = QuinticExtension::<F>::batch_multiplicative_inverse(&ready_den);
    for (i, ((d, n), inv)) in ready_idx
        .into_iter()
        .zip(ready_den.iter().zip(ready_num.iter()).zip(invs.iter()))
    {
        let q = *n * *inv;
        let q_arr: [F; 5] = q.to_basefield_array();
        let d_arr: [F; 5] = d.to_basefield_array();
        cache[i] = Some((
            [
                d_arr[0].to_canonical_u64(),
                d_arr[1].to_canonical_u64(),
                d_arr[2].to_canonical_u64(),
                d_arr[3].to_canonical_u64(),
                d_arr[4].to_canonical_u64(),
            ],
            [
                q_arr[0].to_canonical_u64(),
                q_arr[1].to_canonical_u64(),
                q_arr[2].to_canonical_u64(),
                q_arr[3].to_canonical_u64(),
                q_arr[4].to_canonical_u64(),
            ],
        ));
    }

    match &cache[my_idx] {
        Some((_, q)) => QuinticExtension::<F>::from_basefield_array([
            F::from_canonical_u64(q[0]),
            F::from_canonical_u64(q[1]),
            F::from_canonical_u64(q[2]),
            F::from_canonical_u64(q[3]),
            F::from_canonical_u64(q[4]),
        ]),
        None => match den.try_inverse() {
            None => QuinticExtension::<F>::ZERO,
            Some(inv) => num * inv,
        },
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::secp256k1_base::Secp256K1Base;
    use plonky2::field::types::{Field, Sample};

    #[test]
    fn batch_inverse_matches_individual_secp() {
        let xs: Vec<Secp256K1Base> = (0..16).map(|_| Secp256K1Base::rand()).collect();
        let batched = Secp256K1Base::batch_multiplicative_inverse(&xs);
        for (x, inv) in xs.iter().zip(batched.iter()) {
            assert_eq!(*x * *inv, Secp256K1Base::ONE);
            assert_eq!(*inv, x.inverse());
        }
    }
}

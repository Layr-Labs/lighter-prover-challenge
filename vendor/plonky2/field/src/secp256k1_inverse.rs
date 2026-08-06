use core::cmp::Ordering;

type U256 = [u64; 4];

#[inline]
fn cmp(a: &U256, b: &U256) -> Ordering {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

#[inline]
fn is_zero(a: &U256) -> bool {
    a.iter().all(|&limb| limb == 0)
}

#[inline]
fn sub_assign(a: &mut U256, b: &U256) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (difference, borrow_1) = a[i].overflowing_sub(b[i]);
        let (difference, borrow_2) = difference.overflowing_sub(borrow);
        a[i] = difference;
        borrow = u64::from(borrow_1 || borrow_2);
    }
    debug_assert_eq!(borrow, 0);
}

#[inline]
fn add_assign(a: &mut U256, b: &U256) -> u64 {
    let mut carry = 0u64;
    for i in 0..4 {
        let sum = a[i] as u128 + b[i] as u128 + carry as u128;
        a[i] = sum as u64;
        carry = (sum >> 64) as u64;
    }
    carry
}

#[inline]
fn shift_right_one(a: &mut U256, high_bit: u64) {
    debug_assert!(high_bit <= 1);
    a[0] = (a[0] >> 1) | (a[1] << 63);
    a[1] = (a[1] >> 1) | (a[2] << 63);
    a[2] = (a[2] >> 1) | (a[3] << 63);
    a[3] = (a[3] >> 1) | (high_bit << 63);
}

#[inline]
fn sub_mod_assign(a: &mut U256, b: &U256, modulus: &U256) {
    if cmp(a, b) != Ordering::Less {
        sub_assign(a, b);
    } else {
        let mut difference = *b;
        sub_assign(&mut difference, a);
        *a = *modulus;
        sub_assign(a, &difference);
    }
}

/// Inverts a value modulo an odd 256-bit prime with binary extended Euclid.
///
/// Both secp256k1 moduli are larger than `2^255`, so an arbitrary 256-bit
/// representation needs at most one subtraction to become canonical.
pub(crate) fn mod_inverse(mut value: U256, modulus: U256) -> Option<U256> {
    debug_assert_eq!(modulus[0] & 1, 1);
    if cmp(&value, &modulus) != Ordering::Less {
        sub_assign(&mut value, &modulus);
    }
    if is_zero(&value) {
        return None;
    }

    let one = [1, 0, 0, 0];
    let mut u = value;
    let mut v = modulus;
    let mut x_u = one;
    let mut x_v = [0; 4];

    while u != one && v != one {
        while u[0] & 1 == 0 {
            shift_right_one(&mut u, 0);
            if x_u[0] & 1 == 0 {
                shift_right_one(&mut x_u, 0);
            } else {
                let carry = add_assign(&mut x_u, &modulus);
                shift_right_one(&mut x_u, carry);
            }
        }
        while v[0] & 1 == 0 {
            shift_right_one(&mut v, 0);
            if x_v[0] & 1 == 0 {
                shift_right_one(&mut x_v, 0);
            } else {
                let carry = add_assign(&mut x_v, &modulus);
                shift_right_one(&mut x_v, carry);
            }
        }
        if cmp(&u, &v) != Ordering::Less {
            sub_assign(&mut u, &v);
            sub_mod_assign(&mut x_u, &x_v, &modulus);
        } else {
            sub_assign(&mut v, &u);
            sub_mod_assign(&mut x_v, &x_u, &modulus);
        }
    }

    Some(if u == one { x_u } else { x_v })
}

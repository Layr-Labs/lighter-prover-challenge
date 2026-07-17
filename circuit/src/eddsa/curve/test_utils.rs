// Portions of this file are derived from ecgfp5
// Copyright (c) 2022 Thomas Pornin
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::Sample;
use rand::thread_rng;

use super::base_field::{Sgn0, SquareRoot};
use crate::types::config::F;

pub fn gfp5_random_non_square() -> QuinticExtension<F> {
    let mut rng = thread_rng();
    loop {
        let attempt = QuinticExtension::<F>::sample(&mut rng);
        if attempt.sqrt().is_none() {
            return attempt;
        }
    }
}

pub fn gfp5_random_sgn0_eq_0() -> QuinticExtension<F> {
    let mut rng = thread_rng();
    loop {
        let attempt = QuinticExtension::<F>::sample(&mut rng);
        if !attempt.sgn0() {
            return attempt;
        }
    }
}

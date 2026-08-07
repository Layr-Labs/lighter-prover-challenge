//! Scratch profiling harness for the zero-padded LDE FFT path. Not part of the build.
use std::time::Instant;

use plonky2_field::batch_util::batch_multiply_inplace;
use plonky2_field::fft::{fft_root_table, FftRootTable};
use plonky2_field::goldilocks_field::GoldilocksField as F;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::Field;
use plonky2_util::reverse_index_bits_in_place;

fn vals(n: usize) -> Vec<F> {
    (0..n)
        .map(|i| F::from_canonical_u64((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 3))
        .collect()
}

fn main() {
    const RATE: usize = 3;
    for degree_bits in [12usize, 13, 14, 15] {
        let degree = 1usize << degree_bits;
        let lde_len = degree << RATE;
        let lg_n = degree_bits + RATE;
        let table: FftRootTable<F> = fft_root_table(lde_len);
        let coeffs = vals(degree);
        let shift = F::coset_shift();
        let coset_powers: Vec<F> = shift.powers().take(degree).collect();

        let reps = 200usize;

        // 1. full LDE (what lde_values does per column)
        let t = Instant::now();
        let mut sink = F::ZERO;
        for _ in 0..reps {
            let mut buffer = Vec::with_capacity(lde_len);
            buffer.extend_from_slice(&coeffs);
            unsafe { buffer.set_len(lde_len) };
            batch_multiply_inplace(&mut buffer[..degree], &coset_powers);
            let v = PolynomialCoeffs::new(buffer)
                .fft_with_options(Some(RATE), Some(&table))
                .values;
            sink += v[7];
        }
        let full = t.elapsed().as_secs_f64() / reps as f64;

        // 2. just the copy
        let t = Instant::now();
        for _ in 0..reps {
            let mut buffer = Vec::with_capacity(lde_len);
            buffer.extend_from_slice(&coeffs);
            unsafe { buffer.set_len(lde_len) };
            sink += buffer[3];
            std::hint::black_box(&buffer);
        }
        let copy = t.elapsed().as_secs_f64() / reps as f64;

        // 3. copy + coset multiply
        let t = Instant::now();
        for _ in 0..reps {
            let mut buffer = Vec::with_capacity(lde_len);
            buffer.extend_from_slice(&coeffs);
            unsafe { buffer.set_len(lde_len) };
            batch_multiply_inplace(&mut buffer[..degree], &coset_powers);
            sink += buffer[3];
            std::hint::black_box(&buffer);
        }
        let copy_mul = t.elapsed().as_secs_f64() / reps as f64;

        // 4. copy + coset multiply + bit reverse of live prefix
        let t = Instant::now();
        for _ in 0..reps {
            let mut buffer = Vec::with_capacity(lde_len);
            buffer.extend_from_slice(&coeffs);
            unsafe { buffer.set_len(lde_len) };
            batch_multiply_inplace(&mut buffer[..degree], &coset_powers);
            reverse_index_bits_in_place(&mut buffer[..degree]);
            sink += buffer[3];
            std::hint::black_box(&buffer);
        }
        let copy_mul_rev = t.elapsed().as_secs_f64() / reps as f64;

        println!(
            "degree 2^{degree_bits} -> lde 2^{lg_n}: full {:8.1}us | copy {:6.1} | +cosetmul {:6.1} | +bitrev {:6.1} | prep-share {:.1}%",
            full * 1e6,
            copy * 1e6,
            copy_mul * 1e6,
            copy_mul_rev * 1e6,
            100.0 * copy_mul_rev / full
        );
        std::hint::black_box(sink);
    }
}

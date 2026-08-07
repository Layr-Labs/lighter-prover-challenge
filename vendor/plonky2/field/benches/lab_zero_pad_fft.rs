//! LOCAL-ONLY lab bench for the zero-padded FFT expansion pipeline.
//! Never part of a submission.
//!
//! Exercises the EXACT production per-polynomial LDE pipeline:
//!   base field  (`PolynomialBatch::lde_values`): LDE-sized buffer, coeffs
//!     copied in, tail left uninitialized (set_len), packed coset-power
//!     multiply over the prefix, zero-padded FFT with a precomputed table.
//!   extension   (`coset_fft_zero_tail`): serial shift-powers scale of the
//!     live prefix into a fresh buffer, tail left uninitialized, same FFT.
//!
//! Production shapes only: degree 2^14 -> LDE 2^17, 2^15 -> 2^18,
//! 2^18 -> 2^21 (r = 3), Goldilocks base and quadratic extension.
//!
//! Modes:
//!   identity <reps> [--arms a,b] [--shape N]
//!       bit-identity: each selected variant vs the production baseline,
//!       raw u64 words AND canonical values, over <reps> random inputs
//!       per shape per field.
//!   time <reps> [--arms a,b] [--shape N]
//!       interleaved ABAB...: per-shape, per-arm medians vs baseline.

use std::time::Instant;

use plonky2_field::batch_util::batch_multiply_inplace;
use plonky2_field::extension::quadratic::QuadraticExtension;
use plonky2_field::fft::{FftRootTable, fft_root_table, lab};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::{Field, PrimeField64};

type F = GoldilocksField;
type FE = QuadraticExtension<GoldilocksField>;

const RATE_BITS: usize = 3;
const SHAPES: [(usize, usize); 3] = [(14, 17), (15, 18), (18, 21)];

// ---------------------------------------------------------------- PRNG

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn random_base(rng: &mut SplitMix64, n: usize) -> Vec<F> {
    (0..n)
        .map(|_| F::from_noncanonical_u64(rng.next()))
        .collect()
}

fn random_ext(rng: &mut SplitMix64, n: usize) -> Vec<FE> {
    (0..n)
        .map(|_| {
            QuadraticExtension([
                F::from_noncanonical_u64(rng.next()),
                F::from_noncanonical_u64(rng.next()),
            ])
        })
        .collect()
}

// ---------------------------------------------------------- pipelines

/// Exact replica of the `lde_values` per-polynomial buffer discipline.
fn uninit_tail_buffer<T: Field>(coeffs: &[T], lde_len: usize) -> Vec<T> {
    let mut buffer = Vec::with_capacity(lde_len);
    buffer.extend_from_slice(coeffs);
    // SAFETY: identical invariant to production `lde_values` /
    // `coset_fft_zero_tail`: with r > 0 and degree >= 2 the zero-padded FFT
    // reads only the live prefix and writes every tail element before
    // reading it.
    unsafe { buffer.set_len(lde_len) };
    buffer
}

type Arm<'a, T> = (String, Box<dyn Fn(&[T]) -> Vec<T> + Sync + 'a>);

/// Base-field arm registry. Every arm consumes raw coefficients and returns
/// the LDE values, replicating the full production per-poly pipeline
/// (including the coset-power multiply for non-folded arms).
fn base_arms<'a>(
    table: &'a FftRootTable<F>,
    folded: &'a FftRootTable<F>,
    powers: &'a [F],
    shift: F,
) -> Vec<Arm<'a, F>> {
    let classic = move |coeffs: &[F], fft: fn(&mut [F], usize, &FftRootTable<F>)| {
        let mut buffer = uninit_tail_buffer(coeffs, coeffs.len() << RATE_BITS);
        batch_multiply_inplace(&mut buffer[..coeffs.len()], powers);
        fft(&mut buffer, RATE_BITS, table);
        buffer
    };
    let folded_arm = move |coeffs: &[F],
                           fft: fn(&mut [F], usize, &FftRootTable<F>, F)| {
        let mut buffer = uninit_tail_buffer(coeffs, coeffs.len() << RATE_BITS);
        fft(&mut buffer, RATE_BITS, folded, shift);
        buffer
    };
    vec![
        ("baseline".into(), Box::new(move |c: &[F]| classic(c, lab::fft_classic_baseline::<F>)) as _),
        ("a1_fused_expand2".into(), Box::new(move |c: &[F]| classic(c, lab::fft_classic_a1::<F>)) as _),
        ("a2_triple_layers".into(), Box::new(move |c: &[F]| classic(c, lab::fft_classic_a2::<F>)) as _),
        ("a12_both".into(), Box::new(move |c: &[F]| classic(c, lab::fft_classic_a12::<F>)) as _),
        ("blkm1".into(), Box::new(move |c: &[F]| classic(c, |v, r, t| lab::fft_classic_block_size(v, r, t, 12))) as _),
        ("blkp1".into(), Box::new(move |c: &[F]| classic(c, |v, r, t| lab::fft_classic_block_size(v, r, t, 14))) as _),
        ("a1_blkm1".into(), Box::new(move |c: &[F]| classic(c, lab::fft_classic_a1_blkm1::<F>)) as _),
        ("b_coset_folded".into(), Box::new(move |c: &[F]| folded_arm(c, lab::fft_classic_coset_folded::<F>)) as _),
        ("b_a12_coset_folded".into(), Box::new(move |c: &[F]| folded_arm(c, lab::fft_classic_coset_folded_a12::<F>)) as _),
        ("combo_b_a1".into(), Box::new(move |c: &[F]| folded_arm(c, lab::fft_classic_coset_a1::<F>)) as _),
        ("combo_b_a1_blkm1".into(), Box::new(move |c: &[F]| folded_arm(c, lab::fft_classic_coset_a1_blkm1::<F>)) as _),
        ("combo_b_a1_blkp1".into(), Box::new(move |c: &[F]| folded_arm(c, lab::fft_classic_coset_a1_blkp1::<F>)) as _),
        ("combo_b_blkp1".into(), Box::new(move |c: &[F]| folded_arm(c, lab::fft_classic_coset_blkp1::<F>)) as _),
    ]
}

/// Extension-field arm registry, mirroring `coset_fft_zero_tail`.
fn ext_arms<'a>(
    table: &'a FftRootTable<FE>,
    folded: &'a FftRootTable<FE>,
    shift: FE,
) -> Vec<Arm<'a, FE>> {
    let classic = move |coeffs: &[FE], fft: fn(&mut [FE], usize, &FftRootTable<FE>)| {
        let lde_len = coeffs.len() << RATE_BITS;
        let mut scaled = Vec::with_capacity(lde_len);
        scaled.extend(shift.powers().zip(coeffs).map(|(r, &c)| r * c));
        unsafe { scaled.set_len(lde_len) };
        fft(&mut scaled, RATE_BITS, table);
        scaled
    };
    let folded_arm = move |coeffs: &[FE],
                           fft: fn(&mut [FE], usize, &FftRootTable<FE>, FE)| {
        let mut buffer = uninit_tail_buffer(coeffs, coeffs.len() << RATE_BITS);
        fft(&mut buffer, RATE_BITS, folded, shift);
        buffer
    };
    vec![
        ("baseline".into(), Box::new(move |c: &[FE]| classic(c, lab::fft_classic_baseline::<FE>)) as _),
        ("a1_fused_expand2".into(), Box::new(move |c: &[FE]| classic(c, lab::fft_classic_a1::<FE>)) as _),
        ("a2_triple_layers".into(), Box::new(move |c: &[FE]| classic(c, lab::fft_classic_a2::<FE>)) as _),
        ("a12_both".into(), Box::new(move |c: &[FE]| classic(c, lab::fft_classic_a12::<FE>)) as _),
        ("blkm1".into(), Box::new(move |c: &[FE]| classic(c, |v, r, t| lab::fft_classic_block_size(v, r, t, 11))) as _),
        ("blkp1".into(), Box::new(move |c: &[FE]| classic(c, |v, r, t| lab::fft_classic_block_size(v, r, t, 13))) as _),
        ("a1_blkm1".into(), Box::new(move |c: &[FE]| classic(c, lab::fft_classic_a1_blkm1::<FE>)) as _),
        ("b_coset_folded".into(), Box::new(move |c: &[FE]| folded_arm(c, lab::fft_classic_coset_folded::<FE>)) as _),
        ("b_a12_coset_folded".into(), Box::new(move |c: &[FE]| folded_arm(c, lab::fft_classic_coset_folded_a12::<FE>)) as _),
        ("combo_b_a1".into(), Box::new(move |c: &[FE]| folded_arm(c, lab::fft_classic_coset_a1::<FE>)) as _),
        ("combo_b_a1_blkm1".into(), Box::new(move |c: &[FE]| folded_arm(c, lab::fft_classic_coset_a1_blkm1::<FE>)) as _),
        ("combo_b_a1_blkp1".into(), Box::new(move |c: &[FE]| folded_arm(c, lab::fft_classic_coset_a1_blkp1::<FE>)) as _),
        ("combo_b_blkp1".into(), Box::new(move |c: &[FE]| folded_arm(c, lab::fft_classic_coset_blkp1::<FE>)) as _),
    ]
}

fn keep(name: &str, filter: &Option<Vec<String>>) -> bool {
    match filter {
        None => true,
        Some(subs) => name == "baseline" || subs.iter().any(|s| name.contains(s.as_str())),
    }
}

// ------------------------------------------------------------- identity

fn raw_base(v: &[F]) -> Vec<u64> {
    v.iter().map(|x| x.0).collect()
}
fn canon_base(v: &[F]) -> Vec<u64> {
    v.iter().map(|x| x.to_canonical_u64()).collect()
}
fn raw_ext(v: &[FE]) -> Vec<u64> {
    v.iter().flat_map(|x| [x.0[0].0, x.0[1].0]).collect()
}
fn canon_ext(v: &[FE]) -> Vec<u64> {
    v.iter()
        .flat_map(|x| [x.0[0].to_canonical_u64(), x.0[1].to_canonical_u64()])
        .collect()
}

fn identity_field<T: Field + Sync>(
    label: &str,
    lg_deg: usize,
    lg_lde: usize,
    reps: usize,
    arms: &[Arm<'_, T>],
    gen: impl Fn(&mut SplitMix64, usize) -> Vec<T> + Sync,
    raw: impl Fn(&[T]) -> Vec<u64> + Sync,
    canon: impl Fn(&[T]) -> Vec<u64> + Sync,
    seed_hi: u64,
) {
    let degree = 1usize << lg_deg;
    let n_arms = arms.len() - 1; // arms[0] is baseline
    let n_threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    let tallies: Vec<Vec<(u64, u64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let gen = &gen;
                let raw = &raw;
                let canon = &canon;
                s.spawn(move || {
                    let mut local = vec![(0u64, 0u64); n_arms];
                    let mut rep = t;
                    while rep < reps {
                        let mut rng = SplitMix64(seed_hi ^ ((lg_lde as u64) << 32) ^ rep as u64);
                        let coeffs = gen(&mut rng, degree);
                        let expect = (arms[0].1)(&coeffs);
                        let e_raw = raw(&expect);
                        let e_canon = canon(&expect);
                        for (i, (_, arm)) in arms[1..].iter().enumerate() {
                            let got = arm(&coeffs);
                            if raw(&got) != e_raw {
                                local[i].0 += 1;
                            }
                            if canon(&got) != e_canon {
                                local[i].1 += 1;
                            }
                        }
                        rep += n_threads;
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for (i, (name, _)) in arms[1..].iter().enumerate() {
        let r: u64 = tallies.iter().map(|t| t[i].0).sum();
        let c: u64 = tallies.iter().map(|t| t[i].1).sum();
        println!(
            "identity {label} 2^{lg_deg}->2^{lg_lde} {name}: raw_mismatch={r}/{reps} canon_mismatch={c}/{reps}"
        );
    }
}

// --------------------------------------------------------------- timing

fn stats(mut xs: Vec<f64>) -> (f64, f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let median = if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    };
    let q1 = xs[n / 4];
    let q3 = xs[(3 * n) / 4];
    (median, xs[0], xs[n - 1], q3 - q1)
}

fn time_field<T: Field>(
    label: &str,
    lg_deg: usize,
    lg_lde: usize,
    reps: usize,
    arms: &[Arm<'_, T>],
    gen: impl Fn(&mut SplitMix64, usize) -> Vec<T>,
    seed_hi: u64,
) {
    let degree = 1usize << lg_deg;
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for rep in 0..reps + 1 {
        let mut rng = SplitMix64(seed_hi ^ ((lg_lde as u64) << 40) ^ rep as u64);
        let coeffs = gen(&mut rng, degree);
        for (i, (_, arm)) in arms.iter().enumerate() {
            let t0 = Instant::now();
            let out = arm(&coeffs);
            let dt = t0.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            drop(out);
            if rep > 0 {
                samples[i].push(dt * 1e3);
            }
        }
    }
    let (base_med, _, _, _) = stats(samples[0].clone());
    println!("--- time {label} 2^{lg_deg}->2^{lg_lde} ({reps} reps) ---");
    for (i, (name, _)) in arms.iter().enumerate() {
        let (med, min, max, iqr) = stats(samples[i].clone());
        println!(
            "  {name:<20} median {med:9.4} ms  min {min:9.4}  max {max:9.4}  iqr {iqr:8.4}  vs-baseline {:+7.2}%",
            (med / base_med - 1.0) * 100.0
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("time");
    let mode = if mode.starts_with('-') { "time" } else { mode };
    let reps: usize = args
        .iter()
        .skip(2)
        .take_while(|a| !a.starts_with("--"))
        .find_map(|a| a.parse().ok())
        .unwrap_or(if mode == "identity" { 1000 } else { 15 });
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .cloned()
    };
    let only: Option<usize> = flag("--shape").and_then(|s| s.parse().ok());
    let arm_filter: Option<Vec<String>> =
        flag("--arms").map(|s| s.split(',').map(|x| x.trim().to_string()).collect());

    for (lg_deg, lg_lde) in SHAPES {
        if let Some(o) = only {
            if o != lg_deg {
                continue;
            }
        }
        let lde_len = 1usize << lg_lde;
        let degree = 1usize << lg_deg;
        let table_b = fft_root_table::<F>(lde_len);
        let table_e = fft_root_table::<FE>(lde_len);
        let shift_b = F::coset_shift();
        let shift_e: FE = F::coset_shift().into();
        let folded_b = lab::coset_folded_root_table::<F>(lde_len, shift_b);
        let folded_e = lab::coset_folded_root_table::<FE>(lde_len, shift_e);
        let powers_b: Vec<F> = shift_b.powers().take(degree).collect();

        let arms_b: Vec<Arm<'_, F>> = base_arms(&table_b, &folded_b, &powers_b, shift_b)
            .into_iter()
            .filter(|(n, _)| keep(n, &arm_filter))
            .collect();
        let arms_e: Vec<Arm<'_, FE>> = ext_arms(&table_e, &folded_e, shift_e)
            .into_iter()
            .filter(|(n, _)| keep(n, &arm_filter))
            .collect();

        match mode {
            "identity" => {
                identity_field(
                    "base", lg_deg, lg_lde, reps, &arms_b, random_base, raw_base, canon_base,
                    0x1234_5678_0000_0000,
                );
                identity_field(
                    "ext ", lg_deg, lg_lde, reps, &arms_e, random_ext, raw_ext, canon_ext,
                    0xabcd_ef01_0000_0000,
                );
            }
            _ => {
                time_field("base", lg_deg, lg_lde, reps, &arms_b, random_base, 0x5555_0000);
                time_field("ext ", lg_deg, lg_lde, reps, &arms_e, random_ext, 0x5556_0000);
            }
        }
    }
}

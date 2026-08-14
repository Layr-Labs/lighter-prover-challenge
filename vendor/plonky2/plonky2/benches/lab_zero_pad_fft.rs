// Local-only screen for the zero-padded FFT lab variants in
// `plonky2_field::fft::lab`. The production path is `fft_classic` (re-exported
// by the lab module as `fft_classic_baseline`); the lab variants are candidate
// rewrites of that path and must be (a) bit-identical to it and (b) faster in
// isolation before they are wired into the production zero-padded FFT.
//
// Every arm is timed through the same shim: a fresh zero-padded buffer
// (first `n >> r` coefficients random, the rest zero), transformed in place.
// Bit-identity is checked once per (field, size, r) BEFORE any timing; a
// mismatch panics and fails the run. Coset-folded variants are checked against
// the equivalent production sequence `scale by shift.powers(); fft_classic`.
//
// This bench target is deliberately NOT referenced by the ranked build
// (`cargo build -p bench --bin prove` compiles only the worker); it exists so
// the lab module's variants can be screened locally before a full A/B.
//
// Run from the vendored workspace:
//   RUSTFLAGS="-C target-cpu=native" cargo bench --release -p plonky2 \
//       --bench lab_zero_pad_fft -- --quick

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::fft::lab;
use plonky2::field::fft::{FftRootTable, fft_root_table};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Sample};

/// A zero-padded coefficient buffer: `n` slots, first `n >> r` random.
fn zero_padded_input<F: Field>(n: usize, r: usize) -> Vec<F> {
    let mut v = F::rand_vec(n >> r);
    v.resize(n, F::ZERO);
    v
}

/// Scale raw coefficients by the coset-shift powers: the production
/// `PolynomialCoeffs::coset_fft_with_options` pre-pass.
fn scale_by_shift_powers<F: Field>(raw: &[F], shift: F) -> Vec<F> {
    shift.powers().zip(raw).map(|(p, &c)| p * c).collect()
}

/// Assert every plain (standard root table) lab variant is bit-identical to
/// the production `fft_classic` baseline.
fn verify_plain<F: Field>(n: usize, r: usize, table: &FftRootTable<F>) {
    let src = zero_padded_input::<F>(n, r);

    let mut base = src.clone();
    lab::fft_classic_baseline(&mut base, r, table);

    let mut a1 = src.clone();
    lab::fft_classic_a1(&mut a1, r, table);
    assert_eq!(a1, base, "A1 mismatch n={n} r={r}");

    let mut a2 = src.clone();
    lab::fft_classic_a2(&mut a2, r, table);
    assert_eq!(a2, base, "A2 mismatch n={n} r={r}");

    let mut a12 = src.clone();
    lab::fft_classic_a12(&mut a12, r, table);
    assert_eq!(a12, base, "A12 mismatch n={n} r={r}");
}

/// Assert the coset-folded variants are bit-identical to the production
/// `scale by shift.powers(); fft_classic` sequence.
fn verify_coset<F: Field>(n: usize, r: usize, shift: F) {
    let plain = fft_root_table::<F>(n);
    let folded = lab::coset_folded_root_table::<F>(n, shift);
    let raw = zero_padded_input::<F>(n, r);

    let mut base = scale_by_shift_powers::<F>(&raw, shift);
    lab::fft_classic_baseline(&mut base, r, &plain);

    let mut b = raw.clone();
    lab::fft_classic_coset_folded(&mut b, r, &folded, shift);
    assert_eq!(b, base, "B mismatch n={n} r={r}");

    let mut ba12 = raw.clone();
    lab::fft_classic_coset_folded_a12(&mut ba12, r, &folded, shift);
    assert_eq!(ba12, base, "B+A12 mismatch n={n} r={r}");

    let mut ca1 = raw.clone();
    lab::fft_classic_coset_a1(&mut ca1, r, &folded, shift);
    assert_eq!(ca1, base, "coset+A1 mismatch n={n} r={r}");
}

fn bench_field<F: Field>(c: &mut Criterion, label: &str, sizes: &[usize], rates: &[usize]) {
    for &n in sizes {
        let table = fft_root_table::<F>(n);
        for &r in rates {
            if r == 0 || r >= n.trailing_zeros() as usize {
                continue;
            }
            // Bit-identity gate first: panic => the bench never times garbage.
            verify_plain::<F>(n, r, &table);
            let shift = F::coset_shift();
            verify_coset::<F>(n, r, shift);
            let folded = lab::coset_folded_root_table::<F>(n, shift);

            let name = format!("{label}/n={n}/r={r}");

            c.bench_with_input(BenchmarkId::new("baseline", &name), &n, |b, _| {
                b.iter_batched(
                    || zero_padded_input::<F>(n, r),
                    |mut v| lab::fft_classic_baseline(&mut v, r, &table),
                    BatchSize::LargeInput,
                )
            });
            c.bench_with_input(BenchmarkId::new("A1", &name), &n, |b, _| {
                b.iter_batched(
                    || zero_padded_input::<F>(n, r),
                    |mut v| lab::fft_classic_a1(&mut v, r, &table),
                    BatchSize::LargeInput,
                )
            });
            c.bench_with_input(BenchmarkId::new("A2", &name), &n, |b, _| {
                b.iter_batched(
                    || zero_padded_input::<F>(n, r),
                    |mut v| lab::fft_classic_a2(&mut v, r, &table),
                    BatchSize::LargeInput,
                )
            });
            c.bench_with_input(BenchmarkId::new("A12", &name), &n, |b, _| {
                b.iter_batched(
                    || zero_padded_input::<F>(n, r),
                    |mut v| lab::fft_classic_a12(&mut v, r, &table),
                    BatchSize::LargeInput,
                )
            });
            c.bench_with_input(BenchmarkId::new("B", &name), &n, |b, _| {
                b.iter_batched(
                    || zero_padded_input::<F>(n, r),
                    |mut v| lab::fft_classic_coset_folded(&mut v, r, &folded, shift),
                    BatchSize::LargeInput,
                )
            });
            c.bench_with_input(BenchmarkId::new("B+A12", &name), &n, |b, _| {
                b.iter_batched(
                    || zero_padded_input::<F>(n, r),
                    |mut v| lab::fft_classic_coset_folded_a12(&mut v, r, &folded, shift),
                    BatchSize::LargeInput,
                )
            });
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_field::<GoldilocksField>(
        c,
        "gf",
        &[1 << 15, 1 << 18, 1 << 21],
        &[2, 3, 4, 6],
    );
    bench_field::<QuadraticExtension<GoldilocksField>>(
        c,
        "ext2",
        &[1 << 13, 1 << 16, 1 << 19],
        &[2, 3, 4],
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

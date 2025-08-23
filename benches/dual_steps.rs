use ark_bn254::Fr;
use ark_ff::PrimeField;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use rand::thread_rng;
use server_aided_SNARK::emsm::raa_code::TOperator;
use server_aided_SNARK::emsm::sparse_vec::SparseVector;

// Bench `into_dense`
fn bench_into_dense<F: PrimeField>(c: &mut Criterion, noise: SparseVector<F>) {
    c.bench_function("SparseVector::into_dense", |b| {
        b.iter(|| {
            let dense = black_box(noise.clone()).into_dense();
            black_box(dense)
        });
    });
}

// Bench `multiply_sparse`
fn bench_multiply_sparse<F: PrimeField>(c: &mut Criterion, t_operator: &TOperator<F>, noise: SparseVector<F>) {
    // Precompute dense outside the loop, since we only want to measure multiply
    let noise_dense = noise.clone().into_dense();

    c.bench_function("TOperator::multiply_sparse", |b| {
        b.iter(|| {
            let lpn_vector: Vec<F> = t_operator.multiply_sparse(black_box(noise_dense.clone()));
            black_box(lpn_vector);
        });
    });
}

// Criterion entry
pub fn criterion_benchmark(c: &mut Criterion) {
    let params = vec![
        (1 << 10, 2 * 311usize),
        (1 << 11, 2 * 308),
        (1 << 12, 2 * 304),
        (1 << 13, 2 * 302),
        (1 << 14, 2 * 301),
        (1 << 15, 2 * 294),
        (1 << 16, 2 * 291),
        (1 << 17, 2 * 287),
        (1 << 18, 2 * 284),
        (1 << 19, 2 * 280),
        (1 << 20, 2 * 277),
        (1 << 21, 2 * 273),
        (1 << 22, 2 * 270),
        (1 << 23, 2 * 266),
        (1 << 24, 2 * 263),
        (1 << 25, 2 * 259),
    ];

    for &(n, d) in &params {
        let N = 4 * n;

        // Setup input (replace with real parameters)
        let t_operator: TOperator<_> = TOperator::<Fr>::rand(n);
        let noise: SparseVector<_> = SparseVector::error_vec(N, d, &mut thread_rng());

        bench_into_dense(c, noise.clone());
        bench_multiply_sparse(c, &t_operator, noise);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use rand::thread_rng;

use ark_bn254::Fr as F;
use server_aided_SNARK::emsm::lpn::dual_lpn::{DualLPNIndex, DualLPNInstance};
use server_aided_SNARK::emsm::sparse_vec::sparse_vec::SparseVector;

fn bench_dual_lpn(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_lpn");
    group.sample_size(10); // Reduce noise and runtime

    let params = vec![
        (1 << 10, 62usize),
        (1 << 11, 60),
        (1 << 12, 58),
        (1 << 13, 55),
        (1 << 14, 53),
        (1 << 15, 50),
        (1 << 16, 47),
        (1 << 17, 45),
        (1 << 18, 42),
        (1 << 19, 40),
        (1 << 20, 40),
    ];

    for &t in &[8, 10] {
        for &(n, d) in &params {
            let N = 4 * n;
            let log_n = usize::BITS - (n as usize).leading_zeros() - 1;
            let bench_id = BenchmarkId::new(format!("bench for t={}, n=2^{}", t, log_n), "");

            group.bench_with_input(bench_id, &n, |b, &_n| {
                b.iter(|| {
                    let rng = &mut thread_rng();
                    let index = DualLPNIndex::<F>::new(rng, n, N, t);
                    let error = SparseVector::error_vec(N, d, rng);
                    let _instance = DualLPNInstance::new(&index, error);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_dual_lpn);
criterion_main!(benches);

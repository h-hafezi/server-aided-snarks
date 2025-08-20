use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use rand::thread_rng;

use ark_bn254::Fr as F;
use server_aided_SNARK::emsm::dual_lpn::{DualLPNInstance};
use server_aided_SNARK::emsm::raa_code::TOperator;
use server_aided_SNARK::emsm::sparse_vec::SparseVector;

fn bench_dual_lpn(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_lpn");
    group.sample_size(10); // Reduce noise and runtime

    let params = vec![
        (1 << 10 311usize),
        (1 << 11, 308usize),
        (1 << 12, 304usize),
        (1 << 13, 301usize),
        (1 << 14, 298usize),
        (1 << 15, 294usize),
        (1 << 16, 291),
        (1 << 17, 287),
        (1 << 18, 284),
        (1 << 19, 280),
        (1 << 20, 277),
        (1 << 21, 273),
        (1 << 22, 270),
        (1 << 23, 266),
        (1 << 24, 263),
        (1 << 25, 259),
    ];

    for &(n, d) in &params {
        let N = 4 * n;
        let log_n = usize::BITS - (n as usize).leading_zeros() - 1;
        let bench_id = BenchmarkId::new(format!("bench for n=2 ** {}", log_n), "");

        let rng = &mut thread_rng();
        let t_operator = TOperator::<F>::rand(n);

        group.bench_with_input(bench_id, &n, |b, &_n| {
            b.iter(|| {
                let error = SparseVector::error_vec(N, d, rng);
                let _instance = DualLPNInstance::new(&t_operator, error);
            });
        });
    }


    group.finish();
}

criterion_group!(benches, bench_dual_lpn);
criterion_main!(benches);


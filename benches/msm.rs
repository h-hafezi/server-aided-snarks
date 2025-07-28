use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use ark_ff::{UniformRand};
use ark_bn254::{Fr, G1Projective};
use ark_std::test_rng;

use server_aided_SNARK::emsm::pederson::Pedersen;

fn bench_pedersen(c: &mut Criterion) {
    let mut group = c.benchmark_group("pedersen_commitment");
    group.sample_size(10);

    let ns: Vec<usize> = (10..=20).map(|i| 1 << i).collect();

    for &n in &ns {
        let log_n = usize::BITS - (n as usize).leading_zeros() - 1;

        // Benchmark Pedersen setup
        let setup_id = BenchmarkId::new(format!("setup n=2^{}", log_n), "");
        group.bench_with_input(setup_id, &n, |b, &n| {
            b.iter(|| {
                let _pedersen = Pedersen::<G1Projective>::new(n);
            });
        });

        // Prepare scalars ahead of time
        let mut rng = test_rng();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let pedersen = Pedersen::<G1Projective>::new(n);

        // Benchmark Pedersen commit
        let commit_id = BenchmarkId::new(format!("commit n=2^{}", log_n), "");
        group.bench_with_input(commit_id, &n, |b, &_n| {
            b.iter(|| {
                let _commitment = pedersen.commit(&scalars);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_pedersen);
criterion_main!(benches);

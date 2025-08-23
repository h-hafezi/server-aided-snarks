use ark_std::UniformRand;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use rand::thread_rng;
use server_aided_SNARK::emsm::dual_lpn::DualLPNInstance;
use server_aided_SNARK::emsm::emsm::EmsmPublicParams;
use server_aided_SNARK::emsm::pederson::Pedersen;
use server_aided_SNARK::emsm::sparse_vec::SparseVector;

type F = ark_bn254::Fr;
type G1Projective = ark_bn254::G1Projective;

fn bench_emsm(c: &mut Criterion) {
    let mut rng = thread_rng();

    let params = vec![
        (1 << 10, 2 * 311usize),
        (1 << 11, 2 * 308),
        (1 << 12, 2 * 304),
        (1 << 13, 2 * 302),
        (1 << 14, 2 * 301),
        (1 << 15, 2 * 298),
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

    for (n, k) in &params {
        // ------------------- Precompute common data -------------------
        let pederson = Pedersen::<ark_bn254::G1Projective>::new(*n);
        let pp = EmsmPublicParams::<F, G1Projective>::new(*n, pederson.generators);

        // Benchmark SparseVector::error_vec
        c.bench_with_input(BenchmarkId::new("error_vec", n), n, |b, &_n| {
            b.iter(|| {
                black_box(SparseVector::<F>::error_vec(n * 4, *k, &mut rng));
            });
        });

        let noise = SparseVector::<F>::error_vec(n * 4, *k, &mut rng);

        // Benchmark DualLPNInstance::new
        c.bench_with_input(BenchmarkId::new("DualLPNInstance::new", n), n, |b, &_n| {
            b.iter(|| {
                black_box(DualLPNInstance::<F>::new(&pp.t_operator, noise.clone()));
            });
        });

        let emsm_instance = DualLPNInstance::<F>::new(&pp.t_operator, noise.clone());

        let witness = (0..*n).map(|_| F::rand(&mut rng)).collect::<Vec<F>>();
        let encrypted_witness = emsm_instance.mask_witness(&pp, witness.as_slice());
        let encrypted_msm = pp.server_computation(encrypted_witness.clone());
        let preprocessed_commitments = pp.preprocess();

        // ------------------- Benchmarks -------------------

        c.bench_with_input(BenchmarkId::new("mask_witness", n), n, |b, &_n| {
            b.iter(|| black_box(emsm_instance.mask_witness(&pp, witness.as_slice())));
        });

        c.bench_with_input(BenchmarkId::new("server_computation", n), n, |b, &_n| {
            b.iter(|| black_box(pp.server_computation(encrypted_witness.clone())));
        });

        c.bench_with_input(BenchmarkId::new("recompute_msm", n), n, |b, &_n| {
            b.iter(|| black_box(emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone())));
        });
    }
}

// Set global sample size to 10
criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_emsm
}
criterion_main!(benches);



use ark_std::UniformRand;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use rand::thread_rng;
use server_aided_SNARK::emsm::emsm::EmsmPublicParams;
use server_aided_SNARK::emsm::pederson::Pedersen;

type F = ark_bn254::Fr;
type G1Projective = ark_bn254::G1Projective;

fn bench_preprocessing(c: &mut Criterion) {
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

    for (n, _k) in &params {
        // ------------------- Precompute common data -------------------
        let pederson = Pedersen::<G1Projective>::new(*n);
        let pp = EmsmPublicParams::<F, G1Projective>::new(*n, pederson.generators);

        // Only benchmark preprocessing
        c.bench_with_input(BenchmarkId::new("preprocessing", n), n, |b, &_n| {
            b.iter(|| black_box(pp.preprocess()));
        });
    }
}

// Set global sample size to 10
criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_preprocessing
}
criterion_main!(benches);

use ark_std::UniformRand;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use rand::thread_rng;
use server_aided_SNARK::emsm::dual_lpn::DualLPNInstance;
use server_aided_SNARK::emsm::emsm::EmsmPublicParams;
use server_aided_SNARK::emsm::pederson::Pedersen;
use server_aided_SNARK::emsm::sparse_vec::SparseVector;
use rayon::prelude::*;
use crossbeam::thread;

type F = ark_bn254::Fr;
type G1Projective = ark_bn254::G1Projective;

fn bench_malicious_emsm(c: &mut Criterion) {
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
        c.bench_with_input(BenchmarkId::new("error_vec_sequential", n), n, |b, &_n| {
            b.iter(|| {
                black_box(SparseVector::<F>::error_vec(n * 4, *k, &mut rng));
                black_box(SparseVector::<F>::error_vec(n * 4, *k, &mut rng));
            });
        });


        let noise1 = SparseVector::<F>::error_vec(n * 4, *k, &mut rng);
        let noise2 = SparseVector::<F>::error_vec(n * 4, *k, &mut rng);


        // Benchmark DualLPNInstance::new
        c.bench_with_input(BenchmarkId::new("DualLPNInstance_sequential", n), n, |b, &_n| {
            b.iter(|| {
                black_box(DualLPNInstance::<F>::new(&pp.t_operator, noise1.clone()));
                black_box(DualLPNInstance::<F>::new(&pp.t_operator, noise2.clone()));
            });
        });

        c.bench_with_input(BenchmarkId::new("DualLPNInstance_parallel", n), n, |b, &_n| {
            b.iter(|| {
                thread::scope(|scope| {
                    let handle1 = scope.spawn(|_| {
                        DualLPNInstance::<F>::new(&pp.t_operator, noise1.clone())
                    });

                    let handle2 = scope.spawn(|_| {
                        DualLPNInstance::<F>::new(&pp.t_operator, noise2.clone())
                    });

                    let instance1 = handle1.join().unwrap();
                    let instance2 = handle2.join().unwrap();

                    black_box((instance1, instance2));
                }).unwrap();
            });
        });


        let emsm_instance1 = DualLPNInstance::<F>::new(&pp.t_operator, noise1.clone());
        let emsm_instance2 = DualLPNInstance::<F>::new(&pp.t_operator, noise2.clone());

        let s = F::rand(&mut rng);
        let witness = (0..*n).map(|_| F::rand(&mut rng)).collect::<Vec<F>>();
        let scaled_witness: Vec<F> = witness.iter().map(|w| w * &s).collect();
        let encrypted_witness = emsm_instance1.mask_witness(&pp, witness.as_slice());
        let encrypted_scaled_witness = emsm_instance2.mask_witness(&pp, scaled_witness.as_slice());
        let encrypted_msm = pp.server_computation(encrypted_witness.clone());
        let encrypted_msm_scaled = pp.server_computation(encrypted_scaled_witness.clone());

        let preprocessed_commitments = pp.preprocess();

        // ------------------- Benchmarks -------------------

        c.bench_with_input(BenchmarkId::new("scaled_and_masked_witness_sequential", n), n, |b, &_n| {
            b.iter(|| {
                let scaled_witness: Vec<F> = witness.iter().map(|w| w * &s).collect();
                let encrypted_witness = emsm_instance1.mask_witness(&pp, witness.as_slice());
                let encrypted_scaled_witness = emsm_instance2.mask_witness(&pp, scaled_witness.as_slice());
                black_box((encrypted_witness, encrypted_scaled_witness))
            });
        });

        c.bench_with_input(
            BenchmarkId::new("scaled_and_masked_witness_parallel", n),
            n,
            |b, &_n| {
                b.iter(|| {
                    let scaled_witness: Vec<F> = witness.par_iter().map(|w| w * &s).collect();

                    thread::scope(|scope| {
                        let handle1 = scope.spawn(|_| emsm_instance1.mask_witness(&pp, &witness));
                        let handle2 = scope.spawn(|_| emsm_instance2.mask_witness(&pp, &scaled_witness));

                        let encrypted_witness = handle1.join().unwrap();
                        let encrypted_scaled_witness = handle2.join().unwrap();

                        black_box((encrypted_witness, encrypted_scaled_witness));
                    }).expect("error");
                });
            },
        );
        
        c.bench_with_input(BenchmarkId::new("naive_client_msm", n), n, |b, &_n| {
            b.iter(|| {
                let _ = black_box(pp.server_computation(encrypted_witness.clone()));
            });
        });

        c.bench_with_input(BenchmarkId::new("server_computation_parallel", n), n, |b, &_n| {
            b.iter(|| {
                thread::scope(|scope| {
                    let handle1 = scope.spawn(|_| {
                        pp.server_computation(encrypted_witness.clone())
                    });

                    let handle2 = scope.spawn(|_| {
                        pp.server_computation(encrypted_scaled_witness.clone())
                    });

                    // Join threads and black_box results
                    let res1 = handle1.join().unwrap();
                    let res2 = handle2.join().unwrap();

                    black_box((res1, res2));
                }).unwrap();
            });
        });


        c.bench_with_input(BenchmarkId::new("recompute_msm", n), n, |b, &_n| {
            b.iter(|| {
                let _ = black_box(emsm_instance1.recompute_msm(&preprocessed_commitments, encrypted_msm.clone()));
                let _ = black_box(emsm_instance2.recompute_msm(&preprocessed_commitments, encrypted_msm_scaled.clone()));
            });
        });
         
    }
}

// Set global sample size to 10
criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_malicious_emsm
}
criterion_main!(benches);



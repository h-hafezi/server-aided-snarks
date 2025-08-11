/*use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use ark_ff::UniformRand;
use ark_bls12_381::{Fr as F, G1Projective};
use ark_std::test_rng;
use server_aided_SNARK::emsm::outsource_msm::dual::{DualEmsmInstance, DualEmsmPublicParams};
use server_aided_SNARK::emsm::matrix::dense::DenseMatrix;

fn bench_msm_protocol(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm_protocol");
    group.sample_size(10);

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

    for &(n, non_zeros) in &params {
        let log_n = usize::BITS - (n as usize).leading_zeros() - 1;

        let mut rng = test_rng();
        let u_matrix = DenseMatrix::<F>::rand(n, n, &mut rng);
        let witness: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        // benchmark ServerState::new
        let server_new_id = BenchmarkId::new(format!("server_state_new n=2^{}", log_n), "");
        group.bench_with_input(server_new_id, &n, |b, &_n| {
            b.iter(|| {
                DualEmsmPublicParams::<F, G1Projective>::new(n, n, u_matrix.clone())
            });
        });

        let pp = DualEmsmPublicParams::<F, G1Projective>::new(n, n, u_matrix.clone());

        // benchmark ClientState::new
        let client_new_id = BenchmarkId::new(format!("client_state_new n=2^{}", log_n), "");
        group.bench_with_input(client_new_id, &n, |b, &_n| {
            b.iter(|| {
                DualEmsmInstance::<F>::new(&pp, non_zeros)
            });
        });

        let emsm_instance = DualEmsmInstance::<F>::new(&pp, non_zeros);

        // benchmark client_phase1
        let client_phase1_id = BenchmarkId::new(format!("client_phase1 n=2^{}", log_n), "");
        group.bench_with_input(client_phase1_id, &n, |b, &_n| {
            b.iter(|| {
                emsm_instance.mask_witness(&pp, witness.as_slice())
            });
        });

        let encrypted_witness = emsm_instance.mask_witness(&pp, witness.as_slice());

        let preprocessed_commitments = pp.preprocess();

        // benchmark server_compute
        let server_compute_id = BenchmarkId::new(format!("server_compute n=2^{}", log_n), "");
        group.bench_with_input(server_compute_id, &n, |b, &_n| {
            b.iter(|| {
                pp.server_computation(encrypted_witness.clone())
            });
        });

        let encrypted_msm = pp.server_computation(encrypted_witness.clone());

        // benchmark client_phase2
        let client_phase2_id = BenchmarkId::new(format!("client_phase2 n=2^{}", log_n), "");
        group.bench_with_input(client_phase2_id, &n, |b, &_n| {
            b.iter(|| {
                emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone())
            });
        });

        // benchmark compute_msm_in_plaintext
        let compute_plaintext_id = BenchmarkId::new(format!("compute_msm_plaintext n=2^{}", log_n), "");
        group.bench_with_input(compute_plaintext_id, &n, |b, &_n| {
            b.iter(|| {
                emsm_instance.compute_msm_in_plaintext(&pp, witness.as_slice())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_msm_protocol);
criterion_main!(benches);

 */
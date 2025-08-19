use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use ark_ff::UniformRand;
use ark_vesta::{Fr, Projective as G1Projective};
use ark_std::{test_rng, fs::File,};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use ark_ec::VariableBaseMSM;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use server_aided_SNARK::emsm::pederson::Pedersen;

fn setup_dir() -> &'static str {
    "setup"
}

fn pedersen_setup_path(n: usize) -> String {
    format!("{}/pedersen_n_{}.bin", setup_dir(), n)
}

/// Load Pedersen setup from file or generate and save it.
fn load_or_generate_pedersen(n: usize) -> Pedersen<G1Projective> {
    let path = pedersen_setup_path(n);
    if Path::new(&path).exists() {
        // Load from file using ark_serialize
        let file = File::open(&path).expect("Failed to open Pedersen setup file");
        let mut reader = BufReader::new(file);
        Pedersen::<G1Projective>::deserialize_uncompressed(&mut reader)
            .expect("Failed to deserialize Pedersen setup")
    } else {
        fs::create_dir_all(setup_dir()).expect("Failed to create setup directory");

        // Generate new setup
        let pedersen = Pedersen::<G1Projective>::new(n);

        // Save to file using ark_serialize
        let file = File::create(&path).expect("Failed to create Pedersen setup file");
        let mut writer = BufWriter::new(file);
        pedersen
            .serialize_uncompressed(&mut writer)
            .expect("Failed to serialize Pedersen setup");

        pedersen
    }
}

fn bench_pedersen(c: &mut Criterion) {
    let mut group = c.benchmark_group("pedersen_commitment");
    group.sample_size(10);

    let ns: Vec<usize> = (10..=25).map(|i| 1 << i).collect();

    for &n in &ns {
        let log_n = usize::BITS - (n as usize).leading_zeros() - 1;

        // Load or generate Pedersen setup
        let binding = load_or_generate_pedersen(n).generators;
        let generators = binding.as_slice();

        // Prepare random scalars
        let mut rng = test_rng();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // Benchmark only commitment (setup already cached)
        let commit_id = BenchmarkId::new(format!("commit n=2^{}", log_n), "");
        group.bench_with_input(commit_id, &n, |b, &_n| {
            b.iter(|| {
                let _ = G1Projective::msm_unchecked(generators, scalars.as_slice());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_pedersen);
criterion_main!(benches);

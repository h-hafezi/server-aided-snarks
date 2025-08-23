use ark_poly::{DenseUVPolynomial, Polynomial};
use ark_poly::univariate::DensePolynomial;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;

use server_aided_SNARK::pcs::kzg::*;
use server_aided_SNARK::nova::constant_for_curves::{E, ScalarField};

type Poly = DensePolynomial<ScalarField>;

fn bench_kzg(c: &mut Criterion) {
    let rng = &mut test_rng();

    // Degrees: 2^10 to 2^25
    let degrees: Vec<usize> = (10..=25).map(|i| 1 << i).collect();

    for &degree in degrees.iter() {
        // Setup parameters for this degree
        let params = KZG10::<E, Poly>::setup(degree, false, rng).unwrap();
        let (ck, _vk) = trim(&params, degree);

        // Random polynomial of degree `degree`
        let poly = Poly::rand(degree, rng);
        let randomness = KZGRandomness::<ScalarField, Poly>::empty();

        // Random evaluation point
        let point = ScalarField::rand(rng);

        // --- Benchmark compute_witness_polynomial ---
        c.bench_with_input(
            BenchmarkId::new("compute_witness", degree),
            &degree,
            |b, &_deg| {
                b.iter(|| {
                    let _ = KZG10::<E, Poly>::compute_witness_polynomial(&poly, point, &randomness)
                        .unwrap();
                })
            },
        );

        // --- Benchmark open_with_witness_polynomial ---
        let (witness_poly, _hiding_poly) = KZG10::<E, Poly>::compute_witness_polynomial(&poly, point, &randomness).unwrap();

        c.bench_with_input(
            BenchmarkId::new("open_with_witness", degree),
            &degree,
            |b, &_deg| {
                b.iter(|| {
                    let _ = KZG10::<E, Poly>::open_with_witness_polynomial(
                        &ck,
                        point,
                        &randomness,
                        &witness_poly,
                        None, // no hiding
                    )
                        .unwrap();
                })
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_kzg
}

criterion_main!(benches);

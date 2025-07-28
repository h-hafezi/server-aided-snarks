#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, Criterion};
use ark_std::test_rng;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::*;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef};
use ark_ff::PrimeField;

use server_aided_SNARK::nova::constant_for_curves::{ScalarField, C1, G1};
use server_aided_SNARK::nova::gadgets::r1cs::{
    R1CSShape, R1CSInstance, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use server_aided_SNARK::nova::commitment::CommitmentScheme;
use server_aided_SNARK::nova::poseidon::PoseidonHash;

/// Generates a random constraint system
fn generate_random_constraint_system<F: PrimeField>(
    num_constraints: usize,
    num_vars: usize,
    num_io: usize,
) -> ConstraintSystemRef<F> {
    let cs = ConstraintSystem::<F>::new_ref();
    let rng = &mut test_rng();

    for _ in 0..num_io - 1 {
        let _ = FpVar::<F>::new_input(cs.clone(), || Ok(F::rand(rng))).unwrap();
    }

    let mut witness_vars = vec![];
    for _ in 0..num_vars - num_constraints {
        let var = FpVar::<F>::new_witness(cs.clone(), || Ok(F::rand(rng))).unwrap();
        witness_vars.push(var);
    }

    for i in 0..num_constraints {
        let j = i % std::cmp::max(1, witness_vars.len());
        let _ = &witness_vars[j] * &witness_vars[j];
    }

    assert_eq!(cs.num_constraints(), num_constraints);
    assert_eq!(cs.num_witness_variables(), num_vars);
    assert_eq!(cs.num_instance_variables(), num_io);

    cs
}

/// Computes the cross-term vector `T`
fn compute_T<G, C>(
    shape: &R1CSShape<G>,
    U1: &RelaxedR1CSInstance<G, C>,
    W1: &RelaxedR1CSWitness<G>,
    U2: &R1CSInstance<G, C>,
    W2: &R1CSWitness<G>,
) -> Vec<G::ScalarField>
where
    G: SWCurveConfig + Clone,
    G::ScalarField: PrimeField,
    C: CommitmentScheme<Projective<G>, PP = Vec<Affine<G>>, SetupAux = ()>,
{
    let z1 = [&U1.X, &W1.W[..]].concat();
    let Az1 = shape.A.multiply_vec(&z1);
    let Bz1 = shape.B.multiply_vec(&z1);
    let Cz1 = shape.C.multiply_vec(&z1);

    let z2 = [&U2.X, &W2.W[..]].concat();
    let Az2 = shape.A.multiply_vec(&z2);
    let Bz2 = shape.B.multiply_vec(&z2);
    let Cz2 = shape.C.multiply_vec(&z2);

    let Az1_Bz2: Vec<_> = Az1.iter().zip(&Bz2).map(|(a, b)| *a * *b).collect();
    let Az2_Bz1: Vec<_> = Az2.iter().zip(&Bz1).map(|(a, b)| *a * *b).collect();

    let u1 = U1.X[0];
    let u1_Cz2: Vec<_> = Cz2.into_iter().map(|cz2| u1 * cz2).collect();

    Az1_Bz2
        .into_iter()
        .zip(Az2_Bz1)
        .zip(u1_Cz2)
        .zip(Cz1)
        .map(|(((a1b2, a2b1), u1cz2), cz1)| a1b2 + a2b1 - u1cz2 - cz1)
        .collect()
}

fn nova(c: &mut Criterion) {
    let n_values: Vec<usize> = (10..=20).map(|i| 1 << i).collect();

    for &n in &n_values {
        let label = format!("n = 2^{}, n = {}", n.ilog2(), n);

        let num_constraints = n / 2;
        let num_vars = n;
        let num_io = 2;

        let pp = C1::setup(n, b"teset", &());
        let cs = generate_random_constraint_system::<ScalarField>(
            num_constraints,
            num_vars,
            num_io,
        );
        assert!(cs.is_satisfied().unwrap());

        let (shape, instance, witness) = {
            let shape = R1CSShape::<G1>::from(cs.clone());
            let cs_borrow = cs.borrow().unwrap();
            let W = cs_borrow.witness_assignment.clone();
            let X = cs_borrow.instance_assignment.clone();
            let witness = R1CSWitness::<G1> { W };
            let commitment_W = witness.commit::<C1>(&pp); // to be benched
            let instance = R1CSInstance::<G1, C1> { commitment_W, X };
            assert!(shape.is_satisfied(&instance, &witness, &pp).is_ok());
            (shape, instance, witness)
        };

        let (relaxed_u, relaxed_w) = shape.random_relaxed_r1cs(&pp, &mut test_rng());
        shape.is_relaxed_satisfied::<C1>(&relaxed_u, &relaxed_w, &pp).unwrap();

        // Commitment of W
        c.bench_function(&format!("commitment_W - {}", label), |b| {
            b.iter(|| {
                let _ = witness.commit::<C1>(&pp);
            });
        });

        // Compute T
        c.bench_function(&format!("compute_T - {}", label), |b| {
            b.iter(|| {
                let _ = compute_T(&shape, &relaxed_u, &relaxed_w, &instance, &witness);
            });
        });

        let t = compute_T(&shape, &relaxed_u, &relaxed_w, &instance, &witness);

        // Commitment of T
        c.bench_function(&format!("commit_T - {}", label), |b| {
            b.iter(|| {
                let _ = C1::commit(&pp, t.as_slice());
            });
        });

        // Poseidon full hash pipeline
        c.bench_function(&format!("Poseidon hash pipeline - {}", label), |b| {
            b.iter(|| {
                // Prepare input
                let mut input = instance.to_sponge_field_elements();
                input.extend_from_slice(&relaxed_u.to_sponge_field_elements());
                std::hint::black_box(&input);

                // Update sponge
                let mut sponge = PoseidonHash::<ScalarField>::new();
                sponge.update_sponge(input.clone());

                // Output sponge
                let _ = sponge.output();
            });
        });

        let mut sponge = PoseidonHash::<ScalarField>::new();
        let mut input = instance.to_sponge_field_elements();
        input.extend_from_slice(&relaxed_u.to_sponge_field_elements());
        sponge.update_sponge(input.clone());
        let r = sponge.output();

        // Fold witness
        c.bench_function(&format!("fold witness - {}", label), |b| {
            b.iter(|| {
                let _ = relaxed_w.fold(&witness, t.as_slice(), &r).unwrap();
            });
        });
    }
}

criterion_group!(benches, nova);
criterion_main!(benches);

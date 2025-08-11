use criterion::{criterion_group, criterion_main, Criterion};
use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_relations::r1cs::{ConstraintSystemRef, ConstraintSystem, OptimizationGoal, ConstraintSynthesizer};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;
use ark_std::{test_rng, rand::SeedableRng, cfg_into_iter, cfg_iter};
use server_aided_SNARK::groth16::{Groth16};
use server_aided_SNARK::groth16::r1cs_to_qap::{LibsnarkReduction, R1CSToQAP};
use ark_poly::GeneralEvaluationDomain;
use std::ops::{Add, Mul};
use std::marker::PhantomData;
use ark_crypto_primitives::snark::CircuitSpecificSetupSNARK;
use ark_ec::bn::Bn;
use server_aided_SNARK::nova::constant_for_curves::{ScalarField};
use ark_std::rand::RngCore;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;

struct RandomCircuit<F: PrimeField> {
    num_constraints: usize,
    num_vars: usize,
    num_io: usize,
    phantom: PhantomData<F>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for RandomCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), ark_relations::r1cs::SynthesisError> {
        let rng = &mut test_rng();
        for _ in 0..self.num_io - 1 {
            let _ = FpVar::<F>::new_input(cs.clone(), || Ok(F::rand(rng))).unwrap();
        }

        let mut witness_vars = vec![];
        for _ in 0..self.num_vars - self.num_constraints {
            let var = FpVar::<F>::new_witness(cs.clone(), || Ok(F::rand(rng))).unwrap();
            witness_vars.push(var);
        }

        for i in 0..self.num_constraints {
            let j = i % std::cmp::max(1, witness_vars.len());
            let _ = &witness_vars[j] * &witness_vars[j];
        }

        Ok(())
    }
}

fn bench_groth16(c: &mut Criterion) {
    for exp in 10..=20 {
        let n = 1 << exp;
        let num_constraints = n / 2;
        let num_vars = n;
        let num_io = 2;
        let mut rng = rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let circuit = RandomCircuit::<<Bn<ark_bn254::Config> as Pairing>::ScalarField> {
            num_constraints,
            num_vars,
            num_io,
            phantom: PhantomData,
        };

        let (pk, _) = Groth16::<Bn254>::setup(circuit, &mut rng).unwrap();

        let circuit = RandomCircuit::<<Bn<ark_bn254::Config> as Pairing>::ScalarField> {
            num_constraints,
            num_vars,
            num_io,
            phantom: PhantomData,
        };

        let r = ScalarField::rand(&mut rng);
        let s = ScalarField::rand(&mut rng);

        let cs = ConstraintSystem::new_ref();

        // Set the optimization goal
        cs.set_optimization_goal(OptimizationGoal::Constraints);

        // Synthesize the circuit.
        circuit.generate_constraints(cs.clone()).expect("panic");
        debug_assert!(cs.is_satisfied().unwrap());
        cs.finalize();

        let prover = cs.borrow().unwrap();

        let id = format!("witness_map_exp_{}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _h = LibsnarkReduction::witness_map::<ScalarField, GeneralEvaluationDomain<ScalarField>>(cs.clone()).unwrap();
            })
        });

        let h = LibsnarkReduction::witness_map::<<Bn<ark_bn254::Config> as Pairing>::ScalarField, GeneralEvaluationDomain<<Bn<ark_bn254::Config> as Pairing>::ScalarField>>(cs.clone()).unwrap();

        // first MSM
        let h_assignment = cfg_into_iter!(&h)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        let h_acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.h_query, &h_assignment);

        let id = format!("first msm {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let h_assignment = cfg_into_iter!(&h)
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let _ = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.h_query, &h_assignment);
            })
        });


        // second MSM
        let aux_assignment_bigint = cfg_iter!(&prover.witness_assignment)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        let l_aux_acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.l_query, &aux_assignment_bigint);

        let id = format!("second msm {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let aux_assignment_bigint = cfg_iter!(&prover.witness_assignment)
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let _ = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.l_query, &aux_assignment_bigint);
            })
        });


        // other operations
        let r_s_delta_g1 = pk.delta_g1 * (r * s);
        let input_assignment_bigint = prover.instance_assignment[1..]
            .iter()
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        let assignment = [&input_assignment_bigint[..], &aux_assignment_bigint[..]].concat();

        // Compute A
        let r_g1 = pk.delta_g1.mul(r);

        let id = format!("other operation (1) {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                // other operations
                let _ = pk.delta_g1 * (r * s);
                let input_assignment_bigint = prover.instance_assignment[1..]
                    .iter()
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let _ = [&input_assignment_bigint[..], &aux_assignment_bigint[..]].concat();

                // Compute A
                let _ = pk.delta_g1.mul(r);
            })
        });

        // third MSM
        let g_a = {
            let el = pk.a_query[0];
            let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.a_query[1..], &assignment);
            r_g1 + el + acc + pk.vk.alpha_g1
        };

        let id = format!("third msm {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = {
                    let el = pk.a_query[0];
                    let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.a_query[1..], &assignment);
                    r_g1 + el + acc + pk.vk.alpha_g1
                };
            })
        });


        let s_g_a = g_a * &s;

        let g1_b = {
            let s_g1 = pk.delta_g1.mul(s);
            let el = pk.b_g1_query[0];

            // forth MSM
            let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.b_g1_query[1..], &assignment);
            s_g1 + el + acc + pk.beta_g1
        };

        let id = format!("forth msm {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let s_g1 = pk.delta_g1.mul(s);
                let el = pk.b_g1_query[0];

                // forth MSM
                let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.b_g1_query[1..], &assignment);
                s_g1 + el + acc + pk.beta_g1
            })
        });


        // Compute B in G2
        let s_g2 = pk.vk.delta_g2.mul(s);
        let _ = {
            let el = pk.b_g2_query[0];

            // fifth MSM
            let acc = <Bn<ark_bn254::Config> as Pairing>::G2::msm_bigint(&pk.b_g2_query[1..], &assignment);
            s_g2.add(&el).add(&acc).add(&pk.vk.beta_g2)
        };

        let id = format!("fifth msm {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = pk.b_g2_query[0];

                // fifth MSM
                let _ = <Bn<ark_bn254::Config> as Pairing>::G2::msm_bigint(&pk.b_g2_query[1..], &assignment);
            })
        });

        let r_g1_b = g1_b * &r;

        let mut g_c = s_g_a;
        g_c += &r_g1_b;
        g_c -= &r_s_delta_g1;
        g_c += &l_aux_acc;
        g_c += &h_acc;

        let id = format!("other operations (2) {}", exp);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let r_g1_b = g1_b * &r;

                let mut g_c = s_g_a;
                g_c += &r_g1_b;
                g_c -= &r_s_delta_g1;
                g_c += &l_aux_acc;
                g_c += &h_acc;
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_groth16
}

criterion_main!(benches); // <- THIS is required!

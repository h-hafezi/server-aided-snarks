use criterion::{criterion_group, criterion_main, Criterion};
use ark_bn254::{Bn254, G2Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, PrimeField, UniformRand};
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
use server_aided_SNARK::nova::constant_for_curves::{G1Affine, G1Projective, ScalarField};
use ark_std::rand::RngCore;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use server_aided_SNARK::emsm::dual_lpn::DualLPNInstance;
use server_aided_SNARK::emsm::emsm::EmsmPublicParams;
use server_aided_SNARK::emsm::sparse_vec::SparseVector;

pub fn cast_field<Fr, Fq>(first_field: Fr) -> Fq
where
    Fr: PrimeField,
    Fq: PrimeField,
{
    let bytes = first_field.into_bigint().to_bytes_le();
    Fq::from_le_bytes_mod_order(bytes.as_slice())
}

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
    let params = vec![
        (1 << 10, 2 * 311usize),
        (1 << 11, 2 * 308),
        (1 << 12, 2 * 304),
        (1 << 14, 2 * 301),
        (1 << 15, 2 * 298),
        (1 << 15, 2 * 294),
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
        let n = n.clone();
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

        let id = format!("witness_map_exp_{}", n);
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

        let id = format!("first msm {}", n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let h_assignment = cfg_into_iter!(&h)
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let _ = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.h_query, &h_assignment);
            })
        });

        let emsm_pp = {
            let mut v = pk.h_query.clone();
            v.push(G1Affine::zero());
            EmsmPublicParams::<ScalarField, G1Projective>::new(n, v)
        };
        let preprocessed_commitments = emsm_pp.preprocess();

        let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
        let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
        let encrypted_witness = emsm_instance.mask_witness(&emsm_pp, h.as_slice());
        let encrypted_msm = emsm_pp.server_computation(encrypted_witness.clone());

        c.bench_function(&format!("first emsm - {}", n), |b| {
            b.iter(|| {
                let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
                let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
                let _ = emsm_instance.mask_witness(&emsm_pp, h.as_slice());
                let _ = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone());
            });
        });

        // second MSM
        let aux_assignment_bigint = cfg_iter!(&prover.witness_assignment)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        let l_aux_acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.l_query, &aux_assignment_bigint);

        let id = format!("second msm {}", n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let aux_assignment_bigint = cfg_iter!(&prover.witness_assignment)
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let _ = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.l_query, &aux_assignment_bigint);
            })
        });

        // emsm parameters
        let emsm_pp = EmsmPublicParams::<ScalarField, G1Projective>::new(n, pk.l_query.clone());
        let preprocessed_commitments = emsm_pp.preprocess();
        let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
        let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
        let encrypted_witness = emsm_instance.mask_witness(&emsm_pp, prover.witness_assignment.as_slice());
        let encrypted_msm = emsm_pp.server_computation(encrypted_witness.clone());

        c.bench_function(&format!("second emsm - {}", n), |b| {
            b.iter(|| {
                let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
                let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
                let _ = emsm_instance.mask_witness(&emsm_pp, prover.witness_assignment.as_slice());
                let _ = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone());
            });
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

        let id = format!("other operation (1) {}", n);
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

        let id = format!("third msm {}", n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = {
                    let el = pk.a_query[0];
                    let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.a_query[1..], &assignment);
                    r_g1 + el + acc + pk.vk.alpha_g1
                };
            })
        });

        // define new msm (not big int)
        let assignment_f = [&prover.instance_assignment[1..], &prover.witness_assignment[1..]].concat();

        // emsm parameters
        let emsm_pp = EmsmPublicParams::<ScalarField, G1Projective>::new(n, pk.a_query[2..].to_vec());
        let preprocessed_commitments = emsm_pp.preprocess();
        let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
        let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
        let encrypted_witness = emsm_instance.mask_witness(&emsm_pp, assignment_f.as_slice());
        let encrypted_msm = emsm_pp.server_computation(encrypted_witness.clone());

        c.bench_function(&format!("third emsm - {}", n), |b| {
            b.iter(|| {
                let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
                let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
                let _ = emsm_instance.mask_witness(&emsm_pp, assignment_f.as_slice());
                let _ = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone());
            });
        });



        let s_g_a = g_a * &s;

        let g1_b = {
            let s_g1 = pk.delta_g1.mul(s);
            let el = pk.b_g1_query[0];

            // forth MSM
            let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.b_g1_query[1..], &assignment);
            s_g1 + el + acc + pk.beta_g1
        };

        let id = format!("forth msm {}", n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let s_g1 = pk.delta_g1.mul(s);
                let el = pk.b_g1_query[0];

                // forth MSM
                let acc = <Bn<ark_bn254::Config> as Pairing>::G1::msm_bigint(&pk.b_g1_query[1..], &assignment);
                s_g1 + el + acc + pk.beta_g1
            })
        });

        // emsm parameters
        let emsm_pp = EmsmPublicParams::<ScalarField, G1Projective>::new(n, pk.b_g1_query[2..].to_vec());
        let preprocessed_commitments = emsm_pp.preprocess();
        let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
        let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
        let encrypted_witness = emsm_instance.mask_witness(&emsm_pp, assignment_f.as_slice());
        let encrypted_msm = emsm_pp.server_computation(encrypted_witness.clone());

        c.bench_function(&format!("forth emsm - {}", n), |b| {
            b.iter(|| {
                // this instance is already encrypted, don't have to encrypt it again
                let _ = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone());
            });
        });

        // Compute B in G2
        let s_g2 = pk.vk.delta_g2.mul(s);
        let _ = {
            let el = pk.b_g2_query[0];

            // fifth MSM
            let acc = <Bn<ark_bn254::Config> as Pairing>::G2::msm_bigint(&pk.b_g2_query[1..], &assignment);
            s_g2.add(&el).add(&acc).add(&pk.vk.beta_g2)
        };

        let id = format!("fifth msm {}", n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = pk.b_g2_query[0];

                // fifth MSM
                let _ = <Bn<ark_bn254::Config> as Pairing>::G2::msm_bigint(&pk.b_g2_query[1..], &assignment);
            })
        });

        let assignment_f = [&prover.instance_assignment[1..], &prover.witness_assignment[1..]].concat();


        // emsm parameters
        let emsm_pp = EmsmPublicParams::<ScalarField, G2Projective>::new(n, pk.b_g2_query[2..].to_vec());
        let preprocessed_commitments = emsm_pp.preprocess();
        let noise = SparseVector::<ScalarField>::error_vec(n * 4, *k, &mut thread_rng());
        let emsm_instance = DualLPNInstance::<ScalarField>::new(&emsm_pp.t_operator, noise.clone());
        let encrypted_witness = emsm_instance.mask_witness(&emsm_pp, assignment_f.as_slice());
        let encrypted_msm = emsm_pp.server_computation(encrypted_witness.clone());

        c.bench_function(&format!("fifth emsm - {}", n), |b| {
            b.iter(|| {
                // this instance is already encrypted, don't have to encrypt it again
                let _ = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm.clone());
            });
        });


        let r_g1_b = g1_b * &r;

        let mut g_c = s_g_a;
        g_c += &r_g1_b;
        g_c -= &r_s_delta_g1;
        g_c += &l_aux_acc;
        g_c += &h_acc;

        let id = format!("other operations (2) {}", n);
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

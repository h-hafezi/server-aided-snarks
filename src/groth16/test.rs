use std::marker::PhantomData;
use std::ops::Mul;
use crate::groth16::{Groth16, Proof};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK};
use ark_ec::pairing::Pairing;
use ark_ec::VariableBaseMSM;
use ark_ff::{PrimeField};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::{
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError},
};
use ark_relations::r1cs::{ConstraintSystem, OptimizationGoal};
use ark_std::{cfg_into_iter, cfg_iter, rand::{RngCore, SeedableRng}, test_rng, UniformRand};
use crate::groth16::r1cs_to_qap::{LibsnarkReduction, R1CSToQAP};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use std::ops::Add;
use ark_ec::CurveGroup;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;

struct MySillyCircuit<F: PrimeField> {
    num_constraints: usize,
    num_vars: usize,
    num_io: usize,
    phantom: PhantomData<F>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for MySillyCircuit<F> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<F>,
    ) -> Result<(), SynthesisError> {
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

fn test_prove_and_verify<E>()
where
    E: Pairing,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    let n = 50000;
    let num_constraints = n / 2;
    let num_vars = n;
    let num_io = 2;


    let (pk, _vk) = Groth16::<E>::setup(
        MySillyCircuit {
            num_constraints,
            num_vars,
            num_io,
            phantom: Default::default(),
        },
        &mut rng
    ).unwrap();


    let circuit = MySillyCircuit {
        num_constraints,
        num_vars,
        num_io,
        phantom: Default::default(),
    };

    let _proof = {
        // generate randomness to make the circuit zk
        let r = E::ScalarField::rand(&mut rng);
        let s = E::ScalarField::rand(&mut rng);

        let cs = ConstraintSystem::new_ref();

        // Set the optimization goal
        cs.set_optimization_goal(OptimizationGoal::Constraints);

        // Synthesize the circuit.
        circuit.generate_constraints(cs.clone()).expect("panic");
        debug_assert!(cs.is_satisfied().unwrap());
        cs.finalize();

        let h = LibsnarkReduction::witness_map::<E::ScalarField, GeneralEvaluationDomain<E::ScalarField>>(cs.clone()).unwrap();
        let prover = cs.borrow().unwrap();

            // first MSM
            let h_assignment = cfg_into_iter!(&h)
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();
            let h_acc = E::G1::msm_bigint(&pk.h_query, &h_assignment);

            // second MSM
            let aux_assignment_bigint = cfg_iter!(&prover.witness_assignment)
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();
            let l_aux_acc = E::G1::msm_bigint(&pk.l_query, &aux_assignment_bigint);

            // other operations
            let r_s_delta_g1 = pk.delta_g1 * (r * s);
            let input_assignment_bigint = prover.instance_assignment[1..]
                .iter()
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();
            let assignment = [&input_assignment_bigint[..], &aux_assignment_bigint[..]].concat();

            // Compute A
            let r_g1 = pk.delta_g1.mul(r);

            // third MSM
            let g_a = {
                let el = pk.a_query[0];
                let acc = E::G1::msm_bigint(&pk.a_query[1..], &assignment);
                r_g1 + el + acc + pk.vk.alpha_g1
            };

            let s_g_a = g_a * &s;


            let g1_b = {
                let s_g1 = pk.delta_g1.mul(s);
                let el = pk.b_g1_query[0];

                // forth MSM
                let acc = E::G1::msm_bigint(&pk.b_g1_query[1..], &assignment);
                s_g1 + el + acc + pk.beta_g1
            };

            // Compute B in G2
            let s_g2 = pk.vk.delta_g2.mul(s);
            let g2_b = {
                let el = pk.b_g2_query[0];

                // fifth MSM
                let acc = E::G2::msm_bigint(&pk.b_g2_query[1..], &assignment);
                s_g2.add(&el).add(&acc).add(&pk.vk.beta_g2)
            };
            let r_g1_b = g1_b * &r;

            let mut g_c = s_g_a;
            g_c += &r_g1_b;
            g_c -= &r_s_delta_g1;
            g_c += &l_aux_acc;
            g_c += &h_acc;

            let proof = Proof {
                a: g_a.into_affine(),
                b: g2_b.into_affine(),
                c: g_c.into_affine(),
            };

            Ok::<Proof<E>, E>(proof)
        }.unwrap();
        // assert!(Groth16::<E>::verify_with_processed_vk(&pvk, &[c], &proof).unwrap());
        // assert!(!Groth16::<E>::verify_with_processed_vk(&pvk, &[a], &proof).unwrap());
}


mod bn_254 {
    use super::test_prove_and_verify;
    use ark_bn254::Bn254;

    #[test]
    fn prove_and_verify() {
        test_prove_and_verify::<Bn254>();
    }
}
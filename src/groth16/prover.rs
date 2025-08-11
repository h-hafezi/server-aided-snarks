use crate::groth16::{r1cs_to_qap::R1CSToQAP, Groth16, Proof, ProvingKey};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal,
    Result as R1CSResult,
};
use ark_std::rand::Rng;
use ark_std::{cfg_into_iter, cfg_iter, ops::{Mul}, vec::Vec};
use std::ops::Add;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<E: Pairing, QAP: R1CSToQAP> Groth16<E, QAP> {
    /// Create a Groth16 proof that is zero-knowledge using the provided
    /// R1CS-to-QAP reduction.
    /// This method samples randomness for zero knowledge via `rng`.
    pub fn create_random_proof_with_reduction<C>(
        pk: &ProvingKey<E>,
        circuit: C,
        rng: &mut impl Rng,
    ) -> R1CSResult<Proof<E>>
    where
        E: Pairing,
        C: ConstraintSynthesizer<E::ScalarField>,
        QAP: R1CSToQAP,
    {
        // generate randomness to make the circuti zk
        let r = E::ScalarField::rand(rng);
        let s = E::ScalarField::rand(rng);

        let cs = ConstraintSystem::new_ref();

        // Set the optimization goal
        cs.set_optimization_goal(OptimizationGoal::Constraints);

        // Synthesize the circuit.
        circuit.generate_constraints(cs.clone())?;
        debug_assert!(cs.is_satisfied().unwrap());
        cs.finalize();
        let h = QAP::witness_map::<E::ScalarField, GeneralEvaluationDomain<E::ScalarField>>(cs.clone())?;
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

        Ok(proof)
    }
}

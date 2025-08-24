use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use rayon::*;
use rayon::iter::IntoParallelRefIterator;
use crate::emsm::dual_lpn::{DualLPNInstance};
use crate::emsm::pederson::Pedersen;
use crate::emsm::raa_code::{accumulate_inplace, inverse_permutation, permute_safe, TOperator};
use rayon::iter::ParallelIterator;

#[derive(Debug, Clone)]
pub struct EmsmPublicParams<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    // Matrix T is n * N
    pub t_operator: TOperator<F>,

    // Group generators for MSM of length n
    pub pedersen: Pedersen<G>,
}

pub struct PreprocessedCommitments<G: CurveGroup> {
    pub p: Pedersen<G>,      // ⟨(T)_{*, i}, g⟩ for i in [0, N)
}

impl<F> DualLPNInstance<F> where
    F: PrimeField,
{
    pub fn mask_witness<G: CurveGroup<ScalarField = F>>(&self, pp: &EmsmPublicParams<F, G>, witness: &[F]) -> Vec<F> {
        assert_eq!(witness.len(), pp.t_operator.n);

        assert_eq!(self.lpn_vector.len(), witness.len(), "Vectors must have equal length");

        self.lpn_vector.iter()
            .zip(witness.iter())
            .map(|(a, b)| *a + *b)
            .collect()
    }

    pub fn recompute_msm<G: CurveGroup<ScalarField = F>>(
        &self,
        preprocessed_commitments: &PreprocessedCommitments<G>,
        commitment: G::Affine,
    ) -> G::Affine {
        let com_1 = preprocessed_commitments.p.commit_sparse(&self.noise);
        (commitment - com_1).into_affine()
    }

    // this function computes the msm in plaintext, can be used to benchmark and also for tests
    pub fn compute_msm_in_plaintext<G: CurveGroup<ScalarField = F>>(&self, pp: &EmsmPublicParams<F, G>, witness: &[F]) -> G::Affine {
        G::msm(&pp.pedersen.generators, witness)
            .expect("MSM computation failed")
            .into_affine()
    }
}

impl<F, G> EmsmPublicParams<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    pub fn new(n: usize, generators: Vec<G::MulBase>) -> Self {
        // Construct the Primal LPN Index (T), we put N = 4 * n
        let t_operator = TOperator::<F>::rand(n);
        
        EmsmPublicParams {
            t_operator,
            pedersen: Pedersen { generators },
        }
    }

    pub fn preprocess(&self) -> PreprocessedCommitments<G> {
        let generators = &self.pedersen.generators;

        // 1. Expand each generator 4 times
        let mut expanded: Vec<_> = generators
            .par_iter()
            .flat_map(|g| {
                let v = g.clone().into();
                vec![v.clone(), v.clone(), v.clone(), v]
            })
            .collect();

        // 2. Inverse permutation p and apply permute_safe
        let p_inv = inverse_permutation(self.t_operator.p.as_slice());
        let mut expanded = permute_safe(&mut expanded, &p_inv);

        // 3. Reverse → accumulate_inplace → reverse
        expanded.reverse();
        accumulate_inplace(&mut expanded, G::zero());
        expanded.reverse();


        // 4. Inverse permutation q and apply permute_safe
        let q_inv = inverse_permutation(self.t_operator.q.as_slice());
        expanded = permute_safe(&mut expanded, &q_inv);

        // 5. Reverse → accumulate_inplace → reverse
        expanded.reverse();
        accumulate_inplace(&mut expanded, G::zero());
        expanded.reverse();

        PreprocessedCommitments {
            p: Pedersen {
                generators: expanded.into_iter().map(|x| x.into_affine()).collect(),
            }
        }
    }

    pub fn server_computation(&self, encrypted_witness: Vec<F>) -> G::Affine {
        // Compute ⟨z_enc, g⟩
        let msm_result = G::msm(&self.pedersen.generators, &encrypted_witness).unwrap();

        msm_result.into_affine()
    }
}


#[cfg(test)]
mod tests {
    use std::ops::Neg;
    use rand::thread_rng;
    use ark_bls12_381::{Fr as F, G1Projective};
    use ark_std::UniformRand;
    use crate::emsm::dual_lpn::DualLPNInstance;
    use crate::emsm::emsm::{EmsmPublicParams};
    use crate::emsm::pederson::Pedersen;
    use crate::emsm::sparse_vec::SparseVector;

    #[test]
    fn preprocess() {
        let mut rng = thread_rng();

        let n = 1024;

        // generate client/server state
        let pederson = Pedersen::<G1Projective>::new(n);
        let pp = EmsmPublicParams::<F, G1Projective>::new(n, pederson.generators);
        let noise = SparseVector::error_vec(n * 4, 30, &mut rng);
        let emsm_instance = DualLPNInstance::<F>::new(&pp.t_operator, noise);

        // generate a random witness
        let witness = (0..n).map(|_| F::rand(&mut rng)).collect::<Vec<F>>();

        // generated the encrypted witness that is going to be passed to the server
        let encrypted_witness = emsm_instance.mask_witness(&pp, witness.as_slice());

        // ensure the length of the witness is correct
        assert_eq!(encrypted_witness.len(), n);

        // server generating preprocessed commitments
        let preprocessed_commitments = pp.preprocess();

        // server generating the encrypted msm
        let encrypted_msm = pp.server_computation(encrypted_witness);

        // client decrypting the msm
        let decrypted_msm = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm);

        // client computing the msm in plaintext
        let msm_in_plaintext = emsm_instance.compute_msm_in_plaintext(&pp, witness.as_slice());

        // checking the equality
        assert_eq!(msm_in_plaintext, decrypted_msm.neg());
    }
}

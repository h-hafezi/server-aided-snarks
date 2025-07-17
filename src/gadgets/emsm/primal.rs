use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use rand::thread_rng;
use crate::gadgets::lpn::primal_lpn::{PrimalLPNIndex, PrimalLPNInstance};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use crate::gadgets::matrix::dense::DenseMatrix;
use crate::gadgets::matrix::Matrix;
use crate::gadgets::pederson::Pedersen;
use crate::gadgets::sparse_vec::sparse_vec::SparseVector;

#[derive(Debug, Clone)]
pub struct PrimalEmsmPublicParams<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    // Public matrix U of size n * m
    pub u_matrix: DenseMatrix<F>,
    // Matrix T is m * m
    pub index: PrimalLPNIndex<F>,

    // Group generators for MSM of length n
    pub pedersen: Pedersen<G>,

    // Sizes
    pub n: usize,   // Rows of U
    pub m: usize,   // cols of U (also rows and columns of T)
    pub t: usize,   // Sparsity level for T (non-zeros per row)
}

#[derive(Debug, Clone)]
pub struct PrimalEmsmInstance<F: PrimeField> {
    pub lpn_instance: PrimalLPNInstance<F>,

    // Sizes
    pub n: usize,   // Rows of U
    pub m: usize,   // cols of U (also rows and columns of T)
    pub t: usize,   // Sparsity level for T (non-zeros per row)
}

pub struct PreprocessedCommitments<G: CurveGroup> {
    pub p: Pedersen<G>,         // ⟨(U * T)_{*, i}, g⟩ for i in [0, m)
    pub p_prime: Pedersen<G>,   // ⟨U_{*, i}, g⟩ for i in [0, n)
}

impl<F: PrimeField> PrimalEmsmInstance<F> {
    pub fn new<G: CurveGroup<ScalarField=F>>(pp: &PrimalEmsmPublicParams<F, G>, non_zero: usize) -> PrimalEmsmInstance<F> {
        let rng = &mut thread_rng();
        let noise = SparseVector::<F>::error_vec(pp.index.rows, non_zero, rng);
        let instance = PrimalLPNInstance::new(rng, &pp.index, noise);

        PrimalEmsmInstance {
            lpn_instance: instance,
            n: pp.n,
            m: pp.m,
            t: pp.t,
        }
    }

    pub fn mask_witness<G: CurveGroup<ScalarField=F>>(&self, pp: &PrimalEmsmPublicParams<F, G>, witness: &[F]) -> Vec<F> {
        assert_eq!(witness.len(), pp.u_matrix.ncols);
        assert_eq!(self.lpn_instance.lpn_vector.len(), witness.len(), "Vectors must have equal length");

        // Compute z_enc = T*s + e + z
        self.lpn_instance.lpn_vector.iter()
            .zip(witness.iter())
            .map(|(a, b)| *a + *b)
            .collect()
    }

    pub fn recompute_msm<G: CurveGroup<ScalarField=F>>(
        &self,
        preprocessed_commitments: &PreprocessedCommitments<G>,
        commitment: G::Affine,
    ) -> G::Affine {
        let com_1 = preprocessed_commitments.p.commit(self.lpn_instance.secret.as_slice()).into_affine();
        let com_2 = preprocessed_commitments.p_prime.commit_sparse(&self.lpn_instance.noise).into_affine();

        (commitment - com_1 - com_2).into()
    }

    // this function computes the msm in plaintext, can be used to benchmark and also for tests
    pub fn compute_msm_in_plaintext<G: CurveGroup<ScalarField=F>>(&self, pp: &PrimalEmsmPublicParams<F, G>, witness: &[F]) -> G::Affine {
        let mapped_witness = pp.u_matrix.right_multiply_vec(witness);

        G::msm(&pp.pedersen.generators, mapped_witness.as_slice())
            .expect("MSM computation failed")
            .into_affine()
    }
}

impl<F, G> PrimalEmsmPublicParams<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    pub fn new(
        n: usize,       // Rows of U
        m: usize,       // Rows of T (cols of U)
        t: usize,       // Sparsity level per row in T
        u_matrix: DenseMatrix<F>,       // Public U matrix
    ) -> Self {
        assert_eq!((u_matrix.nrows, u_matrix.ncols), (n, m));
        let rng = &mut thread_rng();

        // Construct the Primal LPN Index (T)
        let index = PrimalLPNIndex::<F>::new(rng, m, m, t);

        // Construct Pedersen generators
        let pedersen = Pedersen::<G>::new(n);

        PrimalEmsmPublicParams {
            index,
            u_matrix,
            pedersen,
            n,
            m,
            t,
        }
    }

    pub fn preprocess(&self) -> PreprocessedCommitments<G> {
        // ⟨(U * T)_{*, i}, g⟩ for i in [0, m)
        let p: Vec<G::Affine> = (0..self.m).into_par_iter()
            .map(|i| {
                let t_col = self.index.t_matrix.matrix().get_column(i);
                let h_col = self.u_matrix.right_multiply_vec(t_col.as_slice());
                G::msm(&self.pedersen.generators, h_col.as_slice())
                    .expect("MSM computation failed")
                    .into_affine()
            })
            .collect();

        // ⟨U_{*, i}, g⟩ for i in [0, m)
        let p_prime: Vec<G::Affine> = (0..self.m).into_par_iter()
            .map(|i| {
                let u_col = self.u_matrix.get_column(i);
                G::msm(&self.pedersen.generators, u_col.as_slice())
                    .expect("MSM computation failed")
                    .into_affine()
            })
            .collect();

        PreprocessedCommitments {
            p: Pedersen { generators: p },
            p_prime: Pedersen { generators: p_prime },
        }
    }

    pub fn server_computation(&self, encrypted_witness: Vec<F>) -> G::Affine {
        // Compute U * z_enc
        let u_z_enc = self.u_matrix.right_multiply_vec(encrypted_witness.as_slice());

        // Compute ⟨U * z_enc, g⟩
        let msm_result = G::msm(&self.pedersen.generators, &u_z_enc).unwrap();

        msm_result.into_affine()
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use ark_bls12_381::{Fr as F, G1Projective};
    use ark_std::UniformRand;
    use crate::gadgets::emsm::primal::{PrimalEmsmInstance, PrimalEmsmPublicParams};
    use crate::gadgets::matrix::dense::DenseMatrix;

    #[test]
    fn preprocess() {
        let mut rng = thread_rng();
        let (m, t, n, non_zeros) = (100, 10, 70, 40usize);
        let u_matrix = DenseMatrix::<F>::rand(n, m, &mut rng);

        // generate client/server state
        let pp = PrimalEmsmPublicParams::<F, G1Projective>::new(n, m, t, u_matrix.clone());
        let emsm_instance = PrimalEmsmInstance::<F>::new(&pp, non_zeros);

        // generate a random witness
        let witness = (0..m).map(|_| F::rand(&mut rng)).collect::<Vec<F>>();

        // generated the encrypted witness that is going to be passed to the server
        let encrypted_witness = emsm_instance.mask_witness(&pp, witness.as_slice());

        // ensure the length of the witness is correct
        assert_eq!(encrypted_witness.len(), m);

        // server generating preprocessed commitments
        let preprocessed_commitments = pp.preprocess();

        // server generating the encrypted msm
        let encrypted_msm = pp.server_computation(encrypted_witness);

        // client decrypting the msm
        let decrypted_msm = emsm_instance.recompute_msm(&preprocessed_commitments, encrypted_msm);

        // client computing the msm in plaintext
        let msm_in_plaintext = emsm_instance.compute_msm_in_plaintext(&pp, witness.as_slice());

        // checking the equality
        assert_eq!(msm_in_plaintext, decrypted_msm);
    }
}

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use crate::gadgets::lpn::dual_lpn::{DualLPNIndex, DualLPNInstance};
use crate::gadgets::matrix::dense::DenseMatrix;
use crate::gadgets::matrix::Matrix;
use crate::gadgets::pederson::Pedersen;
use crate::gadgets::sparse_vec::sparse_vec::SparseVector;

#[derive(Debug, Clone)]
pub struct DualEmsmPublicParams<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    // Public matrix U of size m * n
    pub u_matrix: DenseMatrix<F>,

    // Matrix T is n * N
    pub index: DualLPNIndex<F>,

    // Group generators for MSM of length m
    pub pedersen: Pedersen<G>,

    // Sizes
    pub n: usize,   // columns of U = rows of T
    pub N: usize,   // columns of T
    pub m: usize,   // rows of U
}

#[derive(Debug, Clone)]
pub struct DualEmsmInstance<F>
where
    F: PrimeField,
{
    // Matrix T is n * N
    pub lpn_instance: DualLPNInstance<F>,

    // Sizes
    pub n: usize,   // columns of U = rows of T
    pub N: usize,   // columns of T
    pub m: usize,   // rows of U
}

pub struct PreprocessedCommitments<G: CurveGroup> {
    pub p: Pedersen<G>,      // ⟨(U * T)_{*, i}, g⟩ for i in [0, N)
}

impl<F> DualEmsmInstance<F> where
    F: PrimeField,
{
    pub fn new<G: CurveGroup<ScalarField = F>>(pp: &DualEmsmPublicParams<F, G>, non_zero: usize) -> DualEmsmInstance<F> {
        let rng = &mut thread_rng();

        let error = SparseVector::<F>::error_vec(4 * pp.n, non_zero, rng);
        let instance = DualLPNInstance::new(&pp.index, error);

        DualEmsmInstance {
            lpn_instance: instance,
            n: pp.n,
            N: 4 * pp.n,
            m: pp.m,
        }
    }

    pub fn mask_witness<G: CurveGroup<ScalarField = F>>(&self, pp: &DualEmsmPublicParams<F, G>, witness: &[F]) -> Vec<F> {
        assert_eq!(witness.len(), pp.u_matrix.ncols);

        assert_eq!(self.lpn_instance.lpn_vector.len(), witness.len(), "Vectors must have equal length");

        self.lpn_instance.lpn_vector.iter()
            .zip(witness.iter())
            .map(|(a, b)| *a + *b)
            .collect()
    }

    pub fn recompute_msm<G: CurveGroup<ScalarField = F>>(
        &self,
        preprocessed_commitments: &PreprocessedCommitments<G>,
        commitment: G::Affine,
    ) -> G::Affine {
        let com_1 = preprocessed_commitments.p.commit_sparse(&self.lpn_instance.noise);

        (commitment - com_1).into()
    }

    // this function computes the msm in plaintext, can be used to benchmark and also for tests
    pub fn compute_msm_in_plaintext<G: CurveGroup<ScalarField = F>>(&self, pp: &DualEmsmPublicParams<F, G>, witness: &[F]) -> G::Affine {
        let mapped_witness = pp.u_matrix.right_multiply_vec(witness);

        G::msm(&pp.pedersen.generators, mapped_witness.as_slice())
            .expect("MSM computation failed")
            .into_affine()
    }
}

impl<F, G> DualEmsmPublicParams<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    pub fn new(
        n: usize,                       // columns of U = rows of T
        m: usize,                       // rows of U
        u_matrix: DenseMatrix<F>,       // Public U matrix
    ) -> Self {
        assert_eq!((u_matrix.nrows, u_matrix.ncols), (m, n));
        let rng = &mut thread_rng();

        // Construct the Primal LPN Index (T), we put N = 4 * n
        let index = DualLPNIndex::<F>::new(rng, n, 4 * n, 10);

        // Construct Pedersen generators
        let pedersen = Pedersen::<G>::new(m);

        DualEmsmPublicParams {
            index: index.clone(),
            u_matrix: u_matrix.clone(),
            pedersen: pedersen.clone(),
            n,
            N: 4 * n,
            m,
        }
    }

    pub fn preprocess(&self) -> PreprocessedCommitments<G> {
        // ⟨(U * T)_{*, i}, g⟩ for i in [0, N)
        let p: Vec<G::Affine> = (0..self.N).into_par_iter()
            .map(|i| {
                let t_col = self.index.get_columns_of_t(i);
                let h_col = self.u_matrix.right_multiply_vec(t_col.as_slice());
                G::msm(&self.pedersen.generators, h_col.as_slice())
                    .expect("MSM computation failed")
                    .into_affine()
            })
            .collect();

        PreprocessedCommitments {
            p: Pedersen{ generators: p }
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
    use crate::gadgets::emsm::dual::{DualEmsmInstance, DualEmsmPublicParams};
    use crate::gadgets::matrix::dense::DenseMatrix;

    #[test]
    fn preprocess() {
        let mut rng = thread_rng();
        let (m, non_zeros, n) = (50, 5, 30usize);
        let u_matrix = DenseMatrix::<F>::rand(m, n, &mut rng);

        // generate client/server state
        let pp = DualEmsmPublicParams::<F, G1Projective>::new(n, m, u_matrix.clone());
        let emsm_instance = DualEmsmInstance::<F>::new(&pp, non_zeros);

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
        assert_eq!(msm_in_plaintext, decrypted_msm);
    }
}

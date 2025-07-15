/*use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use crate::gadgets::lpn::primal_lpn::{PrimalLPNIndex, PrimalLPNInstance};
use crate::gadgets::matrix::dense::DenseMatrix;
use crate::gadgets::matrix::Matrix;
use crate::gadgets::pederson::Pedersen;

#[derive(Debug, Clone)]
pub struct PrimalSeverState<F, G>
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
pub struct PrimalClientState<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    // Public matrix U of size n * m
    pub u_matrix: DenseMatrix<F>,

    // LPN Index holding the sparse matrix T of size m * k
    pub index: PrimalLPNIndex<F>,
    pub lpn_instance: PrimalLPNInstance<F>,

    // Group generators for MSM of length n
    pub pedersen: Pedersen<G>,

    // Sizes
    pub n: usize,   // Rows of U
    pub m: usize,   // cols of U (also rows and columns of T)
    pub t: usize,   // Sparsity level for T (non-zeros per row)
}

pub fn generate_states<F, G>(
    n: usize,       // Rows of U
    m: usize,       // Rows of T (cols of U)
    t: usize,       // Sparsity level per row in T
    u_matrix: DenseMatrix<F>,       // Public U matrix
) -> (PrimalClientState<F, G>, PrimalSeverState<F, G>)
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    assert_eq!((u_matrix.nrows, u_matrix.ncols), (n, m));
    let rng = &mut thread_rng();

    // Construct the Primal LPN Index (T)
    let index = PrimalLPNIndex::<F>::new(rng, m, m, t);
    let instance = PrimalLPNInstance::new(rng, index.clone());

    // Construct Pedersen generators
    let pedersen = Pedersen::<G>::new(n);

    (
        PrimalClientState {
            index: index.clone(),
            lpn_instance: instance.clone(),
            u_matrix: u_matrix.clone(),
            pedersen: pedersen.clone(),
            n,
            m,
            t,
        },
        PrimalSeverState {
             index: index.clone(),
             u_matrix: u_matrix.clone(),
             pedersen: pedersen.clone(),
             n,
             m,
             t,
        }
    )
}


pub struct PreprocessedCommitments<G: CurveGroup> {
    pub p: Vec<G::Affine>,      // ⟨(U * T)_{*, i}, g⟩ for i in [0, m)
    pub p_prime: Vec<G::Affine> // ⟨U_{*, i}, g⟩ for i in [0, n)
}


impl<F, G> PrimalClientState<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    pub fn is_valid(&self) {
        let u_dims_match = self.u_matrix.nrows == self.n && self.u_matrix.ncols == self.m;
        let pedersen_size_ok = self.pedersen.generators.len() == self.n;
        let matrix_dims_match = self.index.rows == self.m && self.index.cols == self.m;
        let sparsity_ok = self.index.t == self.t;

        assert!(u_dims_match && pedersen_size_ok && matrix_dims_match && sparsity_ok)
    }

    pub fn client_phase1(&self, witness: &[F]) -> Vec<F> {
        assert_eq!(witness.len(), self.u_matrix.ncols);

        // Compute z_enc = T*s + e + z
        let t_s = self.index.t_matrix.matrix().right_multiply_vec(&self.lpn_instance.secret);
        let z_enc: Vec<F> = (0..self.index.t_matrix.matrix().cols)
            .map(|i| {
                t_s[i] + self.lpn_instance.noise[i] + witness[i]
            })
            .collect();

        z_enc
    }

    pub fn client_phase2(
        &self,
        preprocessed_commitments: PreprocessedCommitments<G>,
        commitment: G::Affine,
    ) -> G::Affine {
        let com_1 = G::msm(preprocessed_commitments.p.as_slice(), self.lpn_instance.secret.as_slice())
            .expect("MSM computation failed")
            .into_affine();

        let com_2 = G::msm(preprocessed_commitments.p_prime.as_slice(), self.lpn_instance.noise.as_slice())
            .expect("MSM computation failed")
            .into_affine();

        (commitment - com_1 - com_2).into()
    }

    // this function computes the msm in plaintext, can be used to benchmark and also for tests
    pub fn compute_msm_in_plaintext(&self, witness: &[F]) -> G::Affine {
        let mapped_witness = self.u_matrix.right_multiply_vec(witness);

        G::msm(&self.pedersen.generators, mapped_witness.as_slice())
            .expect("MSM computation failed")
            .into_affine()
    }
}

impl<F, G> PrimalSeverState<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
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

        PreprocessedCommitments { p, p_prime }
    }

    pub fn server_compute(&self, encrypted_witness: Vec<F>) -> G::Affine {
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
    use crate::gadgets::emsm::primal::{generate_states};
    use crate::gadgets::matrix::dense::DenseMatrix;

    #[test]
    fn preprocess() {
        let mut rng = thread_rng();
        let (m, t, n) = (5, 2, 3usize);
        let u_matrix = DenseMatrix::<F>::rand(n, m, &mut rng);

        // generate client/server state
        let (client_state, server_state) = generate_states::<F, G1Projective>(n, m, t, u_matrix.clone());

        // assert that the generated state is valid
        client_state.is_valid();

        // generate a random witness
        let witness = (0..m).map(|_| F::rand(&mut rng)).collect::<Vec<F>>();

        // generated the encrypted witness that is going to be passed to the server
        let encrypted_witness = client_state.client_phase1(witness.as_slice());

        // ensure the length of the witness is correct
        assert_eq!(encrypted_witness.len(), m);

        // server generating preprocessed commitments
        let preprocessed_commitments = server_state.preprocess();

        // server generating the encrypted msm
        let encrypted_msm = server_state.server_compute(encrypted_witness);

        // client decrypting the msm
        let decrypted_msm = client_state.client_phase2(preprocessed_commitments, encrypted_msm);

        // client computing the msm in plaintext
        let msm_in_plaintext = client_state.compute_msm_in_plaintext(witness.as_slice());

        // checking the equality
        assert_eq!(msm_in_plaintext, decrypted_msm);
    }
}
 */

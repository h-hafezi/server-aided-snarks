/*use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_std::rand::Rng;
use rand::thread_rng;
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
    pub index: PrimalLPNIndex<F>,

    // Group generators for MSM of length n
    pub pedersen: Pedersen<G>,

    // Sizes
    pub n: usize,   // Rows of U
    pub m: usize,   // Rows of T (cols of U)
    pub k: usize,   // Columns of T
    pub t: usize,   // Sparsity level for T (non-zeros per row)
}

#[derive(Debug, Clone)]
pub struct PrimalClientState<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    // LPN Index holding the sparse matrix T of size m * k
    pub index: PrimalLPNIndex<F>,
    pub lpn_instance: PrimalLPNInstance<F>,

    // Public matrix U of size n * m
    pub u_matrix: DenseMatrix<F>,

    // Group generators for MSM of length n
    pub pedersen: Pedersen<G>,

    // Sizes
    pub n: usize,   // Rows of U
    pub m: usize,   // Rows of T (cols of U)
    pub k: usize,   // Columns of T
    pub t: usize,   // Sparsity level for T (non-zeros per row)
}

pub fn generate_states<F, G>(
    n: usize,       // Rows of U
    m: usize,       // Rows of T (cols of U)
    k: usize,       // Columns of T
    t: usize,       // Sparsity level per row in T
    u_matrix: DenseMatrix<F>,       // Public U matrix
) -> (PrimalClientState<F, G>, PrimalSeverState<F, G>)
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    let rng = &mut thread_rng();

    // Construct the Primal LPN Index (T)
    let index = PrimalLPNIndex::<F>::new(rng, m, k, t);
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
            k,
            t,
        },
        PrimalSeverState {
             index: index.clone(),
             u_matrix: u_matrix.clone(),
             pedersen: pedersen.clone(),
             n,
             m,
             k,
             t,
         }
    )
}


pub struct PreprocessedCommitments<G: CurveGroup> {
    pub p: Vec<G::Affine>,      // ⟨(U * T)_{*, i}, g⟩ for i in [0, k)
    pub p_prime: Vec<G::Affine> // ⟨U_{*, i}, g⟩ for i in [0, n)
}


impl<F, G> PrimalClientState<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    pub fn is_valid(&self) -> bool {
        let u_dims_match = self.u_matrix.nrows == self.n && self.u_matrix.ncols == self.m;
        let pedersen_size_ok = self.pedersen.generators.len() == self.n;
        let matrix_dims_match = self.index.rows == self.m && self.index.cols == self.k;
        let sparsity_ok = self.index.t == self.t;
        u_dims_match && pedersen_size_ok && matrix_dims_match && sparsity_ok
    }

    pub fn client_phase1<R: Rng>(&self, rng: &mut R, witness: Vec<F>) -> Vec<F> {
        assert_eq!(witness.len(), self.index.t_matrix.cols);

        // Compute z_enc = T*s + e + z
        let t_s = self.index.t_matrix.right_multiply_vec(&self.lpn_instance.secret);
        let z_enc: Vec<F> = (0..self.index.t_matrix.cols)
            .map(|i| {
                t_s[i] + self.lpn_instance.noise[i] + witness[i]
            })
            .collect();

        z_enc
    }

    pub fn client_phase2(
        &self,
        preprocessed_commitments: PreprocessedCommitments<G>,
        _witness: Vec<F>,
        commitment: G::Affine,
    ) -> G::Affine {
        let com_1 = G::msm(preprocessed_commitments.p.as_slice(), self.lpn_instance.secret.as_slice())
            .expect("MSM computation failed")
            .into_affine();

        let com_2 = G::msm(preprocessed_commitments.p_prime.as_slice(), self.lpn_instance.noise.as_slice())
            .expect("MSM computation failed")
            .into_affine();

        ((commitment - com_1 - com_2)).into()
    }
}

impl<F, G> PrimalSeverState<F, G> where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>
{
    pub fn preprocess(&self) -> PreprocessedCommitments<G> {
        let p: Vec<G::Affine> = (0..self.k)
            .map(|i| {
                let t_col = self.index.t_matrix.get_column(i);
                let h_col = self.u_matrix.right_multiply_vec(t_col.as_slice());
                G::msm(&self.pedersen.generators, &h_col)
                    .expect("MSM computation failed")
                    .into_affine()
            })
            .collect();

        let p_prime: Vec<G::Affine> = (0..self.m)
            .map(|i| {
                let u_col = self.u_matrix.get_column(i);
                G::msm(&self.pedersen.generators, &u_col)
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


/*

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use crate::gadgets::lpn::primal_lpn::PrimalLPNIndex;
    use ark_bls12_381::{Fr as F, G1Projective};
    use crate::gadgets::emsm::primal::EmsmPrimalLPN;
    use crate::gadgets::matrix::Matrix;

    #[test]
    fn preprocess() {
        let mut rng = thread_rng();
        let (m, k, t, n) = (10, 10, 2, 10usize); // n = m
        let u_matrix = Matrix::<F>::identity(n);

        let emsm = EmsmPrimalLPN::<F, G1Projective>::new(&mut rng, n, m, k, t, u_matrix);
    }
}

 */

 */



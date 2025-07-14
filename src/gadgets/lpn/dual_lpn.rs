use ark_ff::PrimeField;
use rand::Rng;
use crate::gadgets::matrix::bidiagonal::UnitLowerBidiagonalMatrix;
use crate::gadgets::matrix::Matrix;
use crate::gadgets::matrix::t_sparse::TSparseMatrix;

#[derive(Debug, Clone)]
pub struct DualLPNIndex<F: PrimeField> {
    pub h1_matrix: TSparseMatrix<F>, // t-sparse matrix h_1 of size n * (N-n)
    pub h2_matrix: UnitLowerBidiagonalMatrix<F>,
    pub n: usize, // n
    pub N: usize, // N = 4 * n
    pub t: usize, // sparsity of H_1 matrix
}

impl<F: PrimeField> DualLPNIndex<F> {
    pub fn new<R: Rng>(rng: &mut R, n: usize, N: usize, t: usize) -> Self {
        let h1_matrix = TSparseMatrix::<F>::new(rng, n, N-n, t);
        let h2_matrix = UnitLowerBidiagonalMatrix::<F>::rand(n, rng);

        DualLPNIndex {
            h1_matrix,
            h2_matrix,
            n,
            N,
            t,
        }
    }

    // T = I_{n*n} || H_2^{-1}H_1
    pub fn get_columns_of_T(&self, i: usize) -> Vec<F> {
        assert!(i < self.N, "i must be less than N");

        if i < self.n {
            // Return i-th column of identity
            (0..self.n).map(|j| if j == i { F::ONE } else { F::ZERO }).collect()
        } else {
            let col = self.h1_matrix.matrix().get_column(i-self.n);
            self.h2_matrix.right_multiply_inverse_vec(col.as_slice())
        }
     }
}

#[derive(Debug, Clone)]
pub struct DualLPNInstance<F: PrimeField> {
    pub index: DualLPNIndex<F>,
    pub noise: Vec<F>, // vector of noise of size N
    pub lpn_vector: Vec<F>, // [I, H_2^{-1} H_1] * noise
}

impl<F: PrimeField> DualLPNInstance<F> {
    pub fn new<R: Rng>(rng: &mut R, index: DualLPNIndex<F>) -> Self {
        // Noise vector as field elements 0 or 1
        let noise: Vec<F> = (0..index.N)
            .map(|_| if rng.gen_bool(0.5) { F::ONE } else { F::ZERO })
            .collect();

        let noise_1: Vec<F> = noise[0..index.n].to_vec();
        let noise_2: Vec<F> = noise[index.n..index.N].to_vec();

        let h1_e = index.h1_matrix.matrix().right_multiply_vec(noise_2.as_slice());

        // asserting h1_e is well-foramtted
        assert_eq!(h1_e.len(), index.h2_matrix.cols() - 1);

        // compute H_2^{-1} (H_1.e)
        let z = index.h2_matrix.right_multiply_inverse_vec(h1_e.as_slice());

        let lpn_vector: Vec<F> = z.iter()
            .zip(noise_1.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        DualLPNInstance {
            index,
            noise,
            lpn_vector,
        }
    }
}


#[cfg(test)]
mod test{
    use rand::thread_rng;
    use crate::gadgets::lpn::dual_lpn::DualLPNIndex;
    use ark_bls12_381::Fr as F;
    use crate::gadgets::matrix::Matrix;

    #[test]
    fn test_get_columns_of_T() {
        let n = 4;
        let N = 16; // e.g., 4n
        let t = 1;
        let mut rng = thread_rng();

        let index = DualLPNIndex::<F>::new(&mut rng, n, N, t);

        // Get H1 as a dense matrix for comparison
        let h1_dense = index.h1_matrix.matrix();

        // Prepare reconstructed H1 columns
        let mut reconstructed_h1_columns: Vec<Vec<F>> = Vec::new();

        for i in n..N {
            // Compute T[:,i] = H2^{-1} H1[:,i-n]
            let t_col = index.get_columns_of_T(i);

            // Compute H2 * T[:,i]
            let h2_times_tcol = index.h2_matrix.right_multiply_vec(&t_col);

            index.h2_matrix.print();
            println!("t_col: {:?}", t_col);

            // Should equal H1[:,i-n]
            let h1_col = h1_dense.get_column(i - n);

            assert_eq!(
                h2_times_tcol, h1_col,
                "H2 * T[:,{}] != H1[:,{}]",
                i,
                i - n
            );

            reconstructed_h1_columns.push(h2_times_tcol);
        }
    }
}
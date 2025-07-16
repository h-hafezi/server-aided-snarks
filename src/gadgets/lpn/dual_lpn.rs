use ark_ff::PrimeField;
use rand::Rng;
use crate::gadgets::matrix::bidiagonal::UnitLowerBidiagonalMatrix;
use crate::gadgets::matrix::Matrix;
use crate::gadgets::matrix::t_sparse::TSparseMatrix;
use crate::gadgets::sparse_vec::sparse_vec::SparseVector;

#[derive(Debug, Clone)]
pub struct DualLPNIndex<F: PrimeField> {
    pub h1_matrix: TSparseMatrix<F>, // t-sparse matrix h_1 of size n * (N-n)
    pub h2_matrix: UnitLowerBidiagonalMatrix<F>,
    pub n: usize, // n
    pub N: usize, // N = 4 * n
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
        }
    }

    // T = I_{n*n} || H_2^{-1}H_1
    pub fn get_columns_of_t(&self, i: usize) -> Vec<F> {
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
    pub noise: SparseVector<F>, // vector of noise of size N
    pub lpn_vector: Vec<F>, // [I, H_2^{-1} H_1] * noise
}

impl<F: PrimeField> DualLPNInstance<F> {
    pub fn new(index: DualLPNIndex<F>, noise: SparseVector<F>) -> Self {
        // ensure the noise has the right format
        assert_eq!(index.N, noise.size);

        // Build noise vector, it will throw error if N < 2^{10} or N > 2^{20}
        let noise_dense = noise.into_dense();

        let noise_1 = &noise_dense[0..index.n];
        let noise_2 = &noise_dense[index.n..index.N];

        // Compute h1_e = H1 * noise_2
        let h1_e = index.h1_matrix.matrix().right_multiply_vec(noise_2);

        // Compute z = H2^{-1} * h1_e
        let z = index.h2_matrix.right_multiply_inverse_vec(h1_e.as_slice());

        // Efficiently compute lpn_vector = z + noise_1
        let lpn_vector: Vec<F> = noise_1.iter()
            .zip(z.iter())
            .map(|(&n1, &z)| n1 + z)
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
    use ark_bls12_381::Fr as F;
    use crate::gadgets::matrix::Matrix;
    use crate::gadgets::sparse_vec::sparse_vec::SparseVector;
    use crate::gadgets::lpn::dual_lpn::{DualLPNIndex, DualLPNInstance};

    #[test]
    fn test_dual_lpn() {
        let rng = &mut thread_rng();
        let (t, n, N) = (10, 1024 * 1024, 4 * 1024 * 1024);
        let index = DualLPNIndex::<F>::new(rng, n, N, t);
        let error = SparseVector::error_vec(N, t, rng);
        let _ = DualLPNInstance::new(index, error);
    }

    #[test]
    fn test_get_columns_of_t() {
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
            let t_col = index.get_columns_of_t(i);

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
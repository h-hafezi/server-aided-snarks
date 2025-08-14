use ark_ff::PrimeField;
use crate::emsm::raa_code::TOperator;
use crate::emsm::sparse_vec::SparseVector;

#[derive(Debug, Clone)]
pub struct DualLPNInstance<F: PrimeField> {
    pub noise: SparseVector<F>, // vector of noise of size N
    pub lpn_vector: Vec<F>, 
}

impl<F: PrimeField> DualLPNInstance<F> {
    pub fn new(t_operator: &TOperator<F>, noise: SparseVector<F>) -> Self {
        // ensure the noise has the right format
        assert_eq!(t_operator.N, noise.size);

        let noise_dense = noise.into_dense();


        // Efficiently compute lpn_vector = z + noise_1
        let lpn_vector: Vec<F> = t_operator.multiply_sparse(noise_dense);
        
        // return the instance
        DualLPNInstance {
            noise,
            lpn_vector,
        }
    }
}


#[cfg(test)]
mod test{
    use rand::thread_rng;
    use ark_bls12_381::Fr as F;
    use crate::emsm::sparse_vec::SparseVector;
    use crate::emsm::dual_lpn::{DualLPNInstance};
    use crate::emsm::raa_code::TOperator;

    #[test]
    fn test_dual_lpn() {
        let rng = &mut thread_rng();
        let (t, n, N) = (10, 1024 * 1024, 4 * 1024 * 1024);
        let index = TOperator::<F>::rand(n);
        let error = SparseVector::error_vec(N, t, rng);
        let _ = DualLPNInstance::new(&index, error);
    }
}
 
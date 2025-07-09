use ark_ff::PrimeField;
use ark_std::rand::Rng;
use crate::gadgets::matrix::Matrix;
use crate::gadgets::matrix::t_sparse::TSparseMatrix;

#[derive(Debug, Clone)]
pub struct PrimalLPNIndex<F: PrimeField> {
    pub t_matrix: TSparseMatrix<F>,
    pub rows: usize,
    pub cols: usize,
    pub t: usize,
}

impl<F: PrimeField> PrimalLPNIndex<F> {
    pub fn new<R: Rng>(rng: &mut R, rows: usize, cols: usize, t: usize) -> Self {
        let t_matrix = TSparseMatrix::<F>::new(rng, rows, cols, t);
        Self { t_matrix, rows, cols, t }
    }
}

#[derive(Debug, Clone)]
pub struct PrimalLPNInstance<F: PrimeField> {
    pub index: PrimalLPNIndex<F>,
    pub secret: Vec<F>,
    pub noise: Vec<F>,
    pub lpn_vector: Vec<F>,
}

impl<F: PrimeField> PrimalLPNInstance<F> {
    pub fn new<R: Rng>(rng: &mut R, index: PrimalLPNIndex<F>) -> Self {
        let secret: Vec<F> = (0..index.cols).map(|_| F::rand(rng)).collect();

        // Noise vector as field elements 0 or 1
        let noise: Vec<F> = (0..index.rows)
            .map(|_| if rng.gen_bool(0.5) { F::ONE } else { F::ZERO })
            .collect();

        let ts = index.t_matrix.matrix().right_multiply_vec(&secret);

        // lpn_vector = ts + noise (element-wise addition)
        let lpn_vector: Vec<F> = ts
            .iter()
            .zip(noise.iter())
            .map(|(ts_i, e_i)| *ts_i + *e_i)
            .collect();

        Self { index, secret, noise, lpn_vector }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_primal_lpn_modular() {
        let mut rng = StdRng::seed_from_u64(42);

        let rows = 5;
        let cols = 8;
        let t = 1;

        let index = PrimalLPNIndex::<Fr>::new(&mut rng, rows, cols, t);
        println!("T = ");
        index.t_matrix.matrix().print();

        let instance = PrimalLPNInstance::new(&mut rng, index);
        println!("Secret s = {:?}", instance.secret);
        println!("Noise e = {:?}", instance.noise);
        println!("LPN vector (T*s + e) = {:?}", instance.lpn_vector);
    }
}


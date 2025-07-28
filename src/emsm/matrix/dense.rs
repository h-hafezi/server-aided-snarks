use ark_ff::PrimeField;
use ark_std::{vec, vec::Vec};
use rand::Rng;
use rayon::prelude::*;
use crate::emsm::matrix::Matrix;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseMatrix<F: PrimeField> {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<F>, // Row-major order
}

impl<F: PrimeField> Matrix<F> for DenseMatrix<F> {
    type Input<'a> = &'a [F] where F: 'a;

    fn new(data: Self::Input<'_>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            nrows: rows,
            ncols: cols,
            data: data.to_vec(),
        }
    }

    fn get_row(&self, row: usize) -> Vec<F> {
        assert!(row < self.nrows);
        let start = row * self.ncols;
        self.data[start..start + self.ncols].to_vec()
    }

    fn get_column(&self, col: usize) -> Vec<F> {
        assert!(col < self.ncols);
        (0..self.nrows).map(|i| self.get(i, col)).collect()
    }

    fn print(&self) {
        for i in 0..self.nrows {
            print!("[ ");
            for j in 0..self.ncols {
                print!("{} ", self.get(i, j));
            }
            println!("]");
        }
    }

    fn right_multiply_vec(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(self.ncols, vec.len());
        (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                self.get_row(i)
                    .iter()
                    .zip(vec)
                    .map(|(a, b)| *a * b)
                    .sum()
            })
            .collect()
    }

    fn left_multiply_vec(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(self.nrows, vec.len());
        (0..self.ncols)
            .into_par_iter()
            .map(|j| {
                (0..self.nrows)
                    .map(|i| vec[i] * self.get(i, j))
                    .sum()
            })
            .collect()
    }

    fn cols(&self) -> usize {
        self.ncols
    }

    fn rows(&self) -> usize {
        self.nrows
    }

    fn transpose(&self) -> Self {
        let mut transposed_data = vec![F::zero(); self.nrows * self.ncols];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                transposed_data[j * self.nrows + i] = self.get(i, j);
            }
        }

        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            data: transposed_data,
        }
    }
}


impl<F: PrimeField> DenseMatrix<F> {
    /// Returns the (i, j)-th element (0-indexed)
    pub fn get(&self, i: usize, j: usize) -> F {
        assert!(i < self.nrows && j < self.ncols);
        self.data[i * self.ncols + j]
    }

    /// Sets the (i, j)-th element
    pub fn set(&mut self, i: usize, j: usize, value: F) {
        assert!(i < self.nrows && j < self.ncols);
        self.data[i * self.ncols + j] = value;
    }

    /// Create an identity matrix of size n x n
    pub fn identity(n: usize) -> Self {
        let mut data = vec![F::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = F::one();
        }
        Self {
            nrows: n,
            ncols: n,
            data,
        }
    }

    /// Generates a random matrix with given dimensions
    pub fn rand<R: Rng + ?Sized>(nrows: usize, ncols: usize, rng: &mut R) -> Self {
        let data = (0..nrows * ncols)
            .map(|_| F::rand(rng))
            .collect();
        Self { nrows, ncols, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero, UniformRand};
    use ark_std::test_rng;

    fn get_sample_matrix() -> DenseMatrix<Fr> {
        let elements: Vec<Fr> = (1..=9).map(|x| Fr::from(x)).collect();
        DenseMatrix::new(elements.as_slice(), 3, 3)
    }

    #[test]
    fn test_identity() {
        let id = DenseMatrix::<Fr>::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { Fr::one() } else { Fr::zero() };
                assert_eq!(id.get(i, j), expected);
            }
        }
    }

    #[test]
    fn test_get_row_and_column() {
        let matrix = get_sample_matrix();

        let expected_row = vec![Fr::from(4), Fr::from(5), Fr::from(6)];
        let expected_col = vec![Fr::from(3), Fr::from(6), Fr::from(9)];

        assert_eq!(matrix.get_row(1), expected_row);
        assert_eq!(matrix.get_column(2), expected_col);
    }

    #[test]
    fn test_right_vector_multiplication() {
        let matrix = get_sample_matrix();
        let vector = vec![Fr::from(1), Fr::zero(), Fr::from(1)];
        let expected = vec![Fr::from(4), Fr::from(10), Fr::from(16)];

        let result = matrix.right_multiply_vec(&vector);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_left_vector_multiplication() {
        let matrix = get_sample_matrix();
        let vector = vec![Fr::from(1), Fr::zero(), Fr::from(1)];
        let expected = vec![Fr::from(8), Fr::from(10), Fr::from(12)];

        let result = matrix.left_multiply_vec(&vector);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_set_and_get() {
        let mut matrix = DenseMatrix::<Fr>::identity(2);
        matrix.set(0, 1, Fr::from(42));

        assert_eq!(matrix.get(0, 1), Fr::from(42));
        assert_eq!(matrix.get(1, 0), Fr::zero());
    }

    #[test]
    fn test_transpose_vector_multiplication_property() {
        let mut rng = test_rng();
        let matrix = get_sample_matrix();
        let matrix_t = matrix.transpose();

        let vector: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();

        let left = matrix_t.left_multiply_vec(&vector);
        let right = matrix.right_multiply_vec(&vector);

        assert_eq!(left, right);
    }
}

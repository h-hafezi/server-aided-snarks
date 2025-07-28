use crate::emsm::matrix::dense::DenseMatrix;
use crate::emsm::matrix::Matrix;
use ark_ff::PrimeField;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitLowerBidiagonalMatrix<F: PrimeField> {
    pub n: usize,
    pub a: Vec<F>, // length n-1
}

impl<F: PrimeField> UnitLowerBidiagonalMatrix<F> {
    /// Construct dense matrix representation
    pub fn construct_dense_matrix(&self) -> DenseMatrix<F> {
        let mut data = vec![F::zero(); self.n * self.n];
        for i in 0..self.n {
            data[i * self.n + i] = F::one(); // diagonal
            if i > 0 {
                data[i * self.n + (i-1)] = self.a[i-1]; // subdiagonal
            }
        }
        DenseMatrix {
            nrows: self.n,
            ncols: self.n,
            data,
        }
    }

    /// Construct inverse dense matrix
    pub fn construct_inverse_dense_matrix(&self) -> DenseMatrix<F> {
        let mut data = vec![F::zero(); self.n * self.n];
        for i in 0..self.n {
            for j in 0..=i {
                let val = if i == j {
                    F::one()
                } else {
                    let mut prod = F::one();
                    for k in j..i {
                        prod *= self.a[k];
                    }
                    if (i-j) % 2 == 1 {
                        prod = -prod;
                    }
                    prod
                };
                data[i * self.n + j] = val;
            }
        }
        DenseMatrix {
            nrows: self.n,
            ncols: self.n,
            data,
        }
    }

    /// Efficient right multiply: H₂ · vec
    pub fn right_multiply_vec(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(vec.len(), self.n);
        let mut out = vec![F::zero(); self.n];
        out[0] = vec[0];
        for i in 1..self.n {
            out[i] = self.a[i-1] * vec[i-1] + vec[i];
        }
        out
    }

    /// Efficient right multiply: H₂⁻¹ · vec
    pub fn right_multiply_inverse_vec(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(vec.len(), self.n);
        let mut out = vec![F::zero(); self.n];
        out[0] = vec[0];
        for i in 1..self.n {
            out[i] = vec[i] - self.a[i-1] * out[i-1];
        }
        out
    }

    /// Generate a random unit lower bidiagonal matrix of size `n`
    pub fn rand<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Self {
        assert!(n >= 1, "Matrix size must be at least 1");
        let a: Vec<F> = (0..n-1).map(|_| F::rand(rng)).collect();

        Self { n, a }
    }
}

impl<F: PrimeField> Matrix<F> for UnitLowerBidiagonalMatrix<F> {
    type Input<'a> = (&'a [F], usize) where F: 'a;

    fn new(input: Self::Input<'_>, rows: usize, cols: usize) -> Self {
        assert_eq!(rows, cols);
        assert_eq!(input.0.len(), rows - 1);
        Self {
            n: rows,
            a: input.0.to_vec(),
        }
    }

    fn get_row(&self, row: usize) -> Vec<F> {
        assert!(row < self.n);
        let mut r = vec![F::zero(); self.n];
        r[row] = F::one();
        if row > 0 {
            r[row-1] = self.a[row-1];
        }
        r
    }

    fn get_column(&self, col: usize) -> Vec<F> {
        assert!(col < self.n);
        let mut c = vec![F::zero(); self.n];
        c[col] = F::one();
        if col + 1 < self.n {
            c[col+1] = self.a[col];
        }
        c
    }

    fn print(&self) {
        self.construct_dense_matrix().print();
    }

    fn right_multiply_vec(&self, vec: &[F]) -> Vec<F> {
        self.right_multiply_vec(vec)
    }

    fn left_multiply_vec(&self, _vec: &[F]) -> Vec<F> {
        unimplemented!("Left multiply for bidiagonal not implemented")
    }

    fn cols(&self) -> usize {
        self.n
    }

    fn rows(&self) -> usize {
        self.n
    }

    fn transpose(&self) -> Self {
        unimplemented!("Transpose for bidiagonal not implemented")
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr as F;
    use ark_ff::{One, UniformRand, Zero};
    use rand::thread_rng;
    use crate::emsm::matrix::bidiagonal::UnitLowerBidiagonalMatrix;
    use crate::emsm::matrix::dense::DenseMatrix;
    use crate::emsm::matrix::Matrix;

    #[test]
    fn test_unit_lower_bidiagonal_matrix() {
        // Construct a 4×4 unit lower bidiagonal matrix
        let a = vec![F::from(2u64), F::from(3u64), F::from(4u64)];
        let bidiag = UnitLowerBidiagonalMatrix::<F> { n: 4, a };

        println!("Bi-diagonal matrix:");
        bidiag.print();

        // Dense version
        let dense = bidiag.construct_dense_matrix();
        println!("Dense version:");
        dense.print();

        let dense_inv = bidiag.construct_inverse_dense_matrix();
        println!("Dense inverse:");
        dense_inv.print();

        let identity = DenseMatrix::<F>::identity(4);

        // Check that H⁻¹H = I column by column
        for col in 0..4 {
            let col_vec = dense.get_column(col);
            let inv_times_col = dense_inv.right_multiply_vec(&col_vec);

            let expected = identity.get_column(col);
            assert_eq!(inv_times_col, expected, "H⁻¹H column {} failed", col);
        }

        // Test get_row and get_column
        let expected_rows = vec![
            vec![F::one(), F::zero(), F::zero(), F::zero()],
            vec![F::from(2u64), F::one(), F::zero(), F::zero()],
            vec![F::zero(), F::from(3u64), F::one(), F::zero()],
            vec![F::zero(), F::zero(), F::from(4u64), F::one()],
        ];
        for i in 0..4 {
            assert_eq!(bidiag.get_row(i), expected_rows[i], "Row {} mismatch", i);
        }

        let expected_cols = vec![
            vec![F::one(), F::from(2u64), F::zero(), F::zero()],
            vec![F::zero(), F::one(), F::from(3u64), F::zero()],
            vec![F::zero(), F::zero(), F::one(), F::from(4u64)],
            vec![F::zero(), F::zero(), F::zero(), F::one()],
        ];
        for j in 0..4 {
            assert_eq!(bidiag.get_column(j), expected_cols[j], "Column {} mismatch", j);
        }

        // Random vector r
        let mut rng = thread_rng();
        let r: Vec<F> = (0..4).map(|_| F::rand(&mut rng)).collect();

        // Compute H * r using both bidiagonal and dense
        let bidiag_times_r = bidiag.right_multiply_vec(&r);
        let dense_times_r = dense.right_multiply_vec(&r);
        assert_eq!(bidiag_times_r, dense_times_r, "H*r mismatch");

        // Compute H⁻¹ * r using bidiagonal and dense_inv
        let bidiag_inv_times_r = bidiag.right_multiply_inverse_vec(&r);
        let dense_inv_times_r = dense_inv.right_multiply_vec(&r);
        assert_eq!(bidiag_inv_times_r, dense_inv_times_r, "H⁻¹*r mismatch");

        let res = dense.right_multiply_vec(bidiag_inv_times_r.as_slice());
        assert_eq!(res, r);
    }
}
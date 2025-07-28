use std::collections::HashSet;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::Rng;
use crate::emsm::matrix::sparse::SparseMatrix;

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TSparseMatrix<F: PrimeField> {
    pub t: usize,
    pub matrix: SparseMatrix<F>,
}

impl<F: PrimeField> TSparseMatrix<F> {
    /// Generates a random matrix with `rows` rows and `cols` columns,
    /// where each row has exactly `t` non-zero uniformly random entries.
    ///
    /// Panics if `t > cols` because it's impossible to select `t` distinct columns.
    pub fn new<R: Rng>(
        rng: &mut R,
        rows: usize,
        cols: usize,
        t: usize,
    ) -> Self {
        assert!(
            t <= cols,
            "Cannot create a row with {} non-zero elements in {} columns.", t, cols
        );

        let mut data = Vec::with_capacity(rows * t);
        let mut indices = Vec::with_capacity(rows * t);
        let mut indptr = Vec::with_capacity(rows + 1);
        indptr.push(0);

        for _ in 0..rows {
            // Select `t` unique column indices
            let mut selected_cols = HashSet::with_capacity(t);
            while selected_cols.len() < t {
                let col = rng.gen_range(0..cols);
                selected_cols.insert(col);
            }

            // Sort the column indices for CSR format
            let mut selected_cols: Vec<usize> = selected_cols.into_iter().collect();
            selected_cols.sort_unstable();

            for &col in &selected_cols {
                let val = F::rand(rng);
                data.push(val);
                indices.push(col);
            }

            let last_ptr = *indptr.last().unwrap();
            indptr.push(last_ptr + t);
        }

        TSparseMatrix {
            t,
            matrix: SparseMatrix {
                data,
                indices,
                indptr,
                cols,
            }
        }
    }

    pub fn matrix(&self) -> &SparseMatrix<F> {
        &self.matrix
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use crate::emsm::matrix::Matrix;
    use crate::emsm::matrix::t_sparse::TSparseMatrix;

    #[test]
    fn test_t_sparse_row_matrix_and_print() {
        // Seed RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(42);

        // Parameters
        let rows = 5;
        let cols = 8;
        let t = 2; // number of non-zero entries per row

        // Generate the t-sparse matrix
        let sparse_matrix = TSparseMatrix::<Fr>::new(&mut rng, rows, cols, t);

        // Print it using your sparse_print function
        sparse_matrix.matrix().print();
    }
}



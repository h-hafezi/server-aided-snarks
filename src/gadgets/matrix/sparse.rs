//! Borrowed from Nexus

use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use crate::gadgets::matrix::Matrix;
use crate::gadgets::sparse_vec::sparse_vec::SparseVector;

pub type MatrixRef<'a, F> = &'a [Vec<(F, usize)>];

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatrix<F: PrimeField> {
    /// all non-zero values in the matrix
    pub data: Vec<F>,
    /// column indices
    pub indices: Vec<usize>,
    /// row information
    pub indptr: Vec<usize>,
    /// number of columns
    pub cols: usize,
}

impl<F: PrimeField> Matrix<F> for SparseMatrix<F> {
    type Input<'a> = MatrixRef<'a, F>;

    /// Construct from the COO representation;
    /// We assume that the rows are sorted during construction.
    fn new(matrix: Self::Input<'_>, rows: usize, cols: usize) -> Self {
        let mut new_matrix = vec![vec![]; rows];
        let matrix_iter = matrix
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().map(move |&(f, j)| (i, j, f)));

        for (row, col, val) in matrix_iter {
            new_matrix[row].push((col, val));
        }

        for row in new_matrix.iter() {
            assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
        }

        let mut indptr = vec![0; rows + 1];
        for (i, row) in new_matrix.iter().enumerate() {
            indptr[i + 1] = indptr[i] + row.len();
        }

        let mut indices = vec![];
        let mut data = vec![];
        for row in new_matrix {
            let (idx, val): (Vec<usize>, Vec<F>) = row.into_iter().unzip();
            indices.extend(idx);
            data.extend(val);
        }

        SparseMatrix { data, indices, indptr, cols }
    }

    fn get_row(&self, row: usize) -> Vec<F> {
        assert!(row < self.rows(), "Row index out of bounds");

        let mut dense_row = vec![F::zero(); self.cols];
        let start = self.indptr[row];
        let end = self.indptr[row + 1];

        for idx in start..end {
            let col = self.indices[idx];
            dense_row[col] = self.data[idx];
        }

        dense_row
    }

    fn get_column(&self, col: usize) -> Vec<F> {
        assert!(col < self.cols, "Column index out of bounds");

        let rows = self.indptr.len() - 1;
        let mut column_vec = vec![F::zero(); rows];

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                if self.indices[idx] == col {
                    column_vec[row] = self.data[idx];
                    break; // At most one entry per (row, col) in CSR
                }
            }
        }

        column_vec
    }

    fn print(&self) {
        let rows = self.indptr.len() - 1;

        for row in 0..rows {
            // Initialize the row with zeros
            let mut dense_row = vec![F::zero(); self.cols];

            // Fill in the non-zero entries from CSR
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            for idx in start..end {
                let col = self.indices[idx];
                dense_row[col] = self.data[idx];
            }

            // Print the row
            let row_str = dense_row
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<_>>()
                .join(", ");
            println!("Row {}: [{}]", row, row_str);
        }
    }

    /// Multiply by a dense vector; M * v where v is a column vector
    fn right_multiply_vec(&self, vector: &[F]) -> Vec<F> {
        assert_eq!(self.cols, vector.len());

        let iter = self.indptr.par_windows(2);
        iter.map(|ptrs| {
            self.get_row_unchecked(ptrs.try_into().unwrap())
                .map(|(val, col_idx)| *val * vector[*col_idx])
                .sum()
        }).collect()
    }

    /// Multiply by a dense vector; v * M where v is a row vector
    fn left_multiply_vec(&self, vector: &[F]) -> Vec<F> {
        assert_eq!(vector.len(), self.indptr.len() - 1);

        let mut result = vec![F::zero(); self.cols];

        // For thread-safe parallel accumulation, use a Mutex or a lock-free structure
        // but simplest to do sequentially for now:

        self.indptr.windows(2).enumerate().for_each(|(row, window)| {
            let start = window[0];
            let end = window[1];
            let scale = vector[row];
            for idx in start..end {
                let col = self.indices[idx];
                let val = self.data[idx];
                result[col] += scale * val;
            }
        });

        result
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        self.indptr.len() - 1
    }

    fn transpose(&self) -> Self {
        let rows = self.indptr.len() - 1;
        let cols = self.cols;
        let nnz = self.data.len();

        // Step 1: Count how many entries are in each column (will become rows in the transposed matrix)
        let mut col_counts = vec![0; cols];
        for &col in &self.indices {
            col_counts[col] += 1;
        }

        // Step 2: Build the indptr for the transposed matrix
        let mut trans_indptr = Vec::with_capacity(cols + 1);
        trans_indptr.push(0);
        for &count in &col_counts {
            trans_indptr.push(trans_indptr.last().unwrap() + count);
        }

        // Step 3: Prepare data and indices arrays
        let mut trans_data = vec![F::zero(); nnz];
        let mut trans_indices = vec![0usize; nnz];

        // Temporary counter to track positions in transposed rows
        let mut current_positions = trans_indptr[..cols].to_vec();

        // Step 4: Fill data and indices
        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                let val = self.data[idx];

                let pos = current_positions[col];
                trans_data[pos] = val;
                trans_indices[pos] = row;

                current_positions[col] += 1;
            }
        }

        Self {
            data: trans_data,
            indices: trans_indices,
            indptr: trans_indptr,
            cols: rows, // number of columns becomes number of rows
        }
    }
}

impl<F: PrimeField> SparseMatrix<F> {
    /// number of non-zero entries
    pub fn len(&self) -> usize {
        *self.indptr.last().unwrap()
    }

    /// empty matrix
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a custom iterator
    pub fn iter(&self) -> Iter<'_, F> {
        let mut row = 0;
        while self.indptr[row + 1] == 0 {
            row += 1;
        }
        Iter {
            matrix: self,
            row,
            i: 0,
            nnz: *self.indptr.last().unwrap(),
        }
    }

    /// Retrieves the data for row slice [i..j] from `ptrs`.
    /// We assume that `ptrs` is indexed from `indptrs` and do not check if the
    /// returned slice is actually a valid row.
    pub fn get_row_unchecked(&self, ptrs: &[usize; 2]) -> impl Iterator<Item=(&F, &usize)> {
        self.data[ptrs[0]..ptrs[1]]
            .iter()
            .zip(&self.indices[ptrs[0]..ptrs[1]])
    }
}

/// Iterator for sparse matrix
pub struct Iter<'a, F: PrimeField> {
    matrix: &'a SparseMatrix<F>,
    row: usize,
    i: usize,
    nnz: usize,
}

impl<'a, F: PrimeField> Iterator for Iter<'a, F> {
    type Item = (usize, usize, F);

    fn next(&mut self) -> Option<Self::Item> {
        // are we at the end?
        if self.i == self.nnz {
            return None;
        }

        // compute current item
        let curr_item = (
            self.row,
            self.matrix.indices[self.i],
            self.matrix.data[self.i],
        );

        // advance the iterator
        self.i += 1;
        // edge case at the end
        if self.i == self.nnz {
            return Some(curr_item);
        }
        // if `i` has moved to next row
        while self.i >= self.matrix.indptr[self.row + 1] {
            self.row += 1;
        }

        Some(curr_item)
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use ark_ff::Zero;
    use ark_std::UniformRand;
    use rand::thread_rng;
    use std::time::Instant;
    use crate::gadgets::matrix::dense::DenseMatrix;
    use crate::gadgets::matrix::t_sparse::TSparseMatrix;
    use super::*;

    type F = Fr;

    #[test]
    fn test_matrix() {
        let matrix_data = vec![
            vec![(F::from(2u64), 1usize)],
            vec![(F::from(3u64), 2usize)],
            vec![(F::from(4u64), 0usize)],
        ];
        let sparse_matrix = SparseMatrix::<F>::new(&matrix_data, 3, 3);

        assert_eq!(
            sparse_matrix.data,
            vec![F::from(2), F::from(3), F::from(4)]
        );
        assert_eq!(sparse_matrix.indices, vec![1, 2, 0]);
        assert_eq!(sparse_matrix.indptr, vec![0, 1, 2, 3]);

        // Construct the full dense matrix for verification
        let mut dense_matrix = vec![vec![F::zero(); 3]; 3];
        for (i, row) in matrix_data.iter().enumerate() {
            for &(val, j) in row {
                dense_matrix[i][j] = val;
            }
        }

        // Check get_row against dense_matrix
        for row in 0..3 {
            let expected_row = &dense_matrix[row];
            let actual_row = sparse_matrix.get_row(row);
            assert_eq!(&actual_row, expected_row, "Mismatch in row {row}");
        }

        // Check get_column against dense_matrix
        for col in 0..3 {
            let expected_col: Vec<F> = dense_matrix.iter().map(|row| row[col]).collect();
            let actual_col = sparse_matrix.get_column(col);
            assert_eq!(actual_col, expected_col, "Mismatch in column {col}");
        }
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        // Sparse matrix in COO format
        let matrix_data = vec![
            vec![(F::from(2), 1), (F::from(7), 2)],
            vec![(F::from(3), 2)],
            vec![(F::from(4), 0)],
        ];
        let sparse_matrix = SparseMatrix::<F>::new(&matrix_data, 3, 3);

        // Equivalent dense matrix
        let dense_elements: Vec<F> = vec![
            0, 2, 7,
            0, 0, 3,
            4, 0, 0,
        ]
            .into_iter()
            .map(F::from)
            .collect();

        let dense_matrix = DenseMatrix::new(dense_elements.as_slice(), 3, 3);

        // Input vector
        let vector = vec![F::from(1), F::from(2), F::from(3)];

        // Test right multiplication: M * v
        let sparse_right = sparse_matrix.right_multiply_vec(&vector);
        let dense_right = dense_matrix.right_multiply_vec(&vector);

        assert_eq!(sparse_right, dense_right, "Sparse and dense right multiplication differ");
        assert_eq!(sparse_right, vec![F::from(25), F::from(9), F::from(4)], "Right multiplication result incorrect");

        // Test left multiplication: v^T * M
        let sparse_left = sparse_matrix.left_multiply_vec(&vector);
        let dense_left = dense_matrix.left_multiply_vec(&vector);

        assert_eq!(sparse_left, dense_left, "Sparse and dense left multiplication differ");
        assert_eq!(sparse_left, vec![F::from(12), F::from(2), F::from(13)], "Left multiplication result incorrect");
    }

    #[test]
    fn test_transpose() {

        let mut rng = thread_rng();

        // Parameters
        let rows = 4;
        let cols = 5;
        let sparsity = 2; // each row has exactly 2 non-zeros

        // Random vector u of length rows
        let u: Vec<Fr> = (0..cols).map(|_| Fr::rand(&mut rng)).collect();

        // Random sparse matrix of size (rows x cols)
        let t_sparse = TSparseMatrix::<Fr>::new(&mut rng, rows, cols, sparsity);
        t_sparse.matrix.print();
        let sparse_matrix = t_sparse.matrix();

        // Compute the transpose
        let sparse_transpose = sparse_matrix.transpose();

        // Verify: (T * v)[i] == (vᵗ * Tᵗ)[i] for all i
        // This follows from (T * v) = Tᵗᵗ * v  <=> uᵗ * T == (Tᵗ * u)ᵗ
        let t_u = sparse_matrix.right_multiply_vec(&u);
        let u_t_t = sparse_transpose.left_multiply_vec(&u);

        assert_eq!(
            t_u, u_t_t,
            "T * u != uᵗ * Tᵗ"
        );

        // Build dense equivalent of sparse_matrix
        let mut dense_elements = vec![Fr::zero(); rows * cols];
        for (row, col, val) in sparse_matrix.iter() {
            dense_elements[row * cols + col] = val;
        }
        let dense_matrix = DenseMatrix::new(dense_elements.as_slice(), rows, cols);

        // But their multiplications still match
        let dense_t_u = dense_matrix.right_multiply_vec(&u);
        assert_eq!(
            t_u, dense_t_u,
            "Sparse and dense give different multiplication result"
        );
    }
}
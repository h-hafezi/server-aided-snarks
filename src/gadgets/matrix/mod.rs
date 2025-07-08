use ark_ff::PrimeField;

pub mod sparse;

pub mod dense;
pub mod t_sparse;

pub trait Matrix<F: PrimeField>: Send + Sync {
    type Input<'a>: ?Sized where F: 'a;

    fn new(matrix: Self::Input<'_>, rows: usize, cols: usize) -> Self;

    fn get_row(&self, row: usize) -> Vec<F>;

    fn get_column(&self, col: usize) -> Vec<F>;

    fn print(&self);

    fn right_multiply_vec(&self, vector: &[F]) -> Vec<F>;

    fn left_multiply_vec(&self, vector: &[F]) -> Vec<F>;

    fn cols(&self) -> usize;

    fn rows(&self) -> usize;

    fn transpose(&self) -> Self;
}

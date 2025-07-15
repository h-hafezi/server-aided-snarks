use ark_ff::PrimeField;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SparseVector<F: PrimeField> {
    pub size: usize,
    pub entries: Vec<(usize, F)>,
}

impl<F: PrimeField> SparseVector<F> {
    /// Create a new sparse vector
    pub fn new(size: usize, entries: Vec<(usize, F)>) -> Self {
        Self { size, entries }
    }

    /// Converts the sparse vector into a dense vector of size `size`
    pub fn into_dense(&self) -> Vec<F> {
        let mut dense = vec![F::ZERO; self.size];
        for (i, v) in &self.entries {
            if i >= &self.size {
                panic!("Index {} is out of bounds for vector of size {}", i, self.size);
            }
            dense[*i] = *v;
        }
        dense
    }

    // throws error if n is not within [2^{10}, 2^{20}]
    pub fn error_vec<R: Rng + ?Sized>(size: usize, t: usize, rng: &mut R) -> Self {
        let chunk_size = size / t;
        let mut entries = Vec::with_capacity(t);

        for i in 0..t {
            let offset = rng.gen_range(0..chunk_size);
            let index = i * chunk_size + offset;

            // No need to enforce non-zero; probability of zero is negligible
            let val = F::rand(rng);
            entries.push((index, val));
        }

        Self {
            size,
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{AdditiveGroup, Zero};
    use ark_std::test_rng;

    #[test]
    fn test_into_dense() {
        let entries = vec![
            (0, Fr::from(3u64)),
            (2, Fr::from(7u64)),
            (4, Fr::from(1u64)),
        ];

        let sparse = SparseVector::new(6, entries.clone());
        let dense = sparse.into_dense();

        // Expected vector: [3, 0, 7, 0, 1, 0]
        assert_eq!(dense.len(), 6);
        for i in 0..6 {
            let expected = entries
                .iter()
                .find(|(idx, _)| *idx == i)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| Fr::ZERO);
            assert_eq!(dense[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    #[should_panic(expected = "Index 10 is out of bounds")]
    fn test_index_out_of_bounds_panics() {
        let entries = vec![(10, Fr::from(5u64))]; // Invalid index
        let sparse = SparseVector::new(5, entries);
        let _ = sparse.into_dense(); // Should panic
    }

    #[test]
    fn test_valid_error_vec_sizes() {
        let mut rng = test_rng();
        let t = 400;

        let sizes = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];
        for &size in &sizes {
            let sparse = SparseVector::<Fr>::error_vec(size,t, &mut rng);

            // Check correct size and number of non-zeros
            assert_eq!(sparse.size, size);
            assert_eq!(sparse.entries.len(), t, "Wrong number of non-zero entries");

            // Check for unique indices
            let indices: HashSet<_> = sparse.entries.iter().map(|(i, _)| *i).collect();
            assert_eq!(indices.len(), t, "Duplicate indices found");

            // Check for non-zero field values
            assert!(
                sparse.entries.iter().all(|(_, v)| !v.is_zero()),
                "Zero value found in sparse vector"
            );
        }
    }

    #[test]
    fn test_randomness_and_distribution() {
        let mut rng = test_rng();
        let (size, t) = (1 << 12, 400usize);

        let vec1 = SparseVector::<Fr>::error_vec(size, t, &mut rng);
        let vec2 = SparseVector::<Fr>::error_vec(size, t, &mut rng);

        // It’s unlikely for two independently sampled sparse vectors to be identical
        assert_ne!(vec1.entries, vec2.entries, "Sparse vectors should differ due to randomness");
    }
}

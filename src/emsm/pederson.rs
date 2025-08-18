use ark_ec::{CurveGroup, ScalarMul, VariableBaseMSM};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use crate::emsm::sparse_vec::SparseVector;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Pedersen<G: CurveGroup> {
    pub generators: Vec<G::Affine>,
}

impl<G: CurveGroup> Pedersen<G>
where
    G::Affine: CanonicalSerialize + CanonicalDeserialize,
{
    /// Create a new Pedersen commitment instance with `n` generators.
    /// The generators are deterministically derived from the label.
    pub fn new(n: usize) -> Self {
        // Use a fixed seed for determinism
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]); // Or derive from a label if needed

        // Sample deterministic group elements
        let gens: Vec<G> = (0..n)
            .map(|_| G::rand(&mut rng))
            .collect();

        let generators: Vec<G::Affine> = G::normalize_batch(&gens);

        Pedersen {
            generators,
        }
    }

    /// Commit to a vector of scalars.
    /// The length of `scalars` must be less than or equal to the number of generators.
    pub fn commit(&self, scalars: &[G::ScalarField]) -> G {
        debug_assert!(
            scalars.len() <= self.generators.len(),
            "Too many scalars for the number of generators"
        );

        // Only use as many generators as needed
        G::msm_unchecked(&self.generators[..scalars.len()], scalars)
    }

    /// Sparse commitment: computes commitment using only non-zero entries of the sparse vector.
    pub fn commit_sparse(&self, vector: &SparseVector<G::ScalarField>) -> G {
        for (i, _) in vector.entries.iter() {
            debug_assert!(
                *i < self.generators.len(),
                "Index {} out of bounds for number of generators {}",
                i,
                self.generators.len()
            );
        }

        debug_assert_eq!(vector.size, self.generators.len());

        let bases: Vec<G::Affine> = vector.entries.iter().map(|(i, _)| self.generators[*i]).collect();
        let scalars: Vec<G::ScalarField> = vector.entries.iter().map(|(_, s)| *s).collect();

        VariableBaseMSM::msm_unchecked(&bases, &scalars)
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Fr, G1Projective};
    use ark_ff::Zero;
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn test_commitment_consistency() {
        let pedersen = Pedersen::<G1Projective>::new(4);

        let mut rng = test_rng();
        let scalars: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();

        let commitment1 = pedersen.commit(&scalars);
        let commitment2 = pedersen.commit(&scalars);

        assert_eq!(commitment1, commitment2, "Commitments must be deterministic and equal");
    }

    #[test]
    fn test_partial_commitment() {
        let pedersen = Pedersen::<G1Projective>::new(5);

        let mut rng = test_rng();
        let scalars: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();

        let commitment = pedersen.commit(&scalars);

        // Check that the result is a valid group element (non-identity in general)
        assert!(!commitment.is_zero(), "Commitment should not be zero for random scalars");
    }

    #[test]
    #[should_panic(expected = "Too many scalars for the number of generators")]
    fn test_commitment_too_many_scalars() {
        let pedersen = Pedersen::<G1Projective>::new(3);

        let mut rng = test_rng();
        let scalars: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();

        // This should panic because there are more scalars than generators
        let _ = pedersen.commit(&scalars);
    }

    #[test]
    fn test_deterministic_generators() {
        let pedersen1 = Pedersen::<G1Projective>::new(4);
        let pedersen2 = Pedersen::<G1Projective>::new(4);

        assert_eq!(
            pedersen1.generators, pedersen2.generators,
            "Generators with the same label should be equal"
        );
    }

    #[test]
    fn test_commit_sparse_matches_dense() {
        let pedersen = Pedersen::<G1Projective>::new(10);

        let mut rng = test_rng();

        // Build a sparse vector: (index, value)
        let terms = vec![
            (2, Fr::rand(&mut rng)),
            (5, Fr::rand(&mut rng)),
            (7, Fr::rand(&mut rng)),
        ];

        let sparse = SparseVector::new(10, terms.clone());

        // Build the dense scalar vector
        let dense= sparse.into_dense();

        // Commit both ways
        let dense_commitment = pedersen.commit(&dense);
        let sparse_commitment = pedersen.commit_sparse(&sparse);

        assert_eq!(
            dense_commitment, sparse_commitment,
            "Sparse and dense commitments should be equal"
        );
    }
}

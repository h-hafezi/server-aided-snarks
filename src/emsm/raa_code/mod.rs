use std::marker::PhantomData;
use ark_ff::{PrimeField, Zero};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::ops::AddAssign;
use ark_std::iterable::Iterable;
use rayon::prelude::ParallelSliceMut;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator};
use rayon::iter::ParallelIterator;

pub struct TOperator<F: PrimeField + Copy + AddAssign + Zero> {
    p: Vec<usize>,
    q: Vec<usize>,
    pub n: usize,
    pub N: usize,
    phantom_data: PhantomData<F>,
}

impl<F> TOperator<F>
where
    F: PrimeField + Copy + AddAssign + Zero,
{
    pub fn new_random(n: usize) -> Self {
        let N = 4usize.checked_mul(n).expect("overflow computing N = 4*n");
        let mut rng = thread_rng();

        // generate first permutation
        let mut p: Vec<usize> = (0..N).collect();
        p.shuffle(&mut rng);

        // generate second permutation
        let mut q: Vec<usize> = (0..N).collect();
        q.shuffle(&mut rng);

        Self { p, q, n, N, phantom_data: Default::default() }
    }

    pub fn multiply_sparse(&self, mut e: &mut [F], parallel: bool) -> Vec<F> {
        assert_eq!(e.len(), self.N, "input sparse vector must have size N");

        // A * e
        accumulate_inplace(e.as_mut());

        // Q * A * e
        let mut after_q = permute_safe(e.as_mut(), &self.q, parallel);

        // A * Q * A * e
        accumulate_inplace(&mut after_q);

        // P * A * Q * A * e
        let after_p = permute_safe(after_q.as_mut_slice(), &self.p, parallel);

        // F * P * A * Q * A * e
        self.apply_F_fold(&after_p, parallel)
    }

    fn apply_F_fold(&self, v: &Vec<F>, parallel: bool) -> Vec<F> {
        assert_eq!(v.len(), self.N);
        let mut out = vec![F::zero(); self.n];

        if parallel {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, out_i)| {
                    let base = 4 * i;
                    let mut s = F::zero();
                    s += v[base];
                    s += v[base + 1];
                    s += v[base + 2];
                    s += v[base + 3];
                    *out_i = s;
                });
        } else {
            for i in 0..self.n {
                let base = 4 * i;
                let mut s = F::zero();
                s += v[base];
                s += v[base + 1];
                s += v[base + 2];
                s += v[base + 3];
                out[i] = s;
            }
        }

        out
    }
}


fn permute_safe<T: Default + Copy + Send + Sync>(v: &mut [T], perm: &[usize], parallel: bool, ) -> Vec<T> {
    // make sure the permutation and vector have equal length
    debug_assert_eq!(v.len(), perm.len());

    // make a vector with default value
    let mut res = vec![T::default(); v.len()];

    if parallel {
        // Parallel version using rayon
        res.par_iter_mut()
            .enumerate()
            .for_each(|(i, r)| {
                *r = v[perm[i]];
            });
    } else {
        // Sequential version
        for (i, &pi) in perm.iter().enumerate() {
            res[i] = v[pi];
        }
    }

    res
}

fn accumulate_inplace<T: Zero + Clone + AddAssign + PartialEq + Send + Sync>(v: &mut [T]) {
    let n = v.len();
    if n == 0 { return; }

    if n < 1_000 {
        let mut acc = T::zero();
        for x in v.iter_mut() {
            acc += x.clone();
            *x = acc.clone();
        }
        return;
    }

    let num_chunks = (rayon::current_num_threads() * 4).min(n);
    let chunk_size = (n + num_chunks - 1) / num_chunks;

    let mut totals: Vec<T> = v
        .par_chunks_mut(chunk_size)
        .map(|chunk| {
            let mut acc = T::zero();
            for x in chunk.iter_mut() {
                acc += x.clone();
                *x = acc.clone();
            }
            acc
        })
        .collect();

    let mut offset = T::zero();
    for t in totals.iter_mut() {
        let tmp = t.clone();
        *t = offset.clone();
        offset += tmp;
    }

    let offsets = totals;
    v.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let off = offsets[i].clone();
            if off != T::zero() {
                for x in chunk.iter_mut() {
                    *x += off.clone();
                }
            }
        });
}

/// Computes the inverse of a permutation `perm`.
/// `perm` is a slice representing a permutation of `[0..perm.len()]`.
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0; n];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}



#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::One;
    use crate::emsm::sparse_vec::sparse_vec::SparseVector;

    #[test]
    fn test_inverse_permutation() {
        let perm = vec![2, 0, 3, 1];
        let inv = inverse_permutation(&perm);
        let inv_inv = inverse_permutation(&inv);
        assert_eq!(inv_inv, perm);

        // Also test that perm followed by inv is identity:
        let n = perm.len();
        for i in 0..n {
            assert_eq!(perm[inv[i]], i);
            assert_eq!(inv[perm[i]], i);
        }
    }

    #[test]
    fn test_accumulate_sorted_sparse_to_dense() {
        let n = 2; // N = 8 for this example
        let op = TOperator::<Fr>::new_random(n);
        // vector: (1, 0, 0, 1, 0, 2, 0, 0)
        let sparse = SparseVector {
            size: 8,
            entries: vec![
                (0, Fr::one()),
                (3, Fr::one()),
                (5, Fr::from(2u64)),
            ],
        };

        let mut dense = sparse.into_dense();

        let expected_full = vec![
            Fr::from(1u64),
            Fr::from(1u64),
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(2u64),
            Fr::from(4u64),
            Fr::from(4u64),
            Fr::from(4u64),
        ];

        accumulate_inplace(dense.as_mut_slice());

        assert_eq!(dense, expected_full);
    }
}

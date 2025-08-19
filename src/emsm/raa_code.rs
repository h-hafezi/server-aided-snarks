use std::marker::PhantomData;
use ark_ff::{PrimeField, Zero};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::ops::{AddAssign, SubAssign};
use rayon::prelude::ParallelSliceMut;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::iter::ParallelIterator;

#[derive(Debug, Clone)]
pub struct TOperator<F: PrimeField + Copy + AddAssign + Zero> {
    /// permutation p : [N] -> [N]
    pub p: Vec<usize>,
    /// permutation q : [N] -> [N]
    pub q: Vec<usize>,
    pub n: usize,
    pub N: usize,
    phantom_data: PhantomData<F>,
}

impl<F> TOperator<F>
where
    F: PrimeField + Copy + AddAssign + Zero,
{
    pub fn rand(n: usize) -> Self {
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

    pub fn multiply_sparse(&self, mut e: Vec<F>) -> Vec<F> {
        assert_eq!(e.len(), self.N, "input sparse vector must have size N");

        // A * e
        accumulate_inplace(e.as_mut_slice(), F::zero());

        // Q * A * e
        let mut after_q = permute_safe(e.as_mut(), &self.q);
        drop(e);

        // A * Q * A * e
        accumulate_inplace(&mut after_q, F::zero());

        // P * A * Q * A * e
        let after_p = permute_safe(after_q.as_mut_slice(), &self.p);
        drop(after_q);

        // F * P * A * Q * A * e
        apply_F_fold(&after_p)
    }
}


pub fn permute_safe<T: Default + Copy + Send + Sync>(v: &mut [T], perm: &[usize]) -> Vec<T> {
    // Make sure the permutation and vector have equal length
    debug_assert_eq!(v.len(), perm.len());

    // Create a result vector with default values
    let mut res = vec![T::default(); v.len()];

    // Threshold for switching to parallel
    const PARALLEL_THRESHOLD: usize = 1 << 16;

    if v.len() > PARALLEL_THRESHOLD {
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


/// (g_0, ..., g_n) ==> (\sum g_i, \sum g_i - g_1, ..., g_n)

pub fn accumulate_inplace<T>(v: &mut [T], zero: T)
where
    T: Clone + AddAssign + SubAssign + PartialEq + Send + Sync,
{
    let n = v.len();
    if n == 0 {
        return;
    }

    const PARALLEL_THRESHOLD: usize = 1 << 17;

    if n <= PARALLEL_THRESHOLD {
        // Serial case
        let mut acc = zero.clone();
        for x in v.iter_mut().rev() {
            acc += x.clone();
            *x = acc.clone();
        }
        return;
    }

    // Parallel case
    let num_chunks = (rayon::current_num_threads() * 4).min(n);
    let chunk_size = (n + num_chunks - 1) / num_chunks;

    // Step 1: Compute partial suffix sums in each chunk (right-to-left)
    let mut totals: Vec<T> = v
        .par_chunks_mut(chunk_size)
        .rev()
        .map(|chunk| {
            let mut acc = zero.clone();
            for x in chunk.iter_mut().rev() {
                acc += x.clone();
                *x = acc.clone();
            }
            acc
        })
        .collect();

    // Step 2: Compute cumulative sums of totals (so each chunk knows how much to add)
    let mut offset = zero.clone();
    for t in totals.iter_mut().rev() {
        let tmp = t.clone();
        *t = offset.clone();
        offset += tmp;
    }

    // Step 3: Add offsets to chunks
    v.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let off = totals[num_chunks - 1 - i].clone();
            if off != zero.clone() {
                for x in chunk.iter_mut() {
                    *x += off.clone();
                }
            }
        });
}

fn apply_F_fold<F: PrimeField>(v: &Vec<F>) -> Vec<F> {
    const PARALLEL_THRESHOLD: usize = 1 << 16; // 65,536 elements

    assert_eq!(v.len() % 4, 0);
    let mut out = vec![F::zero(); v.len() / 4];

    if v.len() > PARALLEL_THRESHOLD {
        // Parallel version
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
        // Sequential version
        for i in 0..v.len() / 4 {
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


/// Computes the inverse of a permutation `perm`.
/// `perm` is a slice representing a permutation of `[0..perm.len()]`.
pub fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
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

    #[test]
    fn test_suffix_accumulate_small() {
        let mut v = vec![1, 2, 3, 4];
        accumulate_inplace(&mut v, 0);
        assert_eq!(v, vec![10, 9, 7, 4]);
    }

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
}

// This bench helps to know what is the threshold for parallelism, e.g. if |vec| > 2^n use parallelism

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use rand::prelude::*;
use rayon::prelude::*;

pub fn permute_safe<T: Default + Copy + Send + Sync>(
    v: &mut [T],
    perm: &[usize],
    parallel: bool,
) -> Vec<T> {
    debug_assert_eq!(v.len(), perm.len());

    let mut res = vec![T::default(); v.len()];

    if parallel {
        res.par_iter_mut()
            .enumerate()
            .for_each(|(i, r)| {
                *r = v[perm[i]];
            });
    } else {
        for (i, &pi) in perm.iter().enumerate() {
            res[i] = v[pi];
        }
    }

    res
}

fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("permute_safe");

    for exp in 10..=25 {
        let len = 1 << exp;

        // Generate random vector and permutation
        let mut rng = rand::thread_rng();
        let mut v: Vec<u64> = (0..len as u64).collect();
        v.shuffle(&mut rng);

        let mut perm: Vec<usize> = (0..len).collect();
        perm.shuffle(&mut rng);

        // Sequential benchmark
        group.bench_with_input(BenchmarkId::new("sequential", len), &len, |b, &_len| {
            let mut v_clone = v.clone();
            let perm_clone = perm.clone();
            b.iter(|| {
                black_box(permute_safe(&mut v_clone, &perm_clone, false));
            });
        });

        // Parallel benchmark
        group.bench_with_input(BenchmarkId::new("parallel", len), &len, |b, &_len| {
            let mut v_clone = v.clone();
            let perm_clone = perm.clone();
            b.iter(|| {
                black_box(permute_safe(&mut v_clone, &perm_clone, true));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_permute);
criterion_main!(benches);

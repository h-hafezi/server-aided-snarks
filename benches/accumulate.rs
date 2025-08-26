// This bench helps to know what is the threshold for parallelism, e.g. if |vec| > 2^n use parallelism

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use std::ops::{AddAssign, SubAssign};
use rayon::prelude::*;

pub fn accumulate_inplace<T>(v: &mut [T], zero: T, parallel: bool)
where
    T: Clone + AddAssign + SubAssign + PartialEq + Send + Sync,
{
    let n = v.len();
    if n == 0 {
        return;
    }

    if !parallel || n < 1_000 {
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

fn bench_accumulate(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulate_inplace");

    for exp in 10..=25 {
        let size = 1usize << exp;

        // Generate random vector
        let mut rng = rand::thread_rng();
        let data: Vec<u64> = (0..size).map(|_| rng.gen()).collect();

        // Sequential benchmark
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut vec_copy = data.clone();
                    accumulate_inplace(&mut vec_copy, 0u64, false);
                });
            },
        );

        // Parallel benchmark
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut vec_copy = data.clone();
                    accumulate_inplace(&mut vec_copy, 0u64, true);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_accumulate);
criterion_main!(benches);

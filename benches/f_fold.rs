// This bench helps to know what is the threshold for parallelism, e.g. if |vec| > 2^n use parallelism

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use rand::thread_rng;
use rayon::prelude::*;
use ark_ff::{PrimeField, Field, UniformRand, Zero};

// Your function, slightly modified to accept preallocated `out`
fn apply_F_fold<F: PrimeField>(v: &Vec<F>, parallel: bool, out: &mut [F]) {
    assert_eq!(v.len() % 4, 0);
    assert_eq!(out.len(), v.len() / 4);

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
}

fn bench_apply_F_fold(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_F_fold");

    for exp in 10..=25 {
        let N = 1 << exp;

        // Generate random input vector
        let mut rng = thread_rng();
        let v: Vec<ark_bn254::Fr> = (0..N).map(|_| ark_bn254::Fr::rand(&mut rng)).collect();

        // Preallocate output buffer
        let mut out = vec![ark_bn254::Fr::zero(); N / 4];

        // Sequential benchmark
        group.bench_with_input(BenchmarkId::new("sequential", N), &N, |b, &_N| {
            b.iter(|| {
                apply_F_fold(&v, false, &mut out);
                black_box(&out);
            });
        });

        // Parallel benchmark
        group.bench_with_input(BenchmarkId::new("parallel", N), &N, |b, &_N| {
            b.iter(|| {
                apply_F_fold(&v, true, &mut out);
                black_box(&out);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_apply_F_fold);
criterion_main!(benches);

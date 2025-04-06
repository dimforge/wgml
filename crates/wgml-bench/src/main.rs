#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use indexmap::IndexMap;

use nalgebra::{DMatrix, DVector};
use std::time::Duration;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::CommandEncoderExt;
use wgcore::shapes::ViewShapeBuffers;
use wgcore::tensor::{GpuCubeView, GpuMatrix, GpuVector, TensorBuilder};
use wgcore::timestamps::GpuTimestamps;
use wgcore::{Pod, Shader};
use wgebra::{Gemm, GemmVariant, GemvVariant};
use wgml::ops::{
    GemvQuant, GpuBlockQ4K, GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ5K, GpuBlockQ5_0x2,
    GpuBlockQ5_1x2, GpuBlockQ6Kx2, GpuBlockQ8K, GpuBlockQ8_0x2, QuantizedValue,
};
use wgml::quantized_matrix::GpuQuantMatrix;
use wgpu::{BufferUsages, ComputePass, Maintain};

mod cli;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum ScalarType {
    F32,
    Q8_0,
    Q5_0,
    Q5_1,
    Q4_0,
    Q4_1,
    Q8K,
    Q6K,
    Q5K,
    Q4K,
}

type MatrixSize = (u32, u32);

#[derive(Debug, Copy, Clone)]
struct GemvBenchEntry {
    matrix: MatrixSize,
    scalar: ScalarType,
    variant: GemvVariant,
    timing: f32,
}

#[derive(Debug, Copy, Clone)]
struct GemmBenchEntry {
    matrix: MatrixSize,
    variant: GemmVariant,
    timing: f32,
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    // TODO:
    // - [x] Allow varying matrix size (round to next power of 2).
    // - [x] Support benchmarking with quantized matrix.
    // - [ ] Support setting the workgroup size.
    // - [x] Support benchmarking gemm.
    // - [ ] Bonus: allow rectangular matrices.
    // - [x] Bonus: generate graph with timings for various shapes.
    // let cli = cli::Cli::parse();
    let gpu = GpuInstance::new().await?;
    let shapes = ViewShapeBuffers::new();
    let mut gemv_bench = Vec::new();
    let mut gemm_bench = Vec::new();

    let gemv = GemvQuant::from_device(gpu.device())?;
    let gemm = Gemm::from_device(gpu.device())?;
    let dim = 512;

    const BENCH_GEMM: bool = false;

    for k in 0..20 {
        let dim = dim * k;
        println!("# Benchmark for {}x{}", dim, dim);

        if dim <= 8192 {
            for variant in [
                GemvVariant::GemvTrFast,
                GemvVariant::GemvTr,
                GemvVariant::GemvFast,
                GemvVariant::Gemv,
            ]
            .iter()
            {
                let runtime = bench_gemv::<f32>(&gpu, dim, dim, |pass, result, m, v| {
                    if let GpuQuantMatrix::F32(m) = m {
                        gemv.gemv_f32.dispatch_generic(
                            gpu.device(),
                            &shapes,
                            pass,
                            result,
                            m,
                            v,
                            *variant,
                        );
                    } else {
                        panic!("invalid matrix type for f32 gemv");
                    }
                })
                .await?;
                gemv_bench.push(GemvBenchEntry {
                    matrix: (dim, dim),
                    scalar: ScalarType::F32,
                    variant: *variant,
                    timing: runtime,
                });
                println!("\t{:?} avg. runtime: {:?}", variant, runtime);
            }

            if BENCH_GEMM && dim <= 2048 {
                for variant in [
                    GemmVariant::GemmTrFast,
                    GemmVariant::GemmTr,
                    GemmVariant::GemmFast,
                    GemmVariant::Gemm,
                ]
                .iter()
                {
                    let runtime = bench_gemm(&gpu, dim, dim, |pass, result, m1, m2| {
                        gemm.dispatch_generic(
                            gpu.device(),
                            &shapes,
                            pass,
                            result,
                            m1,
                            m2,
                            *variant,
                        );
                    })
                    .await?;
                    gemm_bench.push(GemmBenchEntry {
                        matrix: (dim, dim),
                        variant: *variant,
                        timing: runtime,
                    });
                    println!("\tGEMM {:?} avg. runtime: {:?}", variant, runtime);
                }
            }
        }

        macro_rules! bench_quantized(
            ($($scalar: ident, $quant: ty);*) => {$(
                let runtime = bench_gemv::<$quant>(&gpu, dim, dim, |pass, result, m, v| {
                    gemv.dispatch(gpu.device(), &shapes, pass, result, m, v);
                }).await?;
                gemv_bench.push(
                    GemvBenchEntry {
                        matrix: (dim, dim),
                        scalar: ScalarType::$scalar,
                        variant: GemvVariant::Gemv,
                        timing: runtime,
                    },
                );
                println!("\t{:?} runtime: {:?}", ScalarType::$scalar, runtime);
            )*}
        );

        bench_quantized!(
            Q8_0, GpuBlockQ8_0x2;
            Q5_0, GpuBlockQ5_0x2;
            Q5_1, GpuBlockQ5_1x2;
            Q4_0, GpuBlockQ4_0x2;
            Q4_1, GpuBlockQ4_1x2
        );

        if dim >= 2048 {
            bench_quantized!(
                Q8K, GpuBlockQ8K;
                Q6K, GpuBlockQ6Kx2;
                Q5K, GpuBlockQ5K;
                Q4K, GpuBlockQ4K
            );
        }
    }

    println!("All results: {:?}", gemv_bench);

    plot_timings(&gemv_bench, &gemm_bench);

    Ok(())
}

async fn bench_gemv<'a, 'b: 'a, T>(
    gpu: &'b GpuInstance,
    nrows: u32,
    ncols: u32,
    gemv: impl Fn(&mut ComputePass<'_>, &GpuVector<f32>, &GpuQuantMatrix, &GpuVector<f32>) + 'a,
) -> anyhow::Result<f32>
where
    T: QuantizedValue + nalgebra::Scalar + Pod,
    GpuQuantMatrix: From<GpuMatrix<T>>,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    let nrows: u32 = nrows / T::DEQUANTIZED_LEN as u32;
    let ncols: u32 = ncols / T::DEQUANTIZED_LEN as u32;

    let mut timestamps = GpuTimestamps::new(gpu.device(), 2);

    let m_cpu = DMatrix::<T>::new_random(nrows as usize, ncols as usize);
    let v_cpu = DVector::<f32>::new_random(nrows as usize);
    let lhs_cpu = DVector::<f32>::new_random(ncols as usize);

    let m = GpuQuantMatrix::from(
        TensorBuilder::matrix(nrows, ncols, BufferUsages::STORAGE)
            .build_init(gpu.device(), m_cpu.as_slice()),
    );
    let v = TensorBuilder::vector(v_cpu.nrows() as u32, BufferUsages::STORAGE)
        .build_init(gpu.device(), v_cpu.as_slice());
    let result = TensorBuilder::vector(ncols, BufferUsages::STORAGE)
        .build_init(gpu.device(), lhs_cpu.as_slice());

    let mut accum_time = Duration::ZERO;
    let mut accum_divisor = 0.0;
    let mut max_time = 0.0f64;
    let mut min_time = 100000.0f64;

    for step in 0..20 {
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const REPEATS: u32 = 10;
        timestamps.clear();
        let mut pass = encoder.compute_pass("compute", Some(&mut timestamps));
        for _ in 0..REPEATS {
            gemv(&mut pass, &result, &m, &v);
        }
        drop(pass);

        timestamps.resolve(&mut encoder);
        gpu.queue().submit(Some(encoder.finish()));
        gpu.device().poll(Maintain::Wait);

        // For measuring runtime, skip the first few runs.
        // In particular, the very first one tends to be orders
        // of magnitudes slower than the rest.
        if step > 10 {
            let timestamps = timestamps.wait_for_results_ms(gpu.device(), gpu.queue());
            accum_time += Duration::from_secs_f64((timestamps[1] - timestamps[0]) / 1000.0); // t0.elapsed();
            accum_divisor += 1.0;
            max_time = max_time.max(timestamps[1] - timestamps[0]);
            min_time = min_time.min(timestamps[1] - timestamps[0]);
        }
    }

    let runtime = accum_time.as_secs_f32() * 1000.0 / accum_divisor;
    Ok(runtime)
}

async fn bench_gemm<'a, 'b: 'a>(
    gpu: &'b GpuInstance,
    nrows: u32,
    ncols: u32,
    gemm: impl Fn(&mut ComputePass<'_>, GpuCubeView<'_, f32>, GpuCubeView<'_, f32>, GpuCubeView<'_, f32>)
        + 'a,
) -> anyhow::Result<f32> {
    let mut timestamps = GpuTimestamps::new(gpu.device(), 2);
    let m1_cpu = DMatrix::<f32>::new_random(nrows as usize, ncols as usize);
    let m2_cpu = DMatrix::<f32>::new_random(ncols as usize, nrows as usize);
    let lhs_cpu = DMatrix::<f32>::new_random(nrows as usize, nrows as usize);

    let m1 = TensorBuilder::matrix(nrows, ncols, BufferUsages::STORAGE)
        .build_init(gpu.device(), m1_cpu.as_slice());
    let m2 = TensorBuilder::matrix(ncols, nrows, BufferUsages::STORAGE)
        .build_init(gpu.device(), m2_cpu.as_slice());
    let result = TensorBuilder::matrix(nrows, nrows, BufferUsages::STORAGE)
        .build_init(gpu.device(), lhs_cpu.as_slice());

    let mut accum_time = Duration::ZERO;
    let mut accum_divisor = 0.0;

    for step in 0..20 {
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const REPEATS: u32 = 10;
        timestamps.clear();

        let mut pass = encoder.compute_pass("compute", Some(&mut timestamps));
        for _ in 0..REPEATS {
            gemm(
                &mut pass,
                result.as_embedded_view(),
                m1.as_embedded_view(),
                m2.as_embedded_view(),
            );
        }
        drop(pass);

        timestamps.resolve(&mut encoder);
        gpu.queue().submit(Some(encoder.finish()));
        gpu.device().poll(Maintain::Wait);

        // For measuring runtime, skip the first few runs.
        // In particular, the very first one tends to be orders
        // of magnitudes slower than the rest.
        if step > 2 {
            let timestamps = timestamps.wait_for_results_ms(gpu.device(), gpu.queue());
            accum_time += Duration::from_secs_f64(timestamps[1] - timestamps[0]); // t0.elapsed();
            accum_divisor += 1.0;
        }
    }

    let runtime = accum_time.as_secs_f32() * 1000.0 / accum_divisor;
    Ok(runtime)
}

fn plot_timings(gemv: &[GemvBenchEntry], gemm: &[GemmBenchEntry]) {
    use plotly::{
        common::{Line, Mode},
        layout::Layout,
        Plot, Scatter,
    };

    let layout = Layout::new()
        .title("WGML matmul benches")
        .width(2000)
        .height(500);
    let mut plot = Plot::new();

    let mut gemv_entries: IndexMap<_, (Vec<_>, Vec<_>)> = IndexMap::new();
    for entry in gemv {
        let (x, y) = gemv_entries
            .entry((entry.variant, entry.scalar))
            .or_insert((Vec::new(), Vec::new()));
        x.push(entry.matrix.0);
        y.push(entry.timing);
    }

    let mut gemm_entries: IndexMap<_, (Vec<_>, Vec<_>)> = IndexMap::new();
    for entry in gemm {
        let (x, y) = gemm_entries
            .entry(entry.variant)
            .or_insert((Vec::new(), Vec::new()));
        x.push(entry.matrix.0);
        y.push(entry.timing);
    }

    println!("{:?}", gemm_entries);

    for (key, values) in gemv_entries {
        let trace = Scatter::new(values.0, values.1)
            .mode(Mode::LinesMarkersText)
            .name(format!("{:?}-{:?}", key.0, key.1))
            .line(Line::new().width(3.0));
        plot.add_trace(trace);
    }

    for (key, values) in gemm_entries {
        let trace = Scatter::new(values.0, values.1)
            .mode(Mode::LinesMarkersText)
            .name(format!("{:?}-f32", key))
            .line(Line::new().width(3.0));
        plot.add_trace(trace);
    }

    plot.set_layout(layout);
    plot.show();
}

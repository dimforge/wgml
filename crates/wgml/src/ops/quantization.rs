use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "quantization.wgsl")]
/// Shader implementing the (de)quantizaiton functions.
pub struct Quantization;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::gemv_quant::{
        GpuBlockQ4K, GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ5K, GpuBlockQ5_0x2, GpuBlockQ5_1x2,
        GpuBlockQ6Kx2, GpuBlockQ8K,
    };
    use crate::ops::GpuBlockQ8_0x2;
    use crate::quantization::{
        BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K, BlockQ8K,
        BlockQ8_0,
    };
    use bytemuck::Pod;
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
    use wgcore::tensor::GpuVector;
    use wgpu::{BufferUsages, ComputePipeline, Device};

    #[derive(Shader)]
    #[shader(
        derive(Quantization),
        src = "quantization_test.wgsl",
        composable = false
    )]
    struct QuantTest {
        dequantize: ComputePipeline,
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn dequantization() {
        let gpu = GpuInstance::new().await.unwrap();
        let quant_test = QuantTest::from_device(gpu.device()).unwrap();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        struct TestData<T, GpuT> {
            in_vec: DVector<T>,
            in_buf: GpuVector<GpuT>,
            out_buf: GpuVector<f32>,
            staging_buf: GpuVector<f32>,
        }

        impl<T: Pod, GpuT: bytemuck::Pod> TestData<T, GpuT> {
            fn new(device: &Device, in_vec: DVector<T>, elts_per_block: usize) -> Self {
                let out_vec = DVector::<f32>::repeat(in_vec.len() * elts_per_block, 0.0);
                let in_buf = GpuVector::init(
                    device,
                    bytemuck::cast_slice(in_vec.as_slice()),
                    BufferUsages::STORAGE,
                );
                let out_buf = GpuVector::init(
                    device,
                    &out_vec,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                );
                let staging_buf = GpuVector::init(
                    device,
                    &out_vec,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                );
                Self {
                    in_vec,
                    in_buf,
                    out_buf,
                    staging_buf,
                }
            }
        }

        // Q8_0
        let in_q8_0 = DVector::from_fn(8, |i, _| {
            let scale = half::f16::from_f32(1.234).to_bits();
            let mut data = [i as i8 * 16 + i8::MIN; 32];
            data.iter_mut().enumerate().for_each(|(k, d)| *d += k as i8);
            BlockQ8_0 { scale, data }
        });
        let data_q8_0 = TestData::<_, GpuBlockQ8_0x2>::new(gpu.device(), in_q8_0, 32);

        // Q4_0
        let in_q4_0 = DVector::from_fn(16, |i, _| {
            let scale = half::f16::from_f32(1.234).to_bits();
            let mut data = [i as u8 * 16; 16];
            data.iter_mut().enumerate().for_each(|(k, d)| *d += k as u8);
            BlockQ4_0 { d: scale, qs: data }
        });
        let data_q4_0 = TestData::<_, GpuBlockQ4_0x2>::new(gpu.device(), in_q4_0, 32);

        // Q4_1
        let in_q4_1 = DVector::from_fn(16, |i, _| {
            let scale = half::f16::from_f32(1.234).to_bits();
            let mid = 3;
            let mut data = [i as u8 * 16; 16];
            data.iter_mut().enumerate().for_each(|(k, d)| *d += k as u8);
            BlockQ4_1 {
                d: scale,
                m: mid,
                qs: data,
            }
        });
        let data_q4_1 = TestData::<_, GpuBlockQ4_1x2>::new(gpu.device(), in_q4_1, 32);

        // Q5_0
        let in_q5_0 = DVector::from_fn(16, |i, _| {
            let scale = half::f16::from_f32(1.234).to_bits();
            let mut data = [i as u8 * 16; 16];
            data.iter_mut().enumerate().for_each(|(k, d)| *d += k as u8);
            BlockQ5_0 {
                d: scale,
                qh: [1, 100, 200, 250],
                qs: data,
            }
        });
        let data_q5_0 = TestData::<_, GpuBlockQ5_0x2>::new(gpu.device(), in_q5_0, 32);

        // Q5_1
        let in_q5_1 = DVector::from_fn(16, |i, _| {
            let scale = half::f16::from_f32(1.234).to_bits();
            let mut data = [i as u8 * 16; 16];
            data.iter_mut().enumerate().for_each(|(k, d)| *d += k as u8);
            BlockQ5_1 {
                d: scale,
                m: 125,
                qh: [1, 100, 200, 250],
                qs: data,
            }
        });
        let data_q5_1 = TestData::<_, GpuBlockQ5_1x2>::new(gpu.device(), in_q5_1, 32);

        // Q8_k
        let in_q8_k = DVector::from_fn(16, |i, _| {
            let mut data = [i as i8 * 8 - 127; 256];
            data.iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d = d.saturating_add(k as i8));
            BlockQ8K {
                d: 1.234,
                qs: data,
                bsums: [0; 16],
            }
        });
        let data_q8_k = TestData::<_, GpuBlockQ8K>::new(gpu.device(), in_q8_k, 256);

        // Q5_k
        let in_q5_k = DVector::from_fn(16, |i, _| {
            let scale1 = half::f16::from_f32(1.234).to_bits();
            let scale2 = half::f16::from_f32(5.678).to_bits();
            let mut scales = [i as u8 * 16; 12];
            let mut qs = [i as u8 * 16; 256 / 2];
            let mut qh = [i as u8 * 16; 256 / 8];
            scales
                .iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d += k as u8);
            qs.iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d = d.saturating_add(k as u8));
            qh.iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d = d.saturating_add(k as u8));
            BlockQ5K {
                d: scale1,
                dmin: scale2,
                scales,
                qh,
                qs,
            }
        });
        let data_q5_k = TestData::<_, GpuBlockQ5K>::new(gpu.device(), in_q5_k, 256);

        // Q4_k
        let in_q4_k = DVector::from_fn(16, |i, _| {
            let scale1 = half::f16::from_f32(1.234).to_bits();
            let scale2 = half::f16::from_f32(5.678).to_bits();
            let mut scales = [i as u8 * 16; 12];
            let mut qs = [i as u8 * 16; 256 / 2];
            let mut qh = [i as u8 * 16; 256 / 8];
            scales
                .iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d += k as u8);
            qs.iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d = d.saturating_add(k as u8));
            qh.iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d = d.saturating_add(k as u8));
            BlockQ4K {
                d: scale1,
                dmin: scale2,
                scales,
                qs,
            }
        });
        let data_q4_k = TestData::<_, GpuBlockQ4K>::new(gpu.device(), in_q4_k, 256);

        // Q6_k
        let in_q6_k = DVector::from_fn(16, |i, _| {
            let mut scales = [i as i8 * 8 - 127; 256 / 16];
            let mut ql = [i as u8 * 8; 256 / 2];
            let mut qh = [i as u8 * 8; 256 / 4];
            scales
                .iter_mut()
                .enumerate()
                .for_each(|(k, d)| *d += k as i8);
            ql.iter_mut().enumerate().for_each(|(k, d)| *d += k as u8);
            qh.iter_mut().enumerate().for_each(|(k, d)| *d += k as u8);
            BlockQ6K {
                ql,
                qh,
                scales,
                d: 123,
            }
        });
        let data_q6_k = TestData::<_, GpuBlockQ6Kx2>::new(gpu.device(), in_q6_k, 256);

        // Kernel call.
        let max_len = data_q8_0
            .in_vec
            .len()
            .max(data_q4_0.in_vec.len())
            .max(data_q4_1.in_vec.len())
            .max(data_q8_k.in_vec.len())
            .max(data_q5_k.in_vec.len())
            .max(data_q4_k.in_vec.len());

        let mut pass = encoder.compute_pass("test", None);
        KernelDispatch::new(gpu.device(), &mut pass, &quant_test.dequantize)
            .bind0([
                data_q8_0.in_buf.buffer(),
                data_q8_0.out_buf.buffer(),
                data_q4_0.in_buf.buffer(),
                data_q4_0.out_buf.buffer(),
                data_q4_1.in_buf.buffer(),
                data_q4_1.out_buf.buffer(),
                data_q5_0.in_buf.buffer(),
                data_q5_0.out_buf.buffer(),
            ])
            .bind(
                1,
                [
                    data_q5_1.in_buf.buffer(),
                    data_q5_1.out_buf.buffer(),
                    data_q8_k.in_buf.buffer(),
                    data_q8_k.out_buf.buffer(),
                    data_q5_k.in_buf.buffer(),
                    data_q5_k.out_buf.buffer(),
                    data_q4_k.in_buf.buffer(),
                    data_q4_k.out_buf.buffer(),
                ],
            )
            .bind(2, [data_q6_k.in_buf.buffer(), data_q6_k.out_buf.buffer()])
            .dispatch(max_len as u32);
        drop(pass);

        data_q8_0
            .staging_buf
            .copy_from(&mut encoder, &data_q8_0.out_buf);
        data_q4_0
            .staging_buf
            .copy_from(&mut encoder, &data_q4_0.out_buf);
        data_q4_1
            .staging_buf
            .copy_from(&mut encoder, &data_q4_1.out_buf);
        data_q5_0
            .staging_buf
            .copy_from(&mut encoder, &data_q5_0.out_buf);
        data_q5_1
            .staging_buf
            .copy_from(&mut encoder, &data_q5_1.out_buf);
        data_q8_k
            .staging_buf
            .copy_from(&mut encoder, &data_q8_k.out_buf);
        data_q5_k
            .staging_buf
            .copy_from(&mut encoder, &data_q5_k.out_buf);
        data_q4_k
            .staging_buf
            .copy_from(&mut encoder, &data_q4_k.out_buf);
        data_q6_k
            .staging_buf
            .copy_from(&mut encoder, &data_q6_k.out_buf);
        gpu.queue().submit(Some(encoder.finish()));

        // Test result Q8_0
        let gpu_result_v8_0 = data_q8_0.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_v8_0: Vec<_> = data_q8_0
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_v8_0),
            DVector::from(cpu_result_v8_0),
            epsilon = 1.0e-3
        );

        // Test result Q4_0
        let gpu_result_q4_0 = data_q4_0.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_q4_0: Vec<_> = data_q4_0
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_q4_0),
            DVector::from(cpu_result_q4_0),
            epsilon = 1.0e-3
        );

        // Test result Q4_1
        let gpu_result_q4_1 = data_q4_1.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_q4_1: Vec<_> = data_q4_1
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_q4_1),
            DVector::from(cpu_result_q4_1),
            epsilon = 1.0e-3
        );

        // Test result Q5_0
        let gpu_result_q5_0 = data_q5_0.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_q5_0: Vec<_> = data_q5_0
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_q5_0),
            DVector::from(cpu_result_q5_0),
            epsilon = 1.0e-3
        );

        // Test result Q5_1
        let gpu_result_q5_1 = data_q5_1.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_q5_1: Vec<_> = data_q5_1
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_q5_1),
            DVector::from(cpu_result_q5_1),
            epsilon = 1.0e-3
        );

        // Test result Q8K
        let gpu_result_v8_k = data_q8_k.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_v8_k: Vec<_> = data_q8_k
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_v8_k),
            DVector::from(cpu_result_v8_k),
            epsilon = 1.0e-3
        );

        // Test result Q5K
        let gpu_result_v5_k = data_q5_k.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_v5_k: Vec<_> = data_q5_k
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_v5_k),
            DVector::from(cpu_result_v5_k),
            epsilon = 1.0e-3
        );

        // Test result Q4K
        let gpu_result_v4_k = data_q4_k.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_v4_k: Vec<_> = data_q4_k
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_v4_k),
            DVector::from(cpu_result_v4_k),
            epsilon = 1.0e-3
        );

        // Test result Q6K
        let gpu_result_v6_k = data_q6_k.staging_buf.read(gpu.device()).await.unwrap();
        let cpu_result_v6_k: Vec<_> = data_q6_k
            .in_vec
            .iter()
            .flat_map(|x| x.dequantize().into_iter())
            .collect();
        approx::assert_relative_eq!(
            DVector::from(gpu_result_v6_k),
            DVector::from(cpu_result_v6_k),
            epsilon = 1.0e-3
        );
    }
}

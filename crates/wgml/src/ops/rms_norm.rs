use bytemuck::Pod;
use nalgebra::{DVector, Dyn, Storage, Vector};
use wgcore::kernel::{KernelDispatch, KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuScalar, GpuVectorView};
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::{ComputePass, ComputePipeline};

#[derive(Shader)]
#[shader(derive(Shape), src = "rms_norm.wgsl", composable = false)]
/// Shader implementing the RMS norm kernel.
pub struct RmsNorm {
    pub main: ComputePipeline,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
pub struct RmsNormConfig {
    pub nudge_factor: f32,
}

impl RmsNorm {
    pub fn dispatch<'a, 'b, T: Pod>(
        &'a self,
        queue: &KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        config: &GpuScalar<RmsNormConfig>,
        result: impl Into<GpuVectorView<'b, T>>,
        value: impl Into<GpuVectorView<'b, T>>,
        weight: impl Into<GpuVectorView<'b, T>>,
    ) {
        let value = value.into();
        let weight = weight.into();
        let result = result.into();

        let value_shape_buf = queue.shape_buffer(value.shape());
        let weight_shape_buf = queue.shape_buffer(weight.shape());
        let result_shape_buf = queue.shape_buffer(result.shape());

        KernelDispatch::new(queue.device(), pass, &self.main)
            .bind0([
                &value_shape_buf,
                &weight_shape_buf,
                &result_shape_buf,
                value.buffer(),
                weight.buffer(),
                result.buffer(),
                config.buffer(),
            ])
            .dispatch(1);
    }

    pub fn run_cpu<SW: Storage<f32, Dyn>>(
        out: &mut DVector<f32>,
        a: &DVector<f32>,
        w: &Vector<f32, Dyn, SW>,
    ) {
        const NUDGE_FACTOR: f32 = 1.0e-5;
        let rms = 1.0 / (a.norm_squared() / (a.nrows() as f32) + NUDGE_FACTOR).sqrt();
        out.zip_zip_apply(a, w, |o, a, w| *o = (a * rms) * w);
    }
}

#[cfg(test)]
mod test {
    use crate::ops::{RmsNorm, RmsNormConfig};
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelInvocationQueue};
    use wgcore::tensor::{GpuScalar, GpuVector};
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rms_norm() {
        let gpu = GpuInstance::new().await.unwrap();
        let rmsnorm = super::RmsNorm::from_device(gpu.device()).unwrap();
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 1757;

        let result = DVector::new_random(LEN as usize);
        let value = DVector::new_random(LEN as usize);
        let weight = DVector::new_random(LEN as usize);

        let gpu_result = GpuVector::init(
            gpu.device(),
            &result,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_value = GpuVector::init(gpu.device(), &value, BufferUsages::STORAGE);
        let gpu_weight = GpuVector::init(gpu.device(), &weight, BufferUsages::STORAGE);
        let gpu_staging = GpuVector::uninit(
            gpu.device(),
            result.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let config = GpuScalar::init(gpu.device(), RmsNormConfig {
            nudge_factor: 1.0e-6
        }, BufferUsages::UNIFORM);

        let mut pass = encoder.compute_pass("test", None);
        rmsnorm.dispatch(&mut queue, &mut pass, &config, &gpu_result, &gpu_value, &gpu_weight);
        drop(pass);

        gpu_staging.copy_from(&mut encoder, &gpu_result);

        gpu.queue().submit(Some(encoder.finish()));

        let mut cpu_result = result;
        RmsNorm::run_cpu(&mut cpu_result, &value, &weight);

        approx::assert_relative_eq!(
            DVector::from(gpu_staging.read(gpu.device()).await.unwrap()),
            cpu_result,
            epsilon = 1.0e-4
        );
    }
}

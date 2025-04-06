use bytemuck::Pod;
use nalgebra::{Dyn, StorageMut, Vector};
use wgcore::kernel::{KernelDispatch, KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuMatrixView, GpuVectorView};
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::{ComputePass, ComputePipeline};

#[derive(Shader)]
#[shader(derive(Shape), src = "softmax.wgsl", composable = false)]
/// Shader implementing the softmax kernel.
pub struct SoftMax {
    pub main: ComputePipeline,
}

impl SoftMax {
    pub fn dispatch<'a, 'b, T: Pod>(
        &'a self,
        queue: &KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        in_out_mat: impl Into<GpuMatrixView<'b, T>>,
    ) {
        let in_out_mat = in_out_mat.into();
        let shape_buf = queue.shape_buffer(in_out_mat.shape());
        KernelDispatch::new(queue.device(), pass, &self.main)
            .bind0([&shape_buf, in_out_mat.buffer()])
            .dispatch(in_out_mat.shape().size[1]);
    }

    /// The softmax function.
    ///
    /// Converts a set of real number into a probability distribution.
    /// See <https://fr.wikipedia.org/wiki/Fonction_softmax>
    pub fn run_cpu<S: StorageMut<f32, Dyn>>(vals: &mut Vector<f32, Dyn, S>) {
        // Note that llama2.c also introduces a bias based on the max value
        // to improve numerical stability. So it is effectively computing:
        // softmax(z) = (e^z - max) / (e^z - max).sum()
        let max_val = vals.max();
        let mut sum = 0.0;

        vals.apply(|x| {
            *x = (*x - max_val).exp();
            sum += *x;
        });

        *vals /= sum;
    }
}

#[cfg(test)]
mod test {
    use crate::ops::SoftMax;
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelInvocationQueue};
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_softmax() {
        let gpu = GpuInstance::new().await.unwrap();
        let softmax = super::SoftMax::from_device(gpu.device()).unwrap();
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 1757;

        let v0 = DVector::new_random(LEN as usize);
        let gpu_v0 = TensorBuilder::vector(LEN, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), v0.as_slice());
        let staging = TensorBuilder::vector(LEN, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
            .build(gpu.device());

        let mut pass = encoder.compute_pass("test", None);
        softmax.dispatch(&mut queue, &mut pass, gpu_v0.as_embedded_view());
        drop(pass);

        staging.copy_from(&mut encoder, &gpu_v0);

        gpu.queue().submit(Some(encoder.finish()));

        let mut cpu_result = v0;
        SoftMax::run_cpu(&mut cpu_result);

        approx::assert_relative_eq!(
            DVector::from(staging.read(gpu.device()).await.unwrap()),
            cpu_result,
            epsilon = 1.0e-7
        );
    }
}

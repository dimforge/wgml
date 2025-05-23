use bytemuck::Pod;
use nalgebra::DVector;
use wgcore::kernel::KernelDispatch;
use wgcore::shapes::ViewShapeBuffers;
use wgcore::tensor::GpuVectorView;
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(derive(Shape), src = "silu.wgsl", composable = false)]
/// Shader implementing the Silu activation function.
pub struct Silu {
    pub main: ComputePipeline,
}

impl Silu {
    pub fn dispatch<'a, 'b, T: Pod>(
        &'a self,
        device: &Device,
        shapes: &ViewShapeBuffers,
        pass: &mut ComputePass,
        in_out_h1: impl Into<GpuVectorView<'b, T>>,
        in_h2: impl Into<GpuVectorView<'b, T>>,
    ) {
        let h1 = in_out_h1.into();
        let h2 = in_h2.into();
        let shape_h1 = shapes.get(device, h1.shape());
        let shape_h2 = shapes.get(device, h2.shape());

        KernelDispatch::new(device, pass, &self.main)
            .bind0([&shape_h1, &shape_h2, h1.buffer(), h2.buffer()])
            .dispatch(h1.len().div_ceil(64));
    }

    pub fn run_cpu(h1: &mut DVector<f32>, h2: &DVector<f32>) {
        // SwiGLU non-linearity.
        fn swish(x: f32, beta: f32) -> f32 {
            // This is the swish function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
            x / (1.0 + (-beta * x).exp())
        }

        h1.zip_apply(h2, |h, h2| *h = h2 * swish(*h, 1.0));
    }
}

#[cfg(test)]
mod test {
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;
    use wgcore::shapes::ViewShapeBuffers;
    use wgcore::tensor::GpuVector;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_silu() {
        let gpu = GpuInstance::new().await.unwrap();
        let silu = super::Silu::from_device(gpu.device()).unwrap();
        let shapes = ViewShapeBuffers::new();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 1757;

        let h1 = DVector::new_random(LEN as usize);
        let h2 = DVector::new_random(LEN as usize);

        let gpu_h1 = GpuVector::init(
            gpu.device(),
            &h1,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_h2 = GpuVector::init(gpu.device(), &h2, BufferUsages::STORAGE);
        let gpu_staging_h1 = GpuVector::uninit(
            gpu.device(),
            h1.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut pass = encoder.compute_pass("test", None);
        silu.dispatch(gpu.device(), &shapes, &mut pass, &gpu_h1, &gpu_h2);
        drop(pass);

        gpu_staging_h1.copy_from(&mut encoder, &gpu_h1);

        gpu.queue().submit(Some(encoder.finish()));

        let mut cpu_result = h1;
        super::Silu::run_cpu(&mut cpu_result, &h2);

        approx::assert_relative_eq!(
            DVector::from(gpu_staging_h1.read(gpu.device()).await.unwrap()),
            cpu_result,
            epsilon = 1.0e-5
        );
    }
}

use super::Quantization;
use crate::quantization::{BlockQ4_K, BlockQ5_K, BlockQ8_K};
use crate::quantized_matrix::GpuQuantMatrix;
use bytemuck::Pod;
use naga_oil::compose::{ComposerError, ShaderDefValue};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::collections::HashMap;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuMatrixView, GpuVectorView, RowMajor};
use wgcore::Shader;
use wgebra::linalg::{row_major_shader_defs, Shape};
use wgebra::Gemv;
use wgpu::ComputePipeline;

const USE_OPTIMIZED: bool = false;

pub trait QuantizedValue {
    /// Number of dequantized elements the quantized value represents.
    const DEQUANTIZED_LEN: usize;
}

impl QuantizedValue for f32 {
    const DEQUANTIZED_LEN: usize = 1;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ8_0x2([u32; 17]);

impl QuantizedValue for GpuBlockQ8_0x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ4_0x2([u32; 9]);

impl QuantizedValue for GpuBlockQ4_0x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ4_1x2([u32; 10]);

impl QuantizedValue for GpuBlockQ4_1x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ5_0x2([u32; 11]);

impl QuantizedValue for GpuBlockQ5_0x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ5_1x2([u32; 12]);

impl QuantizedValue for GpuBlockQ5_1x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

pub type GpuBlockQ8_K = BlockQ8_K;
pub type GpuBlockQ5_K = BlockQ5_K;
pub type GpuBlockQ4_K = BlockQ4_K;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ6_Kx2([u32; 105]);

impl QuantizedValue for GpuBlockQ6_Kx2 {
    const DEQUANTIZED_LEN: usize = 512;
}

// SAFETY: These impls are safe, they don’t exist in bytemuck because they don’t
// provide impls for non-power-of-two largeish arrays.
unsafe impl bytemuck::Zeroable for GpuBlockQ6_Kx2 {}
unsafe impl bytemuck::Pod for GpuBlockQ6_Kx2 {}

macro_rules! impl_rand {
    ($($t: ident, $len: literal);*) => {$(
        impl Distribution<$t> for Standard {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $t {
                // TODO: are all bit representations valid?
                $t([0; $len].map(|_| 0)) // rng.gen()))
            }
        }
    )*};
}

impl_rand!(
    GpuBlockQ8_0x2, 17;
    GpuBlockQ4_0x2, 9;
    GpuBlockQ4_1x2, 10;
    GpuBlockQ5_0x2, 11;
    GpuBlockQ5_1x2, 12;
    GpuBlockQ6_Kx2, 105
);

pub struct GemvQuant {
    pub gemv_f32: Gemv,
    pub gemv_q8: GemvQ8_0x2,
    pub gemv_q5: GemvQ5_0x2,
    pub gemv_q4: GemvQ4_0x2,
    pub gemv_q5_1: GemvQ5_1x2,
    pub gemv_q4_1: GemvQ4_1x2,
    pub gemv_q8_k: GemvQ8_K,
    pub gemv_q6_k: GemvQ6_Kx2,
    pub gemv_q5_k: GemvQ5_K,
    pub gemv_q4_k: GemvQ4_K,
}

impl GemvQuant {
    pub fn from_device(device: &wgpu::Device) -> Result<Self, ComposerError> {
        Ok(Self {
            gemv_f32: Gemv::from_device(device)?,
            gemv_q8: GemvQ8_0x2::from_device(device)?,
            gemv_q5: GemvQ5_0x2::from_device(device)?,
            gemv_q5_1: GemvQ5_1x2::from_device(device)?,
            gemv_q4: GemvQ4_0x2::from_device(device)?,
            gemv_q4_1: GemvQ4_1x2::from_device(device)?,
            gemv_q8_k: GemvQ8_K::from_device(device)?,
            gemv_q6_k: GemvQ6_Kx2::from_device(device)?,
            gemv_q5_k: GemvQ5_K::from_device(device)?,
            gemv_q4_k: GemvQ4_K::from_device(device)?,
        })
    }
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q8_0x2_slow.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ8_0x2 {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q5_0x2.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ5_0x2 {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q5_1x2.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ5_1x2 {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q4_0x2_slow.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ4_0x2 {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q4_1x2.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ4_1x2 {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q8_k.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ8_K {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q6_kx2_slow.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ6_Kx2 {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q5_k_slow.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ5_K {
    pub gemv: ComputePipeline,
}

#[derive(Shader)]
#[shader(
    derive(Shape, Quantization),
    src = "gemv_quant_q4_k_slow.wgsl",
    shader_defs = "row_major_shader_defs",
    composable = false
)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ4_K {
    pub gemv: ComputePipeline,
}

impl GemvQuant {
    /// Queues this shader to compute `out = m * v`.
    pub fn queue<'a, 'b>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        out: impl Into<GpuVectorView<'b, f32>>,
        m: &GpuQuantMatrix,
        v: impl Into<GpuVectorView<'b, f32>>,
    ) {
        let out = out.into();

        // TODO: add a function to convert a View<f32> to a View<vec4<f32>>
        //       then remove `ViewShape::f32_to_vec4`.
        let v = v.into();

        // assert_eq!(
        //     m.shape().size[1],
        //     v.shape().size[0],
        //     "Gemv: dimension mismatch."
        // );
        // assert_eq!(
        //     out.shape().size[0],
        //     m.shape().size[0],
        //     "Gemv: dimension mismatch."
        // );

        let v_shape = match m {
            GpuQuantMatrix::F32(_) => v.shape(),
            GpuQuantMatrix::Q8_0(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q5_0(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q5_1(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q4_0(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q4_1(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q8_K(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q6_K(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q5_K(_) => v.shape().f32_to_vec4::<RowMajor>(),
            GpuQuantMatrix::Q4_K(_) => v.shape().f32_to_vec4::<RowMajor>(),
        };

        let optimized_shape = if USE_OPTIMIZED {
            out.shape().f32_to_vec4::<RowMajor>()
        } else {
            out.shape()
        };
        let out_shape = match m {
            GpuQuantMatrix::F32(_) => out.shape(),
            GpuQuantMatrix::Q8_0(_) => optimized_shape,
            GpuQuantMatrix::Q5_0(_) => out.shape(),
            GpuQuantMatrix::Q5_1(_) => out.shape(),
            GpuQuantMatrix::Q4_0(_) => optimized_shape,
            GpuQuantMatrix::Q4_1(_) => out.shape(),
            GpuQuantMatrix::Q8_K(_) => out.shape(),
            GpuQuantMatrix::Q6_K(_) => optimized_shape,
            GpuQuantMatrix::Q5_K(_) => optimized_shape,
            GpuQuantMatrix::Q4_K(_) => optimized_shape,
        };

        let out_shape_buf = queue.shape_buffer(out_shape);
        let v_shape_buf = queue.shape_buffer(v_shape);
        let m_shape_buf = queue.shape_buffer(m.shape());

        let kernel = match m {
            GpuQuantMatrix::F32(_) => &self.gemv_f32.gemv,
            GpuQuantMatrix::Q8_0(_) => &self.gemv_q8.gemv,
            GpuQuantMatrix::Q5_0(_) => &self.gemv_q5.gemv,
            GpuQuantMatrix::Q5_1(_) => &self.gemv_q5_1.gemv,
            GpuQuantMatrix::Q4_0(_) => &self.gemv_q4.gemv,
            GpuQuantMatrix::Q4_1(_) => &self.gemv_q4_1.gemv,
            GpuQuantMatrix::Q8_K(_) => &self.gemv_q8_k.gemv,
            GpuQuantMatrix::Q6_K(_) => &self.gemv_q6_k.gemv,
            GpuQuantMatrix::Q5_K(_) => &self.gemv_q5_k.gemv,
            GpuQuantMatrix::Q4_K(_) => &self.gemv_q4_k.gemv,
        };

        let optimized_dispatch = if USE_OPTIMIZED {
            // assert_eq!(out.shape().size[0] % 4, 0);
            out.shape().size[0] / 4
        } else {
            m.shape().size[0].div_ceil(64)
        };

        let dispatch = match m {
            GpuQuantMatrix::F32(_) => m.shape().size[0].div_ceil(64),
            GpuQuantMatrix::Q8_0(_) => optimized_dispatch,
            GpuQuantMatrix::Q5_0(_) => m.shape().size[0].div_ceil(64),
            GpuQuantMatrix::Q5_1(_) => m.shape().size[0].div_ceil(64),
            GpuQuantMatrix::Q4_0(_) => optimized_dispatch,
            GpuQuantMatrix::Q4_1(_) => m.shape().size[0].div_ceil(64),
            GpuQuantMatrix::Q8_K(_) => m.shape().size[0].div_ceil(64),
            GpuQuantMatrix::Q6_K(_) => optimized_dispatch,
            GpuQuantMatrix::Q5_K(_) => optimized_dispatch,
            GpuQuantMatrix::Q4_K(_) => optimized_dispatch,
        };

        KernelInvocationBuilder::new(queue, &kernel)
            .bind0([
                &out_shape_buf,
                &m_shape_buf,
                &v_shape_buf,
                out.buffer(),
                m.buffer(),
                v.buffer(),
            ])
            .queue(dispatch);
    }
}

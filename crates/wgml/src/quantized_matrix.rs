use crate::gguf::GgufTensorData;
use crate::ops::{
    GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ4_K, GpuBlockQ5_0x2, GpuBlockQ5_1x2, GpuBlockQ5_K,
    GpuBlockQ6_Kx2, GpuBlockQ8_0x2, GpuBlockQ8_K,
};
use wgcore::shapes::ViewShape;
use wgcore::tensor::{GpuMatrix, RowMajor};
use wgpu::Buffer;

pub enum GpuQuantMatrix {
    F32(GpuMatrix<f32>),
    Q8_0(GpuMatrix<GpuBlockQ8_0x2>),
    Q5_0(GpuMatrix<GpuBlockQ5_0x2>),
    Q5_1(GpuMatrix<GpuBlockQ5_1x2>),
    Q4_0(GpuMatrix<GpuBlockQ4_0x2>),
    Q4_1(GpuMatrix<GpuBlockQ4_1x2>),
    Q8_K(GpuMatrix<GpuBlockQ8_K>),
    Q6_K(GpuMatrix<GpuBlockQ6_Kx2>),
    Q5_K(GpuMatrix<GpuBlockQ5_K>),
    Q4_K(GpuMatrix<GpuBlockQ4_K>),
}

macro_rules! impl_from(
    ($($variant: ident, $scalar: ident);*) => {$(
        impl From<GpuMatrix<$scalar>> for GpuQuantMatrix {
            fn from(value: GpuMatrix<$scalar>) -> Self {
                Self::$variant(value)
            }
        }
    )*}
);

impl_from!(
    F32, f32;
    Q8_0, GpuBlockQ8_0x2;
    Q5_0, GpuBlockQ5_0x2;
    Q5_1, GpuBlockQ5_1x2;
    Q4_0, GpuBlockQ4_0x2;
    Q4_1, GpuBlockQ4_1x2;
    Q8_K, GpuBlockQ8_K;
    Q6_K, GpuBlockQ6_Kx2;
    Q5_K, GpuBlockQ5_K;
    Q4_K, GpuBlockQ4_K
);

impl GpuQuantMatrix {
    pub fn shape(&self) -> ViewShape {
        match self {
            Self::F32(m) => m.as_view::<RowMajor>().shape(),
            Self::Q8_0(m) => m.as_view::<RowMajor>().shape(),
            Self::Q5_0(m) => m.as_view::<RowMajor>().shape(),
            Self::Q5_1(m) => m.as_view::<RowMajor>().shape(),
            Self::Q4_0(m) => m.as_view::<RowMajor>().shape(),
            Self::Q4_1(m) => m.as_view::<RowMajor>().shape(),
            Self::Q8_K(m) => m.as_view::<RowMajor>().shape(),
            Self::Q6_K(m) => m.as_view::<RowMajor>().shape(),
            Self::Q5_K(m) => m.as_view::<RowMajor>().shape(),
            Self::Q4_K(m) => m.as_view::<RowMajor>().shape(),
        }
    }

    pub fn buffer(&self) -> &Buffer {
        match self {
            Self::F32(m) => m.buffer(),
            Self::Q8_0(m) => m.buffer(),
            Self::Q5_0(m) => m.buffer(),
            Self::Q5_1(m) => m.buffer(),
            Self::Q4_0(m) => m.buffer(),
            Self::Q4_1(m) => m.buffer(),
            Self::Q8_K(m) => m.buffer(),
            Self::Q6_K(m) => m.buffer(),
            Self::Q5_K(m) => m.buffer(),
            Self::Q4_K(m) => m.buffer(),
        }
    }
}

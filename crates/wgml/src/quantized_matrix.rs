use crate::ops::{
    GpuBlockQ4K, GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ5K, GpuBlockQ5_0x2, GpuBlockQ5_1x2,
    GpuBlockQ6Kx2, GpuBlockQ8K, GpuBlockQ8_0x2,
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
    Q8K(GpuMatrix<GpuBlockQ8K>),
    Q6K(GpuMatrix<GpuBlockQ6Kx2>),
    Q5K(GpuMatrix<GpuBlockQ5K>),
    Q4K(GpuMatrix<GpuBlockQ4K>),
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
    Q8K, GpuBlockQ8K;
    Q6K, GpuBlockQ6Kx2;
    Q5K, GpuBlockQ5K;
    Q4K, GpuBlockQ4K
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
            Self::Q8K(m) => m.as_view::<RowMajor>().shape(),
            Self::Q6K(m) => m.as_view::<RowMajor>().shape(),
            Self::Q5K(m) => m.as_view::<RowMajor>().shape(),
            Self::Q4K(m) => m.as_view::<RowMajor>().shape(),
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
            Self::Q8K(m) => m.buffer(),
            Self::Q6K(m) => m.buffer(),
            Self::Q5K(m) => m.buffer(),
            Self::Q4K(m) => m.buffer(),
        }
    }
}

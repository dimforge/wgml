//! Primitives for building LLM inferences.

mod batched_multiquery_attention;
mod gemv_quant;
mod layernorm;
mod quantization;
mod rms_norm;
mod rope;
mod silu;
mod softmax;
mod unary;

pub use batched_multiquery_attention::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams,
};
pub use gemv_quant::{
    GemvQuant, GpuBlockQ4K, GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ5K, GpuBlockQ5_0x2,
    GpuBlockQ5_1x2, GpuBlockQ6Kx2, GpuBlockQ8K, GpuBlockQ8_0x2, QuantizedValue,
};
pub use layernorm::LayerNorm;
pub use quantization::Quantization;
pub use rms_norm::{RmsNorm, RmsNormConfig};
pub use rope::{RoPE, RoPEConfig, RoPEVariant};
pub use silu::Silu;
pub use softmax::SoftMax;
pub use unary::{Unary, UnaryInplace, UnaryOp};

//! Llama2 inference on the GPU or CPU.
//!
use crate::ops::RoPEVariant;
pub use tokenizer::*;
pub use transformer::*;

pub mod cpu;
mod tokenizer;
mod transformer;

// Enum for all llama-like models that can be instantiated from
// the llama2 transformer with minor modifications.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LlamaModelType {
    Llama,
    Qwen2,
}

impl LlamaModelType {
    pub fn gguf_model_name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }

    pub fn rope_variant(self) -> RoPEVariant {
        match self {
            Self::Llama => RoPEVariant::Original,
            Self::Qwen2 => RoPEVariant::Neox,
        }
    }
}

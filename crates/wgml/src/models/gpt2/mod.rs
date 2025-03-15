//! GPT-2 inference on the GPU or CPU.

pub use self::transformer::{Gpt2, Gpt2LayerWeights, Gpt2State, Gpt2Weights};
pub use tokenizer::Gpt2Tokenizer;

pub mod cpu;
mod tokenizer;
pub mod transformer;

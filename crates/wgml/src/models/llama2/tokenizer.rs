// NOTE: similar to https://raw.githubusercontent.com/rahoua/pecca-rs/main/src/llama2/tokenizer.rs
//       but adjusted to load from gguf.

use crate::gguf::Gguf;
use std::collections::HashMap;
use std::fmt;

pub struct LlamaTokenizer {
    eos: usize,
    bos: usize,
    unk: usize,
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_index: HashMap<String, usize>,
    byte_pieces: Vec<char>, // stores all single-byte strings
}

impl LlamaTokenizer {
    /// The default chat template for this model.
    // From https://github.com/chujiezheng/chat_templates (MIT license).
    pub const CHAT_TEMPLATE: &'static str = include_str!("llama-2-chat.jinja");

    pub fn from_gguf(gguf: &Gguf) -> LlamaTokenizer {
        let vocab_scores = gguf.metadata["tokenizer.ggml.scores"]
            .as_f32_array()
            .to_vec();
        let vocab = gguf.metadata["tokenizer.ggml.tokens"]
            .as_string_array()
            .to_vec();

        let bos = gguf.metadata["tokenizer.ggml.bos_token_id"].unwrap_u32() as usize;
        let eos = gguf.metadata["tokenizer.ggml.eos_token_id"].unwrap_u32() as usize;
        let unk = gguf.metadata["tokenizer.ggml.unknown_token_id"].unwrap_u32() as usize;

        let byte_pieces: Vec<char> = (0..=256).map(|i| i as u8 as char).collect();

        let mut vocab_index = HashMap::new();
        for n in 0..vocab.len() {
            vocab_index.insert(vocab[n].clone(), n);
        }

        LlamaTokenizer {
            bos,
            eos,
            unk,
            vocab,
            vocab_scores,
            vocab_index,
            byte_pieces,
        }
    }

    pub fn eos(&self) -> usize {
        self.eos
    }

    pub fn bos(&self) -> usize {
        self.bos
    }

    pub fn unk(&self) -> usize {
        self.unk
    }

    pub fn bos_str(&self) -> &str {
        &self.vocab[self.bos]
    }

    pub fn eos_str(&self) -> &str {
        &self.vocab[self.eos]
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let text = text.replace(" ", "▁");

        let mut tokens: Vec<usize> = Vec::new();
        if bos {
            tokens.push(self.bos);
        }

        for ch in text.chars() {
            let ch_str = ch.to_string();
            match self.vocab_index.get(&ch_str) {
                Some(&id) => tokens.push(id),
                None => {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
                    for byte in ch_str.as_bytes() {
                        tokens.push(*byte as usize + 3);
                    }
                }
            }
        }

        // Look for special token <s>.
        let mut i = 0;
        while i < tokens.len().saturating_sub(2) {
            let tok = format!(
                "{}{}{}",
                self.vocab[tokens[i]],
                self.vocab[tokens[i + 1]],
                self.vocab[tokens[i + 2]]
            );

            if tok == "<s>" {
                tokens[i] = self.bos;
                tokens.remove(i + 2);
                tokens.remove(i + 1);
            }

            i += 1;
        }

        // Look for special token </s>.
        let mut i = 0;
        while i < tokens.len().saturating_sub(3) {
            let tok = format!(
                "{}{}{}{}",
                self.vocab[tokens[i]],
                self.vocab[tokens[i + 1]],
                self.vocab[tokens[i + 2]],
                self.vocab[tokens[i + 3]],
            );

            if tok == "</s>" {
                tokens[i] = self.eos;
                tokens.remove(i + 3);
                tokens.remove(i + 2);
                tokens.remove(i + 1);
            }

            i += 1;
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0;
            let mut best_idx = None;

            for i in 0..tokens.len() - 1 {
                let pair = format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
                if let Some(&id) = self.vocab_index.get(&pair) {
                    if self.vocab_scores[id] > best_score {
                        best_score = self.vocab_scores[id];
                        best_id = id;
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_idx {
                tokens[idx] = best_id;
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(self.eos);
        }

        tokens
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        let mut piece = self.vocab[token].as_str();
        if prev_token == 1 {
            piece = piece.trim_start();
        }
        if let Some(hex) = piece.strip_prefix("<0x") {
            if let Ok(byte) = usize::from_str_radix(&hex[..2], 16) {
                return self.byte_pieces[byte].to_string();
            }
        }
        piece.replace("▁", " ").to_string()
    }
}

impl fmt::Debug for LlamaTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tokenizer with vocab size: {}", self.vocab.len())
    }
}

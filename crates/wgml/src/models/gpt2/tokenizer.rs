// NOTE: this tokenizer was copied from https://github.com/nreHieW/r-nn/blob/main/examples/gpt2/tokenizer.rs
//       (Apache 2 license) and a bit modified to load from gguf metadata.

#![allow(dead_code)]
use crate::gguf::Gguf;
use regex::Regex;
use std::path::Path;
use std::{
    collections::{HashMap, HashSet},
    fs,
};

fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    for i in 0..word.len() - 1 {
        pairs.insert((word[i].clone(), word[i + 1].clone()));
    }
    pairs
}

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs: Vec<u16> = bs.iter().map(|&b| b as u16).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().map(|n| char::from_u32(n as u32).unwrap()))
        .collect()
}

pub struct Gpt2Tokenizer {
    bos: usize,
    eos: usize,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    encoder: HashMap<String, u32>,
    decoder: HashMap<u64, String>,
    merges: HashMap<(String, String), usize>,
    pat: Regex,
}

fn read_json_file(path: impl AsRef<Path>) -> serde_json::Value {
    let data = fs::read_to_string(path).expect("Unable to read file");
    serde_json::from_str(&data).expect("Unable to parse JSON")
}

impl Gpt2Tokenizer {
    pub fn from_gguf(gguf: &Gguf) -> Self {
        let vocab = gguf.metadata["tokenizer.ggml.tokens"].as_string_array();
        let bpe_merges = gguf.metadata["tokenizer.ggml.merges"].as_string_array();
        let bos = gguf.metadata["tokenizer.ggml.bos_token_id"].unwrap_u32() as usize;
        let eos = gguf.metadata["tokenizer.ggml.eos_token_id"].unwrap_u32() as usize;
        let mut merges = HashMap::new();
        let mut merge_pairs = vec![];
        bpe_merges.iter().enumerate().for_each(|(i, line)| {
            let mut parts = line.split_whitespace();
            let pair = (
                parts.next().unwrap().to_string(),
                parts.next().unwrap().to_string(),
            );
            merge_pairs.push(pair.clone());
            merges.insert(pair, i);
        });

        let pat = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .unwrap();
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(a, &b)| (b, *a)).collect();
        let mut decoder = HashMap::new();
        for (value, key) in vocab.iter().enumerate() {
            decoder.insert(value as u64, key.clone());
        }
        let encoder: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, str)| (str.clone(), i as u32))
            .collect();

        Self {
            bos,
            eos,
            byte_encoder: bytes_to_unicode(),
            byte_decoder,
            encoder,
            decoder,
            merges,
            pat,
        }
    }

    pub fn eos(&self) -> usize {
        self.eos
    }

    pub fn bos(&self) -> usize {
        self.bos
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut bpe_tokens = Vec::new();

        let tokenize = |substr, out: &mut Vec<usize>| {
            for token in self.pat.find_iter(substr) {
                let token = token.as_str();
                let encoded_token: String = token
                    .as_bytes()
                    .iter()
                    .map(|&b| self.byte_encoder[&b])
                    .collect();
                for bpe_token in self.bpe(&encoded_token).split(' ') {
                    if let Some(token_id) = self.encoder.get(bpe_token) {
                        out.push(*token_id as usize);
                    }
                }
            }
        };

        // TODO: don’t hard-code the special tokens?
        let regex = Regex::new("<[^<>]*>").unwrap();
        let mut cursor = 0;
        for m in regex.find_iter(text) {
            if cursor != m.start() {
                // Tokenize the non-matched piece.
                tokenize(&text[cursor..m.start()], &mut bpe_tokens);
            }
            // Tokenize the special string.
            println!("Found match: {}", m.as_str());
            bpe_tokens.push(self.encoder[m.as_str()] as usize);
            cursor = m.end();
        }

        // Tokenize the rest that didn’t match.
        if cursor < text.len() {
            tokenize(&text[cursor..], &mut bpe_tokens);
        }

        bpe_tokens
    }

    pub fn bos_str(&self) -> &str {
        self.decoder.get(&(self.bos as u64)).unwrap()
    }

    pub fn eos_str(&self) -> &str {
        self.decoder.get(&(self.eos as u64)).unwrap()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let text: Vec<String> = tokens
            .iter()
            .filter_map(|&token| self.decoder.get(&(token as u64)).cloned())
            .collect();
        let decoded: String = text
            .join("")
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c))
            .map(|&b| b as char)
            .collect();
        decoded
    }

    fn bpe(&self, token: &str) -> String {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        let mut pairs = get_pairs(&word);

        if pairs.is_empty() {
            return token.to_string();
        }

        loop {
            let bigram = pairs
                .iter()
                .min_by_key(|&pair| self.merges.get(pair).unwrap_or(&usize::MAX))
                .cloned();

            if let Some((first, second)) = bigram {
                let mut new_word = Vec::new();
                if !self.merges.contains_key(&(first.clone(), second.clone())) {
                    break;
                }
                let mut i = 0;
                while i < word.len() {
                    if let Some(j) = word[i..].iter().position(|x| x == &first) {
                        new_word.extend(word[i..i + j].iter().cloned());
                        i += j;
                        if i < word.len() - 1 && word[i + 1] == second {
                            new_word.push(first.clone() + &second);
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    } else {
                        new_word.extend(word[i..].iter().cloned());
                        break;
                    }
                }
                word = new_word;
                if word.len() == 1 {
                    break;
                } else {
                    pairs = get_pairs(&word);
                }
            } else {
                break;
            }
        }

        word.join(" ")
    }
}

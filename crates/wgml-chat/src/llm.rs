use crate::chat_gpt2::ChatGpt2;
use crate::chat_llama2::ChatLlama2;
use crate::chat_template::ChatTemplate;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::SamplerParams;
use dioxus::hooks::UnboundedSender;
use wgcore::gpu::GpuInstance;
use wgcore::re_exports::Device;
use wgml::gguf::{Gguf, GgufMetadataValue};
use wgml::models::gpt2::Gpt2Tokenizer;
use wgml::models::llama2::{LlamaModelType, LlamaTokenizer};

pub enum ChatLlm {
    Llama(ChatLlama2),
    Qwen(ChatLlama2),
    Gpt2(ChatGpt2),
}

impl ChatLlm {
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Gpt2(_) => "gpt2",
            Self::Llama(_) => "llama",
            Self::Qwen(_) => "qwen2",
        }
    }
}

impl ChatLlm {
    pub fn from_gguf(device: &Device, gguf: &Gguf) -> anyhow::Result<Self> {
        let Some(GgufMetadataValue::String(name)) = gguf.metadata.get("general.architecture")
        else {
            anyhow::bail!("Unrecognized model")
        };

        if name.to_lowercase().contains("llama") {
            Ok(Self::Llama(ChatLlama2::from_gguf(device, gguf)?))
        } else if name.to_lowercase().contains("gpt") {
            Ok(Self::Gpt2(ChatGpt2::from_gguf(device, gguf)?))
        } else if name.to_lowercase().contains("qwen2") {
            Ok(Self::Qwen(ChatLlama2::from_gguf_with_model_type(
                device,
                gguf,
                LlamaModelType::Qwen2,
            )?))
        } else {
            anyhow::bail!("Unrecognized model")
        }
    }

    pub async fn forward(
        &self,
        gpu: &GpuInstance,
        prompt: Prompt,
        sampler_params: SamplerParams,
        chat_template: ChatTemplate,
        next_pos: usize,
        out: UnboundedSender<ChatEvent>,
    ) {
        match self {
            Self::Llama(llm) => {
                llm.forward(gpu, prompt, sampler_params, chat_template, next_pos, out)
                    .await
            }
            Self::Qwen(llm) => {
                llm.forward(gpu, prompt, sampler_params, chat_template, next_pos, out)
                    .await
            }
            Self::Gpt2(_llm) => {
                todo!()
                // llm.forward(gpu, prompt, sampler_params, next_pos, out)
                //     .await
            }
        }
    }
}

pub enum AnyTokenizer {
    Llama(LlamaTokenizer),
    Gpt2(Gpt2Tokenizer),
}

impl AnyTokenizer {
    pub fn from_gguf(gguf: &Gguf) -> anyhow::Result<Self> {
        let tokenizer_type = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .ok_or(anyhow::anyhow!("Missing tokenizer.ggml.model"))?
            .as_string();
        if tokenizer_type == "gpt2" {
            Ok(AnyTokenizer::Gpt2(Gpt2Tokenizer::from_gguf(gguf)))
        } else if tokenizer_type == "llama" {
            Ok(AnyTokenizer::Llama(LlamaTokenizer::from_gguf(gguf)))
        } else {
            anyhow::bail!("Unrecognized tokenizer type: {}", tokenizer_type)
        }
    }

    pub fn eos(&self) -> usize {
        match self {
            Self::Llama(t) => t.eos(),
            Self::Gpt2(t) => t.eos(),
        }
    }

    #[allow(dead_code)]
    pub fn bos(&self) -> usize {
        match self {
            Self::Llama(t) => t.bos(),
            Self::Gpt2(t) => t.bos(),
        }
    }

    pub fn bos_str(&self) -> &str {
        match self {
            Self::Llama(t) => t.bos_str(),
            Self::Gpt2(t) => t.bos_str(),
        }
    }

    pub fn eos_str(&self) -> &str {
        match self {
            Self::Llama(t) => t.eos_str(),
            Self::Gpt2(t) => t.eos_str(),
        }
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        match self {
            Self::Llama(t) => t.decode(prev_token, token),
            Self::Gpt2(t) => t.decode(&[token as u32]),
        }
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        match self {
            Self::Llama(t) => t.encode(text, bos, eos),
            // TODO: auto-instert bos/eos based on the flag?
            Self::Gpt2(t) => t.encode(text),
        }
    }
}

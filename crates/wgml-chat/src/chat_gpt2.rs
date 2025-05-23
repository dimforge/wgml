use crate::llm::AnyTokenizer;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::{sample_next_token, SamplerParams};
use async_channel::Sender;
use nalgebra::DVector;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::CommandEncoderExt;
use wgcore::re_exports::bytemuck;
use wgcore::re_exports::Device;
use wgcore::shapes::ViewShapeBuffers;
use wgml::gguf::Gguf;
use wgml::models::gpt2::cpu::Gpt2Params;
use wgml::models::gpt2::{Gpt2, Gpt2State, Gpt2Weights};
use wgml::ops::BatchedMultiqueryAttentionParams;

pub struct ChatGpt2 {
    transformer: Gpt2,
    weights: Gpt2Weights,
    tokenizer: AnyTokenizer,
    config: Gpt2Params,
    state: Gpt2State,
    shapes: ViewShapeBuffers,
}

impl ChatGpt2 {
    pub fn from_gguf(device: &Device, gguf: &Gguf) -> anyhow::Result<ChatGpt2> {
        let transformer = Gpt2::new(device)?;
        let config = Gpt2Params::from_gguf(gguf);
        let weights = Gpt2Weights::from_gguf(device, &config, gguf);
        let tokenizer = AnyTokenizer::from_gguf(gguf)?;
        let state = Gpt2State::new(device, &config);

        Ok(Self {
            transformer,
            weights,
            tokenizer,
            config,
            state,
            shapes: ViewShapeBuffers::new(),
        })
    }

    pub async fn forward(
        &self,
        gpu: &GpuInstance,
        prompt: Prompt,
        sampler_params: SamplerParams,
        start_pos: usize,
        out: Sender<ChatEvent>,
    ) {
        let (mut sampler, mut sampler_res) = sampler_params.sampler();

        let prompt_toks =
            self.tokenizer
                .encode(prompt.entries().last().unwrap().as_str(), false, false);

        // let start = Instant::now();
        let mut token = prompt_toks[0];
        let start = web_time::Instant::now();

        for pos in start_pos.. {
            let head_size = self.config.n_embd / self.config.n_head;
            let attn_params = BatchedMultiqueryAttentionParams {
                seq_len: self.config.n_seq as u32,
                kv_dim: self.config.n_embd as u32,
                kv_mul: 1,
                n_heads: self.config.n_head as u32,
                head_size: head_size as u32,
                pos: pos as u32,
            };

            // Run the transformer.
            // let t0 = Instant::now();
            let mut logits = {
                let mut encoder = gpu.device().create_command_encoder(&Default::default());
                gpu.queue().write_buffer(
                    self.state.attn_params().buffer(),
                    0,
                    bytemuck::cast_slice(&[attn_params]),
                );

                let mut main_pass = encoder.compute_pass("main_pass", None);
                self.transformer.dispatch(
                    gpu.device(),
                    &self.shapes,
                    &mut main_pass,
                    &self.state,
                    &self.weights,
                    &self.config,
                    token as u32,
                    pos as u32,
                );
                drop(main_pass);

                self.state
                    .logits_readback()
                    .copy_from(&mut encoder, self.state.logits());
                gpu.queue().submit(Some(encoder.finish()));

                // TODO: don’t allocate for the readback.
                DVector::from(
                    self.state
                        .logits_readback()
                        .read(gpu.device())
                        .await
                        .unwrap(),
                )

                // transformer_time += t0.elapsed().as_secs_f64();
            };

            // Find the token and loop.
            let next = sample_next_token(
                &mut sampler,
                &mut sampler_res,
                &mut logits,
                &prompt_toks,
                pos,
            );
            let prev_token = token;
            token = next;

            if pos + 1 >= prompt_toks.len() {
                if token == self.tokenizer.eos() {
                    break;
                } else {
                    let token_string = self.tokenizer.decode(prev_token, token);
                    if out
                        .send(ChatEvent::Token {
                            string: token_string.clone(),
                            next_pos: pos,
                            token_count: pos - start_pos,
                            token_time: start.elapsed().as_secs_f64(),
                        })
                        .await
                        .is_err()
                    {
                        // Early-exit if an error was returned.
                        return;
                    }
                }
            };
        }
        // println!(
        //     "\n[GPU] achieved tok/s: {}, transformer time: {}, kernels time: {}, queue_time: {}",
        //     steps as f64 / start.elapsed().as_secs_f64(),
        //     transformer_time / steps as f64,
        //     kernels_time / steps as f64,
        //     queue_time / steps as f64,
        // );
    }
}

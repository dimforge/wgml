use crate::chat_template::ChatTemplate;
use crate::llm::AnyTokenizer;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::{sample_next_token, SamplerParams};
use dioxus::hooks::UnboundedSender;
use nalgebra::DVector;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::CommandEncoderExt;
use wgcore::re_exports::bytemuck;
use wgcore::re_exports::Device;
use wgcore::shapes::ViewShapeBuffers;
use wgml::gguf::Gguf;
use wgml::models::llama2::cpu::Llama2Config;
use wgml::models::llama2::{Llama2, Llama2State, Llama2Weights, LlamaModelType};

pub struct ChatLlama2 {
    transformer: Llama2,
    weights: Llama2Weights,
    tokenizer: AnyTokenizer,
    config: Llama2Config,
    state: Llama2State,
    shapes: ViewShapeBuffers,
}

impl ChatLlama2 {
    pub fn from_gguf(device: &Device, gguf: &Gguf) -> anyhow::Result<ChatLlama2> {
        Self::from_gguf_with_model_type(device, gguf, LlamaModelType::Llama)
    }

    pub fn from_gguf_with_model_type(
        device: &Device,
        gguf: &Gguf,
        model_type: LlamaModelType,
    ) -> anyhow::Result<ChatLlama2> {
        let transformer = Llama2::new(device, model_type)?;
        let config = Llama2Config::from_gguf_with_model_type(gguf, model_type);
        let weights = Llama2Weights::from_gguf(device, &config, gguf);
        let tokenizer = AnyTokenizer::from_gguf(gguf)?;
        let state = Llama2State::new(device, &config);

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
        template: ChatTemplate,
        start_pos: usize,
        out: UnboundedSender<ChatEvent>,
    ) {
        log::info!("Original prompt:\n{}", prompt);

        let bos_str = self.tokenizer.bos_str();
        let eos_str = self.tokenizer.eos_str();
        println!("eos_str: {}, bos_str: {}", eos_str, bos_str);
        let prompt_str = template.apply(&prompt, bos_str, eos_str);
        println!("Forwarding prompt: ’’’{}’’’", prompt_str);
        if out
            .unbounded_send(ChatEvent::TemplatedPrompt(prompt_str.clone()))
            .is_err()
        {
            return;
        }

        let (mut sampler, mut sampler_res) = sampler_params.sampler();
        let prompt_toks = self.tokenizer.encode(&prompt_str, false, false);
        // let prompt_toks =
        //     self.tokenizer
        //         .encode(&prompt_str, !prompt_str.starts_with(&bos_str), false);
        log::info!("Promp tokens: {:?}", prompt_toks);

        let prompt_toks_map: Vec<_> = prompt_toks
            .iter()
            .map(|tok| {
                let tok_str = self.tokenizer.decode(0, *tok);
                (*tok, tok_str)
            })
            .collect();
        if out
            .unbounded_send(ChatEvent::PromptTokens(prompt_toks_map))
            .is_err()
        {
            return;
        }

        // Skip the first token in the tok/s timing since it is particularly slow due to gpu initialization.
        let timing_delay = 1;

        let mut token = prompt_toks[start_pos];
        let mut start = None;
        let mut logits = DVector::zeros(self.config.vocab_size);

        for pos in start_pos.. {
            if pos == start_pos + timing_delay {
                start = Some(web_time::Instant::now());
            }

            // let t0 = std::time::Instant::now();
            self.forward_logits(gpu, &self.shapes, pos as u32, token as u32, &mut logits)
                .await;
            // let elapsed = t0.elapsed().as_secs_f64();
            // println!("Logits time: {} (= {:.3} tok/s)", elapsed, 1.0 / elapsed);

            // let t0 = std::time::Instant::now();
            let next = sample_next_token(
                &mut sampler,
                &mut sampler_res,
                &mut logits,
                &prompt_toks,
                pos,
            );
            let token_string = self.tokenizer.decode(token, next);
            // println!("Sampling time: {}", t0.elapsed().as_secs_f64());
            token = next;

            if pos + 1 >= prompt_toks.len() {
                if token == self.tokenizer.eos() {
                    break;
                } else {
                    let (token_count, token_time) = if let Some(start) = &start {
                        (
                            pos - start_pos - timing_delay,
                            start.elapsed().as_secs_f64(),
                        )
                    } else {
                        (0, 0.0)
                    };

                    if out
                        .unbounded_send(ChatEvent::Token {
                            string: token_string,
                            next_pos: pos,
                            token_count,
                            token_time,
                        })
                        .is_err()
                    {
                        // Early-exit if an error was returned.
                        return;
                    }
                }
            }
        }
    }

    async fn forward_logits(
        &self,
        gpu: &GpuInstance,
        shapes: &ViewShapeBuffers,
        pos: u32,
        token: u32,
        out: &mut DVector<f32>,
    ) {
        shapes.clear_tmp();

        let (rope_config, rms_norm_config, attn_params) = self.config.derived_configs(pos);

        // Run the transformer.
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        gpu.queue().write_buffer(
            self.state.rope_config().buffer(),
            0,
            bytemuck::cast_slice(&[rope_config]),
        );
        gpu.queue().write_buffer(
            self.state.rms_norm_config().buffer(),
            0,
            bytemuck::cast_slice(&[rms_norm_config]),
        );
        gpu.queue().write_buffer(
            self.state.attn_params().buffer(),
            0,
            bytemuck::cast_slice(&[attn_params]),
        );

        self.state
            .x
            .copy_from_view(&mut encoder, self.weights.token_embd.column(token));

        // let t0 = std::time::Instant::now();
        let mut compute_pass = encoder.compute_pass("transformer", None);
        self.transformer.dispatch(
            gpu.device(),
            shapes,
            gpu.queue(),
            &mut compute_pass,
            &self.state,
            &self.weights,
            &self.config,
            &attn_params,
            pos,
        );
        drop(compute_pass);
        // println!("queue time: {}", t0.elapsed().as_secs_f64());

        self.state
            .logits_readback()
            .copy_from(&mut encoder, self.state.logits());

        // DEBUG: uncomment if useful for debugging the transformer
        // self.state.xb_read.copy_from(&mut encoder, &self.state.xb);
        // self.state.q_read.copy_from(&mut encoder, &self.state.q);

        gpu.queue().submit(Some(encoder.finish()));

        // let t0 = std::time::Instant::now();
        self.state
            .logits_readback()
            .read_to(gpu.device(), out.as_mut_slice())
            .await
            .unwrap();
        // log::info!("logits readback: {}", t0.elapsed().as_secs_f64());

        // if pos == 1 {
        //     let debug = self.state.q_read.read(gpu.device()).await.unwrap();
        //     println!("debug: {:?}", &debug[120..130]);
        //     std::process::exit(0);
        // }
    }
}

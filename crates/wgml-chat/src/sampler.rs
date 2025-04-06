use llm_samplers::prelude::{Logits, SampleLocallyTypical, SampleTemperature, SampleTopP};
use llm_samplers::samplers::{SampleRandDistrib, SampleTopK};
use llm_samplers::types::{SamplerChain, SimpleSamplerResources};
use nalgebra::DVector;
use rand::SeedableRng;
use wgml::models::llama2::LlamaTokenizer;

#[derive(Copy, Clone)]
pub struct TopKParams {
    pub k: usize,
    pub min_keep: usize,
}

#[derive(Copy, Clone)]
pub struct LocallyTypicalParams {
    pub p: f32,
    pub min_keep: usize,
}

#[derive(Copy, Clone)]
pub struct TopPParams {
    pub p: f32,
    pub min_keep: usize,
}

#[derive(Copy, Clone)]
pub struct TemperatureParams {
    pub temperature: f32,
}

#[derive(Copy, Clone)]
pub struct SamplerParams {
    pub top_k: TopKParams,
    pub top_k_enabled: bool,
    pub typical: LocallyTypicalParams,
    pub typical_enabled: bool,
    pub top_p: TopPParams,
    pub top_p_enabled: bool,
    pub temperature: TemperatureParams,
    pub temperature_enabled: bool,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            top_k: TopKParams { k: 40, min_keep: 1 },
            top_k_enabled: true,
            typical: LocallyTypicalParams {
                p: 1.0,
                min_keep: 1,
            },
            typical_enabled: true,
            top_p: TopPParams {
                p: 0.950,
                min_keep: 1,
            },
            top_p_enabled: true,
            temperature: TemperatureParams { temperature: 0.7 },
            temperature_enabled: true,
        }
    }
}

impl SamplerParams {
    pub fn sampler(&self) -> (SamplerChain, SimpleSamplerResources) {
        let mut result = SamplerChain::new();

        if self.top_k_enabled {
            result += SampleTopK::new(self.top_k.k, self.top_k.min_keep);
        }
        if self.typical_enabled {
            result += SampleLocallyTypical::new(self.typical.p, self.typical.min_keep);
        }
        if self.top_p_enabled {
            result += SampleTopP::new(self.top_p.p, self.top_p.min_keep);
        }
        if self.temperature_enabled {
            result += SampleTemperature::new(self.temperature.temperature);
        }

        result += SampleRandDistrib::new();

        let resources = SimpleSamplerResources::new(
            Some(Box::new(rand::rngs::StdRng::from_entropy())),
            Some(vec![]),
        );

        (result, resources)
    }
}

pub fn sample_next_token(
    sampler: &mut SamplerChain,
    sampler_res: &mut SimpleSamplerResources,
    logits: &mut DVector<f32>,
    prompt_toks: &[usize],
    pos: usize,
) -> usize {
    use llm_samplers::types::HasSamplerResources;
    use llm_samplers::types::Sampler;

    // Find the token and loop.
    let is_forced_token = pos < prompt_toks.len() - 1;

    const USE_OWN_SAMPLER: bool = true;
    if USE_OWN_SAMPLER {
        if is_forced_token {
            prompt_toks[pos + 1]
        } else {
            let mut sampler = wgml::models::sampler::Sampler::new(logits.len(), 0.7, 0.95);
            let next = sampler.sample(logits);
            next
        }
    } else {
        let next = if is_forced_token {
            prompt_toks[pos + 1]
        } else {
            // PERF: we are allocating too much for the logits (once for the readback, once here).
            // println!("logits: {:?}", logits);
            let t0 = std::time::Instant::now();
            let mut logits = Logits::try_from_iter(logits.iter().copied()).unwrap();
            println!("Logits recreate: {}", t0.elapsed().as_secs_f32());

            for logit in logits.iter_mut() {
                logit.prob = logit.logit;
            }
            logits.ensure_softmax();
            logits.set_softmax(true);

            let t0 = std::time::Instant::now();
            let res = sampler
                .sample_token(sampler_res, &mut logits)
                .unwrap()
                .unwrap_or(0) as usize; // Tokenizer::UNKNOWN as u32) as usize
            println!("Actual sampling: {}", t0.elapsed().as_secs_f32());
            res
        };

        sampler_res
            .with_last_tokens_mut(&mut |tokens| tokens.push(next as u32))
            .unwrap();

        next
    }
}

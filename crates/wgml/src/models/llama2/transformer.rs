use crate::gguf::Gguf;
use crate::models::llama2::cpu::{Llama2Config, TransformerLayerWeights, TransformerWeights};
use crate::models::llama2::LlamaModelType;
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, GemvQuant, RmsNorm, RoPE,
    RoPEShape, RoPEVariant, Silu, SoftMax,
};
use crate::quantized_matrix::GpuQuantMatrix;
use naga_oil::compose::ComposerError;
use nalgebra::{DMatrix, DVector};
use wgcore::kernel::KernelInvocationQueue;
use wgcore::tensor::{GpuMatrix, GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::linalg::{Gemv, OpAssign, OpAssignVariant};
use wgpu::{BufferUsages, Device};

pub struct Llama2State {
    /// Activation at current time stamp.
    pub x: GpuVector<f32>,
    /// Activation at current time stamp, inside a residual branch.
    pub xb: GpuVector<f32>,
    // DEBUG: useful for debugging the transformer.
    // pub xb_read: GpuVector<f32>,
    /// Additional buffer for convenience.
    xb2: GpuVector<f32>,
    /// Buffer for hidden dimension in the Feed-Forward net.
    hb: GpuVector<f32>,
    /// Another buffer for hidden dimension in the Feed-Forward net.
    hb2: GpuVector<f32>,
    /// Query.
    pub q: GpuVector<f32>,
    // DEBUG: useful for debugging the transformer.
    // pub q_read: GpuVector<f32>,
    /// Scores/attention values.
    att: GpuMatrix<f32>,
    /// Output logits.
    logits: GpuVector<f32>,
    logits_readback: GpuVector<f32>,
    // KV cache. Each Vec contains `layer` elements.
    key_cache: Vec<GpuMatrix<f32>>,
    value_cache: Vec<GpuMatrix<f32>>,
    rope_shape: GpuScalar<RoPEShape>,
    attn_params: GpuScalar<BatchedMultiqueryAttentionParams>,
}

impl Llama2State {
    pub fn new(device: &Device, config: &Llama2Config) -> Self {
        println!("config: {:?}", config);
        let q_head_size = config.dim / config.n_q_heads;
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_q_heads;
        const STORAGE: BufferUsages = BufferUsages::STORAGE;
        const UNIFORM: BufferUsages = BufferUsages::UNIFORM;

        Self {
            x: GpuVector::uninit(device, config.dim as u32, STORAGE | BufferUsages::COPY_DST),
            xb: GpuVector::uninit(device, config.dim as u32, STORAGE | BufferUsages::COPY_SRC),
            // DEBUG: useful for debugging the transformer.
            // xb_read: GpuVector::uninit(
            //     device,
            //     config.dim as u32,
            //     BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            // ),
            xb2: GpuVector::uninit(device, config.dim as u32, STORAGE),
            hb: GpuVector::uninit(device, config.hidden_dim as u32, STORAGE),
            hb2: GpuVector::uninit(device, config.hidden_dim as u32, STORAGE),
            q: GpuVector::uninit(device, config.dim as u32, STORAGE | BufferUsages::COPY_SRC),
            // DEBUG: useful for debugging the transformer.
            // q_read: GpuVector::uninit(
            //     device,
            //     config.dim as u32,
            //     BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            // ),
            // TODO: for these two, the `kv_dim` doesn’t match the dimension in the field’s comment.
            key_cache: (0..config.n_layers)
                .map(|_| GpuMatrix::uninit(device, kv_dim as u32, config.seq_len as u32, STORAGE))
                .collect(),
            value_cache: (0..config.n_layers)
                .map(|_| GpuMatrix::uninit(device, kv_dim as u32, config.seq_len as u32, STORAGE))
                .collect(),
            att: GpuMatrix::uninit(
                device,
                config.seq_len as u32,
                config.n_q_heads as u32,
                STORAGE,
            ),
            logits: GpuVector::uninit(
                device,
                config.vocab_size as u32,
                STORAGE | BufferUsages::COPY_SRC,
            ),
            logits_readback: GpuVector::uninit(
                device,
                config.vocab_size as u32,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
            rope_shape: GpuScalar::uninit(device, UNIFORM | BufferUsages::COPY_DST),
            attn_params: GpuScalar::uninit(device, UNIFORM | BufferUsages::COPY_DST),
        }
    }

    pub fn rope_shape(&self) -> &GpuScalar<RoPEShape> {
        &self.rope_shape
    }

    pub fn attn_params(&self) -> &GpuScalar<BatchedMultiqueryAttentionParams> {
        &self.attn_params
    }

    pub fn logits(&self) -> &GpuVector<f32> {
        &self.logits
    }

    pub fn logits_readback(&self) -> &GpuVector<f32> {
        &self.logits_readback
    }
}

pub struct Llama2LayerWeights {
    pub attn_norm: GpuVector<f32>,
    pub attn_k: GpuQuantMatrix,
    pub attn_q: GpuQuantMatrix,
    pub attn_v: GpuQuantMatrix,
    pub attn_k_bias: Option<GpuVector<f32>>,
    pub attn_q_bias: Option<GpuVector<f32>>,
    pub attn_v_bias: Option<GpuVector<f32>>,
    pub ffn_down: GpuQuantMatrix,
    pub ffn_gate: GpuQuantMatrix,
    pub ffn_norm: GpuVector<f32>,
    pub ffn_up: GpuQuantMatrix,
    pub attn_output: GpuQuantMatrix,
}

pub struct Llama2Weights {
    pub layers: Vec<Llama2LayerWeights>,
    pub token_embd: GpuMatrix<f32>,
    pub output: GpuQuantMatrix,
    pub output_norm: GpuVector<f32>,
}

impl Llama2Weights {
    pub fn from_gguf(device: &Device, config: &Llama2Config, gguf: &Gguf) -> Self {
        let usage = BufferUsages::STORAGE;

        let head_size = config.dim / config.n_q_heads;
        let num_kv_heads_times_head_size = config.n_kv_heads * head_size;

        let mut layers = vec![];

        for i_layer in 0..config.n_layers {
            log::info!("Loop {}/{}", i_layer, config.n_layers);
            let attn_q = format!("blk.{}.attn_q.weight", i_layer);
            let attn_k = format!("blk.{}.attn_k.weight", i_layer);
            let attn_v = format!("blk.{}.attn_v.weight", i_layer);
            let attn_q_bias = format!("blk.{}.attn_q.bias", i_layer);
            let attn_k_bias = format!("blk.{}.attn_k.bias", i_layer);
            let attn_v_bias = format!("blk.{}.attn_v.bias", i_layer);
            let attn_output = format!("blk.{}.attn_output.weight", i_layer);
            let ffn_down = format!("blk.{}.ffn_down.weight", i_layer);
            let ffn_gate = format!("blk.{}.ffn_gate.weight", i_layer);
            let ffn_up = format!("blk.{}.ffn_up.weight", i_layer);
            let ffn_norm = format!("blk.{}.ffn_norm.weight", i_layer);
            let attn_norm = format!("blk.{}.attn_norm.weight", i_layer);

            let attn_q = gguf.tensors[&attn_q]
                .data()
                .to_gpu_matrix(device, config.dim, config.dim)
                .unwrap();
            let attn_k = gguf.tensors[&attn_k]
                .data()
                .to_gpu_matrix(device, num_kv_heads_times_head_size, config.dim)
                .unwrap();
            let attn_v = gguf.tensors[&attn_v]
                .data()
                .to_gpu_matrix(device, num_kv_heads_times_head_size, config.dim)
                .unwrap();
            let attn_q_bias = gguf
                .tensors
                .get(&attn_q_bias)
                .map(|t| GpuVector::init(device, t.data().as_f32().unwrap(), usage));
            let attn_k_bias = gguf
                .tensors
                .get(&attn_k_bias)
                .map(|t| GpuVector::init(device, t.data().as_f32().unwrap(), usage));
            let attn_v_bias = gguf
                .tensors
                .get(&attn_v_bias)
                .map(|t| GpuVector::init(device, t.data().as_f32().unwrap(), usage));
            let attn_output = gguf.tensors[&attn_output]
                .data()
                .to_gpu_matrix(device, config.dim, config.dim)
                .unwrap();
            let ffn_down = gguf.tensors[&ffn_down]
                .data()
                .to_gpu_matrix(device, config.dim, config.hidden_dim)
                .unwrap();
            let ffn_gate = gguf.tensors[&ffn_gate]
                .data()
                .to_gpu_matrix(device, config.hidden_dim, config.dim)
                .unwrap();
            let ffn_up = gguf.tensors[&ffn_up]
                .data()
                .to_gpu_matrix(device, config.hidden_dim, config.dim)
                .unwrap();

            let ffn_norm = gguf.tensors[&ffn_norm].data().as_f32().unwrap();
            let attn_norm = gguf.tensors[&attn_norm].data().as_f32().unwrap();

            layers.push(Llama2LayerWeights {
                attn_k,
                attn_norm: GpuVector::init(device, &attn_norm, usage),
                attn_q,
                attn_v,
                attn_k_bias,
                attn_q_bias,
                attn_v_bias,
                ffn_down,
                ffn_gate,
                ffn_norm: GpuVector::init(device, &ffn_norm, usage),
                ffn_up,
                attn_output,
            });
        }

        log::info!("Loop done");
        let token_embd_name = "token_embd.weight";
        let output = "output.weight";
        let output_norm = "output_norm.weight";

        // TODO: keep the token embeddings in quantized form
        let token_embd = &gguf.tensors[token_embd_name].data().dequantize().unwrap();
        let token_embd = DMatrix::from_column_slice(config.dim, config.vocab_size, token_embd);
        let token_embd = GpuMatrix::init(device, &token_embd, usage | BufferUsages::COPY_SRC);

        let output = gguf
            .tensors
            .get(output)
            .map(|v| {
                v.data()
                    .to_gpu_matrix(device, config.vocab_size, config.dim)
                    .unwrap()
            })
            .unwrap_or_else(|| {
                gguf.tensors[token_embd_name]
                    .data()
                    .to_gpu_matrix(device, config.vocab_size, config.dim)
                    .unwrap()
            });
        let output_norm = gguf.tensors[output_norm].data().as_f32().unwrap();
        let output_norm = DVector::from_row_slice(output_norm);
        let output_norm = GpuVector::init(device, &output_norm, usage);

        Self {
            layers,
            token_embd,
            output,
            output_norm,
        }
    }
}

pub struct Llama2 {
    model_type: LlamaModelType,
    attn: BatchedMultiqueryAttention,
    rms_norm: RmsNorm,
    rope: RoPE,
    silu: Silu,
    matmul: GemvQuant,
    soft_max: SoftMax,
    add_assign: OpAssign,
    copy: OpAssign,
}

impl Llama2 {
    pub fn new(device: &Device, model_type: LlamaModelType) -> Result<Self, ComposerError> {
        Ok(Self {
            model_type,
            attn: BatchedMultiqueryAttention::from_device(device)?,
            rms_norm: RmsNorm::from_device(device)?,
            rope: RoPE::from_device(device)?,
            silu: Silu::from_device(device)?,
            matmul: GemvQuant::from_device(device)?,
            soft_max: SoftMax::from_device(device)?,
            add_assign: OpAssign::new(device, OpAssignVariant::Add)?,
            copy: OpAssign::new(device, OpAssignVariant::Copy)?,
        })
    }

    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        state: &Llama2State,
        weights: &Llama2Weights,
        config: &Llama2Config,
        attn_params: &BatchedMultiqueryAttentionParams,
        pos: u32,
    ) {
        for l in 0..config.n_layers {
            let wl = &weights.layers[l];
            self.rms_norm
                .queue(queue, &state.xb, &state.x, &wl.attn_norm);

            let k_cache = state.key_cache[l].column(pos);
            let v_cache = state.value_cache[l].column(pos);

            self.matmul.queue(queue, &state.q, &wl.attn_q, &state.xb);
            self.matmul.queue(queue, k_cache, &wl.attn_k, &state.xb);
            self.matmul.queue(queue, v_cache, &wl.attn_v, &state.xb);

            if let Some(q_bias) = &wl.attn_q_bias {
                self.add_assign.queue(queue, &state.q, q_bias);
            }
            if let Some(k_bias) = &wl.attn_k_bias {
                self.add_assign.queue(queue, k_cache, k_bias);
            }
            if let Some(v_bias) = &wl.attn_v_bias {
                self.add_assign.queue(queue, v_cache, v_bias);
            }

            let rope_variant = self.model_type.rope_variant();
            self.rope
                .queue(queue, rope_variant, &state.rope_shape, &state.q, k_cache);

            // Start attention.
            self.queue_attn(queue, state, weights, config, l, attn_params, pos);

            self.matmul
                .queue(queue, &state.xb2, &wl.attn_output, &state.xb);
            // End attention.

            self.add_assign.queue(queue, &state.x, &state.xb2);
            self.rms_norm
                .queue(queue, &state.xb, &state.x, &wl.ffn_norm);

            // Start ffn_silu
            self.matmul.queue(queue, &state.hb, &wl.ffn_gate, &state.xb);
            self.matmul.queue(queue, &state.hb2, &wl.ffn_up, &state.xb);
            self.silu.queue(queue, &state.hb, &state.hb2);
            self.matmul
                .queue(queue, &state.xb2, &wl.ffn_down, &state.hb);
            // End ffn_silu

            self.add_assign.queue(queue, &state.x, &state.xb2);
        }

        self.rms_norm
            .queue(queue, &state.xb, &state.x, &weights.output_norm);

        self.matmul
            .queue(queue, &state.logits, &weights.output, &state.xb);
    }

    fn queue_attn<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        state: &Llama2State,
        weights: &Llama2Weights,
        config: &Llama2Config,
        layer: usize,
        attn_params: &BatchedMultiqueryAttentionParams,
        pos: u32,
    ) {
        // BatchedMultiqueryAttention::queue_multi(
        //     queue,
        //     &self.matmul.gemv_f32,
        //     &self.soft_max,
        //     &attn_params,
        //     &state.q,
        //     &state.key_cache[layer],
        //     &state.value_cache[layer],
        //     &state.att,
        //     &state.xb,
        // );

        self.attn.queue(
            queue,
            config.n_q_heads as u32,
            &state.attn_params,
            &state.q,
            &state.key_cache[layer],
            &state.value_cache[layer],
            &state.att,
            &state.xb,
        );
    }
}

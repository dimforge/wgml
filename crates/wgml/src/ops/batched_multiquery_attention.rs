use crate::models::llama2::cpu::softmax;
use crate::ops::SoftMax;
use bytemuck::Pod;
use nalgebra::{DMatrix, DVector};
use wgcore::kernel::KernelDispatch;
use wgcore::shapes::ViewShapeBuffers;
use wgcore::tensor::{ColumnMajor, GpuMatrix, GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgebra::{Gemv, GemvVariant};
use wgpu::{ComputePass, ComputePipeline, Device, Queue};

#[derive(Shader)]
#[shader(
    derive(Shape),
    src = "batched_multiquery_attention.wgsl",
    composable = false
)]
/// Shader implementing batched multi-query attention.
pub struct BatchedMultiqueryAttention {
    /// The compute pipeline representing the batched multi-query attention.
    pub main: ComputePipeline,
    pub mult_mask_attn: ComputePipeline,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Eq, Debug)]
/// Parameters needed to run the [`BatchedMultiqueryAttention`] kernel. Matches the layout of the
/// corresponding WGSL struct.
pub struct BatchedMultiqueryAttentionParams {
    pub seq_len: u32,
    pub kv_dim: u32,
    pub kv_mul: u32,
    pub n_heads: u32,
    pub head_size: u32,
    pub pos: u32,
}

impl BatchedMultiqueryAttention {
    fn dispatch_mult_mask<T: Pod>(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        params: &BatchedMultiqueryAttentionParams,
        params_gpu: &GpuScalar<BatchedMultiqueryAttentionParams>,
        attn: &GpuMatrix<T>,
    ) {
        let rounded_pos = (params.pos + 1).div_ceil(4) * 4;

        KernelDispatch::new(device, pass, &self.mult_mask_attn)
            .bind_at(0, [(params_gpu.buffer(), 0), (attn.buffer(), 4)])
            .dispatch((params.n_heads * rounded_pos).div_ceil(64));
    }

    pub fn dispatch_multi<'a, T: Pod>(
        &'a self,
        device: &Device,
        shapes: &ViewShapeBuffers,
        gpu_queue: &Queue,
        pass: &mut ComputePass,
        gemv: &'a Gemv,
        softmax: &'a SoftMax,
        params: &BatchedMultiqueryAttentionParams,
        params_gpu: &GpuScalar<BatchedMultiqueryAttentionParams>,
        q: &GpuVector<T>,
        key_cache: &GpuMatrix<T>,
        value_cache: &GpuMatrix<T>,
        attn: &GpuMatrix<T>,
        xb: &GpuVector<T>,
    ) {
        let n_q_heads = params.n_heads;
        let n_kv_heads = n_q_heads / params.kv_mul;
        // Pos rounded to a multiple of 4 to match the matmul element alignment.
        let rounded_pos = (params.pos + 1).div_ceil(4) * 4;
        // [head_size, pos + 1, n_kv_heads] -> [128, ..., 2] -> (transposed for gemv_tr: ) [..., 128, 2]
        let k = key_cache.reshape::<ColumnMajor, 3>(
            [params.head_size, rounded_pos, n_kv_heads],
            Some(params.head_size * n_kv_heads),
            Some(params.head_size),
        );
        // [head_size, kv_mul, n_kv_heads] -> [128, 6, 2]
        let q =
            q.reshape::<ColumnMajor, 3>([params.head_size, params.kv_mul, n_kv_heads], None, None);
        // [pos + 1, kv_mul, n_kv_heads] -> [..., 6, 2]
        let att = attn.reshape([rounded_pos, params.kv_mul, n_kv_heads], None, None);
        // [pos + 1, n_q_heads, 1] -> [..., 12, 1]
        let att_softmax = attn
            .reshape([params.pos + 1, n_q_heads, 1], Some(rounded_pos), None)
            .matrix(0);
        // [head_size, pos + 1, n_kv_heads] -> [128, ..., 2]
        let v = value_cache.reshape::<ColumnMajor, 3>(
            [params.head_size, rounded_pos, n_kv_heads],
            Some(params.head_size * n_kv_heads),
            Some(params.head_size),
        );
        // [head_size, kv_mul, n_kv_heads] -> [128, 6, 2]
        let xb =
            xb.reshape::<ColumnMajor, 3>([params.head_size, params.kv_mul, n_kv_heads], None, None);

        // gemv.queue_tr(queue, att, k, q);
        // PERF: because we are taking a shapes depending on `pos` we will be
        //       creating a new Buffer for the shape at each forward.
        //       The shape cache should have a mechanism for updating some existing
        //       buffers in-place? Or switch to a LRU cache?

        shapes.put_tmp(device, gpu_queue, k.shape());
        shapes.put_tmp(device, gpu_queue, att.shape());
        shapes.put_tmp(device, gpu_queue, att_softmax.shape());
        shapes.put_tmp(device, gpu_queue, v.shape());

        gemv.dispatch_generic(device, shapes, pass, att, k, q, GemvVariant::GemvTrFast);
        self.dispatch_mult_mask(device, pass, params, params_gpu, attn);
        softmax.dispatch(device, shapes, pass, att_softmax);
        gemv.dispatch(device, shapes, pass, xb, v, att);
    }

    pub fn dispatch<T: Pod>(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        n_heads: u32,
        params: &GpuScalar<BatchedMultiqueryAttentionParams>,
        q: &GpuVector<T>,
        key_cache: &GpuMatrix<T>,
        value_cache: &GpuMatrix<T>,
        attn: &GpuMatrix<T>,
        xb: &GpuVector<T>,
    ) {
        KernelDispatch::new(device, pass, &self.main)
            .bind0([
                params.buffer(),
                q.buffer(),
                key_cache.buffer(),
                value_cache.buffer(),
                attn.buffer(),
                xb.buffer(),
            ])
            .dispatch(n_heads.div_ceil(64));
    }

    pub fn run_cpu(
        params: &BatchedMultiqueryAttentionParams,
        q: &DVector<f32>,
        key_cache: &DMatrix<f32>,
        value_cache: &DMatrix<f32>,
        attn: &mut DMatrix<f32>,
        xb: &mut DVector<f32>,
    ) {
        // The number of embedding vector elements associated to each query head.
        let head_size = params.head_size as usize;
        // The number of query head associated to one key/value head.
        let kv_mul = params.kv_mul as usize;

        // Multihead attention. Iterate over all head.
        // TODO: in llama2.c, each head is iterated on in parallel.
        for h in 0..params.n_heads as usize {
            // Get the query vector for this head.
            let q = q.rows(h * head_size, head_size);
            // Attention scores for this head.
            let mut att = attn.column_mut(h);

            // Iterate over all timesteps (tokens in the sequence), including the current one, but
            // not past the current one due to causality.
            // See the KV cache explanation there: https://youtu.be/Mn_9W1nCFLo?si=3n4GH9f2OzMb5Np0&t=2940
            // -> This is iterating through all the green columns (from K^t) that are the rotated
            //    (by RoPE). The values set in this loop into the `att` variable here (attention
            //    scores) are the elements in the pink row (at the bottom of the QK^t matrix) divide
            //    by sqrt(params.head_size) (in other words, this is what’s given to softmax afterward.
            for t in 0..=params.pos as usize {
                // Get the key vector for this head and at this timestep.
                let k = key_cache.column(t); // TODO: does key_cache have the right dim?
                let k_head = k.rows((h / kv_mul) * head_size, head_size);

                // Calculate the attention score as the dot product of q and k.
                let mut score = q.dot(&k_head);
                score /= (head_size as f32).sqrt();
                // Save the score to the attention buffer.
                att[t] = score;
            }

            // Softmax the scores to get attention weights from 0..=pos inclusively.
            softmax(&mut att.rows_mut(0, params.pos as usize + 1));

            // Weighted sum of the values, store back into xb.
            // /!\ xb is now changing semantic, storing the weighted sums for all the heads.
            //       Now xb contains the "Attention 4" row from https://youtu.be/Mn_9W1nCFLo?si=550ar5aUg1I1k60l&t=2940.
            let mut xb = xb.rows_mut(h * head_size, head_size);
            xb.fill(0.0);
            for t in 0..=params.pos as usize {
                let v = value_cache.column(t);
                let v_head = v.rows((h / kv_mul) * head_size, head_size);
                xb.axpy(att[t], &v_head, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::ops::{BatchedMultiqueryAttentionParams, SoftMax};
    use nalgebra::{DMatrix, DVector};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;
    use wgcore::shapes::ViewShapeBuffers;
    use wgcore::tensor::{GpuMatrix, GpuScalar, GpuVector};
    use wgcore::Shader;
    use wgebra::Gemv;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_device(gpu.device()).unwrap();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        // let mut params = BatchedMultiqueryAttentionParams { seq_len: 131072, kv_dim: 256, kv_mul: 6, n_heads: 12, head_size: 128, pos: 9 };
        let params = BatchedMultiqueryAttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 6,
        };

        let q = DVector::new_random((params.n_heads * params.head_size) as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::zeros(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::zeros((params.n_heads * params.head_size) as usize);

        let gpu_params = GpuScalar::init(gpu.device(), params, BufferUsages::UNIFORM);
        let gpu_q = GpuVector::init(gpu.device(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = GpuMatrix::init(gpu.device(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = GpuMatrix::init(gpu.device(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = GpuMatrix::init(
            gpu.device(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = GpuVector::init(
            gpu.device(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = GpuVector::uninit(
            gpu.device(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = GpuMatrix::uninit(
            gpu.device(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut pass = encoder.compute_pass("test", None);
        batched_multihead_attention.dispatch(
            gpu.device(),
            &mut pass,
            params.n_heads,
            &gpu_params,
            &gpu_q,
            &gpu_key_cache,
            &gpu_value_cache,
            &gpu_attn,
            &gpu_xb,
        );
        drop(pass);

        gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
        gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

        gpu.queue().submit(Some(encoder.finish()));

        super::BatchedMultiqueryAttention::run_cpu(
            &params,
            &q,
            &key_cache,
            &value_cache,
            &mut attn,
            &mut xb,
        );

        approx::assert_relative_eq!(
            DVector::from(gpu_staging_xb.read(gpu.device()).await.unwrap()),
            xb,
            epsilon = 1.0e-5
        );

        approx::assert_relative_eq!(
            DMatrix::from_vec(
                attn.nrows(),
                attn.ncols(),
                gpu_staging_attn.read(gpu.device()).await.unwrap()
            ),
            attn,
            epsilon = 1.0e-5
        );
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention_multi() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_device(gpu.device()).unwrap();
        let shapes = ViewShapeBuffers::new();
        let matmul = Gemv::from_device(gpu.device()).unwrap();
        let softmax = SoftMax::from_device(gpu.device()).unwrap();

        // let mut params = BatchedMultiqueryAttentionParams { seq_len: 131072, kv_dim: 256, kv_mul: 6, n_heads: 12, head_size: 128, pos: 0 };
        let mut params = BatchedMultiqueryAttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 0,
        };

        let q = DVector::new_random((params.n_heads * params.head_size) as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::zeros(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::zeros((params.n_heads * params.head_size) as usize);

        let gpu_q = GpuVector::init(gpu.device(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = GpuMatrix::init(gpu.device(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = GpuMatrix::init(gpu.device(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = GpuMatrix::init(
            gpu.device(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = GpuVector::init(
            gpu.device(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = GpuVector::uninit(
            gpu.device(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = GpuMatrix::uninit(
            gpu.device(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        for pos in 0..9 {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            params.pos = pos;

            let gpu_params = GpuScalar::init(gpu.device(), params, BufferUsages::UNIFORM);

            let mut pass = encoder.compute_pass("test", None);
            batched_multihead_attention.dispatch_multi(
                gpu.device(),
                &shapes,
                gpu.queue(),
                &mut pass,
                &matmul,
                &softmax,
                &params,
                &gpu_params,
                &gpu_q,
                &gpu_key_cache,
                &gpu_value_cache,
                &gpu_attn,
                &gpu_xb,
            );
            drop(pass);

            gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
            gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

            gpu.queue().submit(Some(encoder.finish()));

            super::BatchedMultiqueryAttention::run_cpu(
                &params,
                &q,
                &key_cache,
                &value_cache,
                &mut attn,
                &mut xb,
            );

            // NOTE: we can’t compare attn since they don’t have the same layout.
            // approx::assert_relative_eq!(
            //     DMatrix::from_vec(
            //         attn.nrows(),
            //         attn.ncols(),
            //         gpu_staging_attn.read(gpu.device()).await.unwrap()
            //     ),
            //     attn,
            //     epsilon = 1.0e-5
            // );

            approx::assert_relative_eq!(
                DVector::from(gpu_staging_xb.read(gpu.device()).await.unwrap()),
                xb,
                epsilon = 1.0e-5
            );
        }
    }
}

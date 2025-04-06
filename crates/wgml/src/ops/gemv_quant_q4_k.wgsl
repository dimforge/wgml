#import wgblas::shape as Shape
#import wgml::quantization as Quant


@group(0) @binding(0)
var<uniform> shape_out: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_m: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape_v: Shape::Shape;
@group(0) @binding(3)
var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> m: array<Quant::BlockQ4K>;
@group(0) @binding(5)
var<storage, read> v: array<vec4<f32>>;

const WORKGROUP_SIZE: u32 = 32;

var<workgroup> sketch: array<vec4<f32>, WORKGROUP_SIZE>;

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

// TODO: needs a lot of optimizations.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let j_ref = Shape::iv(shape_v, 0u);
    let m_ref = Shape::im(shape_m, 0u, 0u);
    var sum = vec4<f32>();

    for (var j = 0u; j < shape_m.ncols; j++) {
        let quant0 = Shape::im(shape_m, workgroup_id.x * 4u + 0, j);
        let quant1 = Shape::im(shape_m, workgroup_id.x * 4u + 1, j);
        let quant2 = Shape::im(shape_m, workgroup_id.x * 4u + 2, j);
        let quant3 = Shape::im(shape_m, workgroup_id.x * 4u + 3, j);

        let parts0 = dequantize_q4_k_workgroup(quant0, local_id.x);
        let parts1 = dequantize_q4_k_workgroup(quant1, local_id.x);
        let parts2 = dequantize_q4_k_workgroup(quant2, local_id.x);
        let parts3 = dequantize_q4_k_workgroup(quant3, local_id.x);

        let j_base = j_ref + j * 64u;
        let jj = local_id.x & 0xfffffff8u; // == (local_id.x / 8u) * 8u;
        let vj_a = v[j_base + local_id.x + jj];
        let vj_b = v[j_base + local_id.x + jj + 8u];

        let mat_a = transpose(mat4x4(parts0[0], parts1[0], parts2[0], parts3[0]));
        sum += mat_a * vj_a;
        let mat_b = transpose(mat4x4(parts0[1], parts1[1], parts2[1], parts3[1]));
        sum += mat_b * vj_b;
    }

    sketch[local_id.x] = sum;

    workgroupBarrier();

//    reduce_sum(local_id.x, 32u);
    reduce_sum(local_id.x, 16u);
    reduce_sum(local_id.x, 8u);
    reduce_sum(local_id.x, 4u);
    reduce_sum(local_id.x, 2u);
    reduce_sum(local_id.x, 1u);

    if local_id.x == 0u {
        let i_out = Shape::iv(shape_out, workgroup_id.x);
        out[i_out] = sketch[0];
    }
}



// #################
// # Dequantization code copied from `quantization.wgsl`, modified to take a
// # pointer as the block argument.
// ################
fn dequantize_q4_k_workgroup(block_id: u32, k: u32) -> array<vec4<f32>, 2> {
    let block = &m[block_id];
    let d_dmin = unpack2x16float((*block).d_dmin);
    let d = d_dmin.x;
    let min = d_dmin.y;

    // 32 threads workgroups
    let j = k / 8u;
    let is = j * 2u;
    let qj_prev1 = (*block).scales[max(is / 4, 1u) - 1];
    let qj1 = (*block).scales[is / 4];
    let qj_next1 = (*block).scales[is / 4 + 1];
    let sc_m1 = Quant::get_scale_min_k4(is, qj_prev1, qj1, qj_next1);
    let d1 = d * f32(sc_m1.x);
    let m1 = min * f32(sc_m1.y);

    let qj_prev2 = (*block).scales[max((is + 1) / 4, 1u) - 1];
    let qj2 = (*block).scales[(is + 1) / 4];
    let qj_next2 = (*block).scales[(is + 1) / 4 + 1];
    let sc_m2 = Quant::get_scale_min_k4(is + 1, qj_prev2, qj2, qj_next2);
    let d2 = d * f32(sc_m2.x);
    let m2 = min * f32(sc_m2.y);

    // NOTE: from the ref implementation:
    //    iq     == (k / 8u) * 8u
    // => iq + l == (k / 8u) * 8u + k % 8u == k
    let qs = (*block).qs[k];

    // NOTE: from the ref implementation:
    //    j * 16  == (k / 8u) * 16u
    //    j * 16 + [0..7u] == (k / 8u) * 8u * 2u + k % 8u
    //                     == (k / 8u) * 8u + k % 8u + (k / 8u) * 8u
    //                     == k + (k / 8u) * 8u
    //                     == k + j * 8u
    let res_a = vec4(
       f32(qs & 0xF),
       f32((qs >> 8) & 0xF),
       f32((qs >> 16) & 0xF),
       f32((qs >> 24) & 0xF),
   ) * d1 - m1;
   let res_b = vec4(
       f32((qs >> 4) & 0xF),
       f32((qs >> 12) & 0xF),
       f32((qs >> 20) & 0xF),
       f32((qs >> 28) & 0xF),
   ) * d2 - m2;

    return array(res_a, res_b);
}
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
var<storage, read> m: array<Quant::BlockQ6Kx2>;
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

        let parts0 = dequantize_q6_kx2_workgroup(quant0, local_id.x);
        let parts1 = dequantize_q6_kx2_workgroup(quant1, local_id.x);
        let parts2 = dequantize_q6_kx2_workgroup(quant2, local_id.x);
        let parts3 = dequantize_q6_kx2_workgroup(quant3, local_id.x);

        let j_base = j_ref + j * 128u;
        let jj = (local_id.x / 16u) * 64u + ((local_id.x / 8u) % 2u) * 32u + (local_id.x % 8u);
        let vj_a = v[j_base + jj];
        let vj_b = v[j_base + jj + 8u];
        let vj_c = v[j_base + jj + 16u];
        let vj_d = v[j_base + jj + 24u];

        let mat_a = transpose(mat4x4(parts0[0], parts1[0], parts2[0], parts3[0]));
        sum += mat_a * vj_a;
        let mat_b = transpose(mat4x4(parts0[1], parts1[1], parts2[1], parts3[1]));
        sum += mat_b * vj_b;
        let mat_c = transpose(mat4x4(parts0[2], parts1[2], parts2[2], parts3[2]));
        sum += mat_c * vj_c;
        let mat_d = transpose(mat4x4(parts0[3], parts1[3], parts2[3], parts3[3]));
        sum += mat_d * vj_d;
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
fn dequantize_q6_kx2_workgroup(block_id: u32, k: u32) -> array<vec4<f32>, 4> {
    const _0xF = vec4(0xFu);
    const _6 = vec4(6u);
    const _4 = vec4(4u);
    const _3 = vec4(3u);
    const _2 = vec4(2u);
    const _32 = vec4(32);

    let block = &m[block_id];

    if k / 16u == 0u {
        // Block A
        // Its data goes from data[0] to half of data[52]
        // It is mostly well aligned with the original BlockQ6K except for the original value.
        let d_a = unpack2x16float((*block).data[52]).x;

        const QL0: u32 = 0u;
        const QH0: u32 = 32u;
        const SC0: u32 = 48u;

        let i = (k / 8u) % 2u;
        let data0 = vec4<f32>(unpack4xI8((*block).data[SC0 + i * 2u])) * d_a;
        let data1 = vec4<f32>(unpack4xI8((*block).data[SC0 + i * 2u + 1])) * d_a;

        let l = k % 8u;
        let is = l / 4u; // NOTE: is is either 0 or 1

        let is_shift = is * 8u;
        let is_shift2 = (is + 2u) * 8u;
        let qh = unpack4xU8((*block).data[l + QH0 + i * 8u]);
        let ql0 = unpack4xU8((*block).data[l + QL0 + i * 16u]);
        let ql32 = unpack4xU8((*block).data[l + QL0 + i * 16u + 8u]);

        let q1 = vec4<i32>((ql0 & _0xF) | ((qh & _3) << _4)) - _32;
        let q2 = vec4<i32>((ql32 & _0xF) | (((qh >> _2) & _3) << _4)) - _32;
        let q3 = vec4<i32>((ql0 >> _4) | (((qh >> _4) & _3) << _4)) - _32;
        let q4 = vec4<i32>((ql32 >> _4) | (((qh >> _6) & _3) << _4)) - _32;

        return array(
            data0[is] * vec4<f32>(q1),
            data0[is + 2] * vec4<f32>(q2),
            data1[is] * vec4<f32>(q3),
            data1[is + 2] * vec4<f32>(q4),
        );
    } else {
        // Block B
        // Its data goes from half of data[52] to data[104].
        // All values are starting with the u16 leftmost bits of the previous index.
        // So this is a copy-past of BlockB butwith the `-1u` and `>> 16`, `<< 16`
        // schenanigans to reconstruct the properly aligned u32 values.
        let d_b = unpack2x16float((*block).data[104]).y;

        const QL0: u32 = 53u + 0u;
        const QH0: u32 = 53u + 32u;
        const SC0: u32 = 53u + 48u;

        let i = (k / 8u) % 2u;
        let l = k % 8u;
        let isc0 = SC0 + i * 2u;
        let isc1 = SC0 + i * 2u + 1;
        let data0 = vec4<f32>(unpack4xI8(((*block).data[isc0 - 1u] >> 16) | ((*block).data[isc0] << 16))) * d_b;
        let data1 = vec4<f32>(unpack4xI8(((*block).data[isc1 - 1u] >> 16) | ((*block).data[isc1] << 16))) * d_b;

        // NOTE: the `* 4u` in these consts is needed because we divide by 4 further down.
        let is = l / 4u; // NOTE: is is either 0 or 1

        let is_shift = is * 8u;
        let is_shift2 = (is + 2u) * 8u;

        let iqh = l + QH0 + i * 8u;
        let iql0 = l + QL0 + i * 16u;
        let iql32 = l + QL0 + i * 16u + 8u;

        let qh = unpack4xU8(((*block).data[iqh - 1u] >> 16) | ((*block).data[iqh] << 16));
        let ql0 = unpack4xU8(((*block).data[iql0 - 1u] >> 16) | ((*block).data[iql0] << 16));
        let ql32 = unpack4xU8(((*block).data[iql32 - 1u] >> 16) | ((*block).data[iql32] << 16));

        let q1 = vec4<i32>((ql0 & _0xF) | ((qh & _3) << _4)) - _32;
        let q2 = vec4<i32>((ql32 & _0xF) | (((qh >> _2) & _3) << _4)) - _32;
        let q3 = vec4<i32>((ql0 >> _4) | (((qh >> _4) & _3) << _4)) - _32;
        let q4 = vec4<i32>((ql32 >> _4) | (((qh >> _6) & _3) << _4)) - _32;

        return array(
            data0[is] * vec4<f32>(q1),
            data0[is + 2] * vec4<f32>(q2),
            data1[is] * vec4<f32>(q3),
            data1[is + 2] * vec4<f32>(q4),
        );
    }
}
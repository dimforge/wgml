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
var<storage, read> m: array<Quant::BlockQ4_0x2>;
@group(0) @binding(5)
var<storage, read> v: array<vec4<f32>>;

//// NOTE: this assumes a rhs size containing this multiple of f32s:
////       64 * 4 = 256
//const WORKGROUP_SIZE: u32 = 32u;
//const COLS_STEP: u32 = 4u;

// NOTE: this assumes a rhs size containing this multiple of f32s:
//       128 * 4 = 512
const WORKGROUP_SIZE: u32 = 64u;
const COLS_STEP: u32 = 8u;

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
    var sum = vec4<f32>();

    for (var j = 0u; j < shape_m.ncols / COLS_STEP; j++) {
        let quant0 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 0, j * COLS_STEP + local_id.x / 8u)];
        let quant1 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 1, j * COLS_STEP + local_id.x / 8u)];
        let quant2 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 2, j * COLS_STEP + local_id.x / 8u)];
        let quant3 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 3, j * COLS_STEP + local_id.x / 8u)];

        let j_base = j_ref + j * (16u * COLS_STEP);
        let vj_a = v[j_base + local_id.x + (local_id.x / 4u) * 4u];
        let vj_b = v[j_base + local_id.x + (local_id.x / 4u) * 4u + 4u];

        if (local_id.x / 4u) % 2u == 0 {
            // Dequantizing block 1
            let lid = local_id.x % 4u;

            let scale0 = unpack2x16float((*quant0).data[0]).x;
            let data0 = (*quant0).data[lid] >> 16 | (*quant0).data[lid + 1u] << 16;
            let parts0 = Quant::dequantize_q4_0x2_part(data0, scale0);
            let scale1 = unpack2x16float((*quant1).data[0]).x;
            let data1 = (*quant1).data[lid] >> 16 | (*quant1).data[lid + 1u] << 16;
            let parts1 = Quant::dequantize_q4_0x2_part(data1, scale1);
            let scale2 = unpack2x16float((*quant2).data[0]).x;
            let data2 = (*quant2).data[lid] >> 16 | (*quant2).data[lid + 1u] << 16;
            let parts2 = Quant::dequantize_q4_0x2_part(data2, scale2);
            let scale3 = unpack2x16float((*quant3).data[0]).x;
            let data3 = (*quant3).data[lid] >> 16 | (*quant3).data[lid + 1u] << 16;
            let parts3 = Quant::dequantize_q4_0x2_part(data3, scale3);

            let mat_a = transpose(mat4x4(parts0[0], parts1[0], parts2[0], parts3[0]));
            sum += mat_a * vj_a;
            let mat_b = transpose(mat4x4(parts0[1], parts1[1], parts2[1], parts3[1]));
            sum += mat_b * vj_b;
        } else {
            // Dequantizing block 2
            let lid = local_id.x % 4u;
            let scale0 = unpack2x16float((*quant0).data[4]).y;
            let data0 = (*quant0).data[lid + 5u];
            let parts0 = Quant::dequantize_q4_0x2_part(data0, scale0);
            let scale1 = unpack2x16float((*quant1).data[4]).y;
            let data1 = (*quant1).data[lid + 5u];
            let parts1 = Quant::dequantize_q4_0x2_part(data1, scale1);
            let scale2 = unpack2x16float((*quant2).data[4]).y;
            let data2 = (*quant2).data[lid + 5u];
            let parts2 = Quant::dequantize_q4_0x2_part(data2, scale2);
            let scale3 = unpack2x16float((*quant3).data[4]).y;
            let data3 = (*quant3).data[lid + 5u];
            let parts3 = Quant::dequantize_q4_0x2_part(data3, scale3);

            let mat_a = transpose(mat4x4(parts0[0], parts1[0], parts2[0], parts3[0]));
            sum += mat_a * vj_a;
            let mat_b = transpose(mat4x4(parts0[1], parts1[1], parts2[1], parts3[1]));
            sum += mat_b * vj_b;
        }
    }

    sketch[local_id.x] = sum;

    workgroupBarrier();

    if WORKGROUP_SIZE >= 64u {
        reduce_sum(local_id.x, 32u);
    }
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

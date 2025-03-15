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
var<storage, read> m: array<Quant::BlockQ8_0x2>;
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
    var sum = vec4<f32>();

    for (var j = 0u; j < shape_m.ncols / 2u; j++) {
        let quant0 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 0, j * 2u + local_id.x / 16u)];
        let quant1 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 1, j * 2u + local_id.x / 16u)];
        let quant2 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 2, j * 2u + local_id.x / 16u)];
        let quant3 = &m[Shape::im(shape_m, workgroup_id.x * 4u + 3, j * 2u + local_id.x / 16u)];

        let j_base = j_ref + j * 32u;
        let vj = v[j_base + local_id.x];

        if (local_id.x / 8u) % 2u == 0 {
            // Dequantizing block 1
            let lid = local_id.x % 8u;
            let scale0 = unpack2x16float((*quant0).data[0]).x;
            let data0 = unpack4xI8((*quant0).data[lid] >> 16 | (*quant0).data[lid + 1u] << 16);
            let scale1 = unpack2x16float((*quant1).data[0]).x;
            let data1 = unpack4xI8((*quant1).data[lid] >> 16 | (*quant1).data[lid + 1u] << 16);
            let scale2 = unpack2x16float((*quant2).data[0]).x;
            let data2 = unpack4xI8((*quant2).data[lid] >> 16 | (*quant2).data[lid + 1u] << 16);
            let scale3 = unpack2x16float((*quant3).data[0]).x;
            let data3 = unpack4xI8((*quant3).data[lid] >> 16 | (*quant3).data[lid + 1u] << 16);

            let mat = transpose(mat4x4(
                vec4<f32>(data0) * scale0,
                vec4<f32>(data1) * scale1,
                vec4<f32>(data2) * scale2,
                vec4<f32>(data3) * scale3,
            ));

            sum += mat * vj;
        } else {
            // Dequantizing block 2
            let lid = local_id.x % 8u;
            let scale0 = unpack2x16float((*quant0).data[8]).y;
            let data0 = unpack4xI8((*quant0).data[lid + 9u]);
            let scale1 = unpack2x16float((*quant1).data[8]).y;
            let data1 = unpack4xI8((*quant1).data[lid + 9u]);
            let scale2 = unpack2x16float((*quant2).data[8]).y;
            let data2 = unpack4xI8((*quant2).data[lid + 9u]);
            let scale3 = unpack2x16float((*quant3).data[8]).y;
            let data3 = unpack4xI8((*quant3).data[lid + 9u]);

            let mat = transpose(mat4x4(
                vec4<f32>(data0) * scale0,
                vec4<f32>(data1) * scale1,
                vec4<f32>(data2) * scale2,
                vec4<f32>(data3) * scale3,
            ));

            sum += mat * vj;
        }
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

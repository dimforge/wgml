#import wgblas::shape as Shape
#import wgml::quantization as Quant


@group(0) @binding(0)
var<uniform> shape_out: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_m: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape_v: Shape::Shape;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;
@group(0) @binding(4)
var<storage, read> m: array<Quant::BlockQ8_0x2>;
@group(0) @binding(5)
var<storage, read> v: array<vec4<f32>>;

const WORKGROUP_SIZE: u32 = 64;

// TODO: needs a lot of optimizations.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x < shape_m.nrows {
        let i_ref = Shape::iv(shape_v, 0u);
        let i_out = Shape::iv(shape_out, invocation_id.x);
        var sum = 0.0;

        for (var j = 0u; j < shape_m.ncols; j++) {
            let quant = m[Shape::im(shape_m, invocation_id.x, j)];
            let dequant = Quant::dequantize_q8_0x2(quant);

            // Unroll calculation with all block elements.
            let i_base = i_ref + j * 16u;
            sum += dot(dequant[0], v[i_base + 0]) +
                dot(dequant[1], v[i_base + 1]) +
                dot(dequant[2], v[i_base + 2]) +
                dot(dequant[3], v[i_base + 3]) +
                dot(dequant[4], v[i_base + 4]) +
                dot(dequant[5], v[i_base + 5]) +
                dot(dequant[6], v[i_base + 6]) +
                dot(dequant[7], v[i_base + 7]) +
                dot(dequant[8], v[i_base + 8]) +
                dot(dequant[9], v[i_base + 9]) +
                dot(dequant[10], v[i_base + 10]) +
                dot(dequant[11], v[i_base + 11]) +
                dot(dequant[12], v[i_base + 12]) +
                dot(dequant[13], v[i_base + 13]) +
                dot(dequant[14], v[i_base + 14]) +
                dot(dequant[15], v[i_base + 15]);
        }

        out[i_out] = sum;
    }
}
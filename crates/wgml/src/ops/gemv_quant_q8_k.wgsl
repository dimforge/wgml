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
var<storage, read> m: array<Quant::BlockQ8K>;
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
            let dequant = Quant::dequantize_q8_k(quant);

            // Unroll calculation with all block elements.
            let i_base = i_ref + j * 64u;

            for (var k = 0u; k < 64u; k += 16u) {
                sum += dot(dequant[k + 0u], v[k + i_base + 0u]) +
                    dot(dequant[k + 1u], v[k + i_base + 1u]) +
                    dot(dequant[k + 2u], v[k + i_base + 2u]) +
                    dot(dequant[k + 3u], v[k + i_base + 3u]) +
                    dot(dequant[k + 4u], v[k + i_base + 4u]) +
                    dot(dequant[k + 5u], v[k + i_base + 5u]) +
                    dot(dequant[k + 6u], v[k + i_base + 6u]) +
                    dot(dequant[k + 7u], v[k + i_base + 7u]) +
                    dot(dequant[k + 8u], v[k + i_base + 8u]) +
                    dot(dequant[k + 9u], v[k + i_base + 9u]) +
                    dot(dequant[k + 10u], v[k + i_base + 10u]) +
                    dot(dequant[k + 11u], v[k + i_base + 11u]) +
                    dot(dequant[k + 12u], v[k + i_base + 12u]) +
                    dot(dequant[k + 13u], v[k + i_base + 13u]) +
                    dot(dequant[k + 14u], v[k + i_base + 14u]) +
                    dot(dequant[k + 15u], v[k + i_base + 15u]);
            }
        }

        out[i_out] = sum;
    }
}
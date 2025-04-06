#import wgml::quantization as Quant

@group(0) @binding(0)
var<storage, read> in_q8: array<Quant::BlockQ8_0x2>;
@group(0) @binding(1)
var<storage, read_write> out_q8: array<f32>;
@group(0) @binding(2)
var<storage, read> in_q4: array<Quant::BlockQ4_0x2>;
@group(0) @binding(3)
var<storage, read_write> out_q4: array<f32>;
@group(0) @binding(4)
var<storage, read> in_q4_1: array<Quant::BlockQ4_1x2>;
@group(0) @binding(5)
var<storage, read_write> out_q4_1: array<f32>;
@group(0) @binding(6)
var<storage, read> in_q5_0: array<Quant::BlockQ5_0x2>;
@group(0) @binding(7)
var<storage, read_write> out_q5_0: array<f32>;

@group(1) @binding(0)
var<storage, read> in_q5_1: array<Quant::BlockQ5_1x2>;
@group(1) @binding(1)
var<storage, read_write> out_q5_1: array<f32>;
@group(1) @binding(2)
var<storage, read> in_q8_k: array<Quant::BlockQ8K>;
@group(1) @binding(3)
var<storage, read_write> out_q8_k: array<f32>;
@group(1) @binding(4)
var<storage, read> in_q5_k: array<Quant::BlockQ5K>;
@group(1) @binding(5)
var<storage, read_write> out_q5_k: array<f32>;
@group(1) @binding(6)
var<storage, read> in_q4_k: array<Quant::BlockQ4K>;
@group(1) @binding(7)
var<storage, read_write> out_q4_k: array<f32>;

@group(2) @binding(0)
var<storage, read> in_q6_k: array<Quant::BlockQ6Kx2>;
@group(2) @binding(1)
var<storage, read_write> out_q6_k: array<f32>;



@compute @workgroup_size(1, 1, 1)
fn dequantize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;

    // Q8_0
    if i < arrayLength(&in_q8) {
        let dequant_q8 = Quant::dequantize_q8_0x2(in_q8[i]);
        for (var k = 0u; k < 16; k++) {
            out_q8[i * 64u + k * 4u] = dequant_q8[k].x;
            out_q8[i * 64u + k * 4u + 1u] = dequant_q8[k].y;
            out_q8[i * 64u + k * 4u + 2u] = dequant_q8[k].z;
            out_q8[i * 64u + k * 4u + 3u] = dequant_q8[k].w;
        }
    }

    // Q4_0
    if i < arrayLength(&in_q4) {
        let dequant_q4 = Quant::dequantize_q4_0x2(in_q4[i]);
        for (var k = 0u; k < 16; k++) {
            out_q4[i * 64u + k * 4u] = dequant_q4[k].x;
            out_q4[i * 64u + k * 4u + 1u] = dequant_q4[k].y;
            out_q4[i * 64u + k * 4u + 2u] = dequant_q4[k].z;
            out_q4[i * 64u + k * 4u + 3u] = dequant_q4[k].w;
        }
    }

    // Q4_1
    if i < arrayLength(&in_q4_1) {
        let dequant_q4_1 = Quant::dequantize_q4_1x2(in_q4_1[i]);
        for (var k = 0u; k < 16; k++) {
            out_q4_1[i * 64u + k * 4u] = dequant_q4_1[k].x;
            out_q4_1[i * 64u + k * 4u + 1u] = dequant_q4_1[k].y;
            out_q4_1[i * 64u + k * 4u + 2u] = dequant_q4_1[k].z;
            out_q4_1[i * 64u + k * 4u + 3u] = dequant_q4_1[k].w;
        }
    }

    // Q5_0
    if i < arrayLength(&in_q5_0) {
        let dequant_q5_0 = Quant::dequantize_q5_0x2(in_q5_0[i]);
        for (var k = 0u; k < 16; k++) {
            out_q5_0[i * 64u + k * 4u] = dequant_q5_0[k].x;
            out_q5_0[i * 64u + k * 4u + 1u] = dequant_q5_0[k].y;
            out_q5_0[i * 64u + k * 4u + 2u] = dequant_q5_0[k].z;
            out_q5_0[i * 64u + k * 4u + 3u] = dequant_q5_0[k].w;
        }
    }

    // Q5_1
    if i < arrayLength(&in_q5_1) {
        let dequant_q5_1 = Quant::dequantize_q5_1x2(in_q5_1[i]);
        for (var k = 0u; k < 16; k++) {
            out_q5_1[i * 64u + k * 4u] = dequant_q5_1[k].x;
            out_q5_1[i * 64u + k * 4u + 1u] = dequant_q5_1[k].y;
            out_q5_1[i * 64u + k * 4u + 2u] = dequant_q5_1[k].z;
            out_q5_1[i * 64u + k * 4u + 3u] = dequant_q5_1[k].w;
        }
    }

    // Q8_k
    if i < arrayLength(&in_q8_k) {
        let dequant_q8_k = Quant::dequantize_q8_k(in_q8_k[i]);
        for (var k = 0u; k < 64; k++) {
            out_q8_k[i * 256 + k * 4u] = dequant_q8_k[k].x;
            out_q8_k[i * 256 + k * 4u + 1u] = dequant_q8_k[k].y;
            out_q8_k[i * 256 + k * 4u + 2u] = dequant_q8_k[k].z;
            out_q8_k[i * 256 + k * 4u + 3u] = dequant_q8_k[k].w;
        }
    }

    // Q5_k
    if i < arrayLength(&in_q5_k) {
        let dequant_q5_k = Quant::dequantize_q5_k(in_q5_k[i]);
        for (var k = 0u; k < 64; k++) {
            out_q5_k[i * 256 + k * 4u] = dequant_q5_k[k].x;
            out_q5_k[i * 256 + k * 4u + 1u] = dequant_q5_k[k].y;
            out_q5_k[i * 256 + k * 4u + 2u] = dequant_q5_k[k].z;
            out_q5_k[i * 256 + k * 4u + 3u] = dequant_q5_k[k].w;
        }
    }

    // Q4_k
    if i < arrayLength(&in_q4_k) {
        let dequant_q4_k = Quant::dequantize_q4_k(in_q4_k[i]);
        for (var k = 0u; k < 64; k++) {
            out_q4_k[i * 256 + k * 4u] = dequant_q4_k[k].x;
            out_q4_k[i * 256 + k * 4u + 1u] = dequant_q4_k[k].y;
            out_q4_k[i * 256 + k * 4u + 2u] = dequant_q4_k[k].z;
            out_q4_k[i * 256 + k * 4u + 3u] = dequant_q4_k[k].w;
        }
    }

    // Q6_k
    if i < arrayLength(&in_q6_k) {
        let dequant_q6_k = Quant::dequantize_q6_kx2(in_q6_k[i]);
        for (var k = 0u; k < 128; k++) {
            out_q6_k[i * 512 + k * 4u] = dequant_q6_k[k].x;
            out_q6_k[i * 512 + k * 4u + 1u] = dequant_q6_k[k].y;
            out_q6_k[i * 512 + k * 4u + 2u] = dequant_q6_k[k].z;
            out_q6_k[i * 512 + k * 4u + 3u] = dequant_q6_k[k].w;
        }
    }
}
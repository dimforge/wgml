#define_import_path wgml::quantization

struct BlockF16x2 {
    data: u32, // [f16; 2]
}


fn dequantize_f16x2(q: BlockF16x2) -> vec2<f32> {
    return unpack2x16float(q.data);
}

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43-L46
// Contains two `BlockQ8_0(f16, [i8; 32])`.
struct BlockQ8_0x2 {
    data: array<u32, 17>
}


const ELEMENTS_PER_BLOCK_Q8_0x2: u32 = 64;

// See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1609
fn dequantize_q8_0x2(q: BlockQ8_0x2) -> array<vec4<f32>, 16> {
    let scale_a = unpack2x16float(q.data[0]).x;

    let data0_a = unpack4xI8(q.data[0] >> 16 | q.data[1] << 16);
    let data1_a = unpack4xI8(q.data[1] >> 16 | q.data[2] << 16);
    let data2_a = unpack4xI8(q.data[2] >> 16 | q.data[3] << 16);
    let data3_a = unpack4xI8(q.data[3] >> 16 | q.data[4] << 16);
    let data4_a = unpack4xI8(q.data[4] >> 16 | q.data[5] << 16);
    let data5_a = unpack4xI8(q.data[5] >> 16 | q.data[6] << 16);
    let data6_a = unpack4xI8(q.data[6] >> 16 | q.data[7] << 16);
    let data7_a = unpack4xI8(q.data[7] >> 16 | q.data[8] << 16);

    let scale_b = unpack2x16float(q.data[8]).y;
    let data0_b = unpack4xI8(q.data[9]);
    let data1_b = unpack4xI8(q.data[10]);
    let data2_b = unpack4xI8(q.data[11]);
    let data3_b = unpack4xI8(q.data[12]);
    let data4_b = unpack4xI8(q.data[13]);
    let data5_b = unpack4xI8(q.data[14]);
    let data6_b = unpack4xI8(q.data[15]);
    let data7_b = unpack4xI8(q.data[16]);

    return array(
        // First block.
        vec4<f32>(data0_a) * scale_a,
        vec4<f32>(data1_a) * scale_a,
        vec4<f32>(data2_a) * scale_a,
        vec4<f32>(data3_a) * scale_a,
        vec4<f32>(data4_a) * scale_a,
        vec4<f32>(data5_a) * scale_a,
        vec4<f32>(data6_a) * scale_a,
        vec4<f32>(data7_a) * scale_a,
        // Second block.
        vec4<f32>(data0_b) * scale_b,
        vec4<f32>(data1_b) * scale_b,
        vec4<f32>(data2_b) * scale_b,
        vec4<f32>(data3_b) * scale_b,
        vec4<f32>(data4_b) * scale_b,
        vec4<f32>(data5_b) * scale_b,
        vec4<f32>(data6_b) * scale_b,
        vec4<f32>(data7_b) * scale_b,
    );
}

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11-L14
// Contains two `BlockQ4_0(f16, [i8; 16])`.
struct BlockQ4_0x2 {
    data: array<u32, 9>,
}

fn dequantize_q4_0x2_part(data: u32, scale: f32) -> array<vec4<f32>, 2> {
    let x0 = i32((data >> (0u * 8u)) & 0x0F) - 8;
    let x1 = i32((data >> (0u * 8u + 4u)) & 0x0F) - 8;
    let x2 = i32((data >> (1u * 8u)) & 0x0F) - 8;
    let x3 = i32((data >> (1u * 8u + 4u)) & 0x0F) - 8;
    let x4 = i32((data >> (2u * 8u)) & 0x0F) - 8;
    let x5 = i32((data >> (2u * 8u + 4u)) & 0x0F) - 8;
    let x6 = i32((data >> (3u * 8u)) & 0x0F) - 8;
    let x7 = i32((data >> (3u * 8u + 4u)) & 0x0F) - 8;
    return array(
        vec4(f32(x0), f32(x2), f32(x4), f32(x6)) * scale,
        vec4(f32(x1), f32(x3), f32(x5), f32(x7)) * scale
    );
}

// See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1515
fn dequantize_q4_0x2(q: BlockQ4_0x2) -> array<vec4<f32>, 16> {
    let zero = vec4(0.0);
    var result = array<vec4<f32>, 16>();

    // First block.
    let scale_a = unpack2x16float(q.data[0]).x;
    for (var k = 0; k < 4; k++) {
        let data = q.data[k] >> 16u | q.data[k + 1] << 16u;
        let parts = dequantize_q4_0x2_part(data, scale_a);
        result[k] = parts[0];
        result[4 + k] = parts[1];
    }

    // Second block.
    let scale_b = unpack2x16float(q.data[4]).y;
    for (var k = 0; k < 4; k++) {
        let data = q.data[k + 5];
        let parts = dequantize_q4_0x2_part(data, scale_b);
        result[8 + k] = parts[0];
        result[12 + k] = parts[1];
    }

    return result;
}

// FIXME: we don’t really need this to be x2.
//        Just one is already properly aligned.
// Contains two `BlockQ4_1(f16, u16, [i8; 16])`.
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L18-L22
struct BlockQ4_1x2 {
    data: array<u32, 10>,
}

fn dequantize_q4_1x2_part(data: u32, scale_mid: vec2<f32>) -> array<vec4<f32>, 2> {
    let x0 = (data >> (0u * 8u)) & 0x0F;
    let x1 = (data >> (0u * 8u + 4u)) & 0x0F;
    let x2 = (data >> (1u * 8u)) & 0x0F;
    let x3 = (data >> (1u * 8u + 4u)) & 0x0F;
    let x4 = (data >> (2u * 8u)) & 0x0F;
    let x5 = (data >> (2u * 8u + 4u)) & 0x0F;
    let x6 = (data >> (3u * 8u)) & 0x0F;
    let x7 = (data >> (3u * 8u + 4u)) & 0x0F;
    return array(
        vec4(f32(x0), f32(x2), f32(x4), f32(x6)) * scale_mid.x + scale_mid.y,
        vec4(f32(x1), f32(x3), f32(x5), f32(x7)) * scale_mid.x + scale_mid.y
    );
}

// See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1535
fn dequantize_q4_1x2(q: BlockQ4_1x2) -> array<vec4<f32>, 16> {
    let zero = vec4(0.0);
    var result = array<vec4<f32>, 16>();

    // First block.
    let scale_mid_a = unpack2x16float(q.data[0]);
    for (var k = 0; k < 4; k++) {
        let parts = dequantize_q4_1x2_part(q.data[k + 1], scale_mid_a);
        result[k] = parts[0];
        result[4 + k] = parts[1];
    }

    // Second block.
    let scale_mid_b = unpack2x16float(q.data[5]);
    for (var k = 0; k < 4; k++) {
        let parts = dequantize_q4_1x2_part(q.data[k + 6], scale_mid_b);
        result[8 + k] = parts[0];
        result[12 + k] = parts[1];
    }

    return result;
}

// Contains two `BlockQ5_0(f16, u32, [i8; 16])`.
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L26-L30
struct BlockQ5_0x2 {
    data: array<u32, 11>,
}

fn dequantize_q5_0x2_part(j0: u32, qh: u32, data: u32, scale: f32) -> array<vec4<f32>, 2> {
    let xh0 = ((qh >> j0) << 4) & 0x10;
    let x0 = i32(((data >> (0u * 8u)) & 0x0F) | xh0) - 16;
    let xh1 = (qh >> (j0 + 12)) & 0x10;
    let x1 = i32(((data >> (0u * 8u + 4u)) & 0x0F) | xh1) - 16;
    let xh2 = ((qh >> (j0 + 1)) << 4) & 0x10;
    let x2 = i32(((data >> (1u * 8u)) & 0x0F) | xh2) - 16;
    let xh3 = (qh >> ((j0 + 1) + 12)) & 0x10;
    let x3 = i32(((data >> (1u * 8u + 4u)) & 0x0F) | xh3) - 16;
    let xh4 = ((qh >> (j0 + 2)) << 4) & 0x10;
    let x4 = i32(((data >> (2u * 8u)) & 0x0F) | xh4) - 16;
    let xh5 = (qh >> ((j0 + 2) + 12)) & 0x10;
    let x5 = i32(((data >> (2u * 8u + 4u)) & 0x0F) | xh5) - 16;
    let xh6 = ((qh >> (j0 + 3)) << 4) & 0x10;
    let x6 = i32(((data >> (3u * 8u)) & 0x0F) | xh6) - 16;
    let xh7 = (qh >> ((j0 + 3) + 12)) & 0x10;
    let x7 = i32(((data >> (3u * 8u + 4u)) & 0x0F) | xh7) - 16;
    return array(
        vec4(f32(x0), f32(x2), f32(x4), f32(x6)) * scale,
        vec4(f32(x1), f32(x3), f32(x5), f32(x7)) * scale
    );
}

// See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1556
fn dequantize_q5_0x2(q: BlockQ5_0x2) -> array<vec4<f32>, 16> {
    let zero = vec4(0.0);
    var result = array<vec4<f32>, 16>();

    // First block.
    let d1 = unpack2x16float(q.data[0]).x;
    let qh1 = q.data[0] >> 16u | q.data[1] << 16u;

    for (var k = 0u; k < 4u; k++) {
        let data = q.data[k + 1] >> 16u | q.data[k + 2] << 16u;
        let parts = dequantize_q5_0x2_part(k * 4, qh1, data, d1);
        result[k] = parts[0];
        result[4 + k] = parts[1];
    }

    // Second block.
    let d2 = unpack2x16float(q.data[5]).y;
    let qh2 = q.data[6];

    for (var k = 0u; k < 4u; k++) {
        let data = q.data[k + 7];
        let parts = dequantize_q5_0x2_part(k * 4, qh2, data, d2);
        result[8 + k] = parts[0];
        result[12 + k] = parts[1];
    }

    return result;
}


// Contains two `BlockQ5_1(f16, u32, [i8; 16])`.
// TODO: could actually just be a single BlockQ5_1 which is already properly aligned
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L26-L30
struct BlockQ5_1x2 {
    data: array<u32, 12>,
}

fn dequantize_q5_1x2_part(j0: u32, qh: u32, data: u32, d_m: vec2<f32>) -> array<vec4<f32>, 2> {
    let xh0 = ((qh >> j0) << 4) & 0x10;
    let x0 = ((data >> (0u * 8u)) & 0x0F) | xh0;
    let xh1 = (qh >> (j0 + 12)) & 0x10;
    let x1 = ((data >> (0u * 8u + 4u)) & 0x0F) | xh1;
    let xh2 = ((qh >> (j0 + 1)) << 4) & 0x10;
    let x2 = ((data >> (1u * 8u)) & 0x0F) | xh2;
    let xh3 = (qh >> ((j0 + 1) + 12)) & 0x10;
    let x3 = ((data >> (1u * 8u + 4u)) & 0x0F) | xh3;
    let xh4 = ((qh >> (j0 + 2)) << 4) & 0x10;
    let x4 = ((data >> (2u * 8u)) & 0x0F) | xh4;
    let xh5 = (qh >> ((j0 + 2) + 12)) & 0x10;
    let x5 = ((data >> (2u * 8u + 4u)) & 0x0F) | xh5;
    let xh6 = ((qh >> (j0 + 3)) << 4) & 0x10;
    let x6 = ((data >> (3u * 8u)) & 0x0F) | xh6;
    let xh7 = (qh >> ((j0 + 3) + 12)) & 0x10;
    let x7 = ((data >> (3u * 8u + 4u)) & 0x0F) | xh7;
    return array(
        vec4(f32(x0), f32(x2), f32(x4), f32(x6)) * d_m.x + d_m.y,
        vec4(f32(x1), f32(x3), f32(x5), f32(x7)) * d_m.x + d_m.y
    );
}

// See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1556
fn dequantize_q5_1x2(q: BlockQ5_1x2) -> array<vec4<f32>, 16> {
    let zero = vec4(0.0);
    var result = array<vec4<f32>, 16>();

    // First block.
    let d_m1 = unpack2x16float(q.data[0]);
    let qh1 = q.data[1];

    for (var k = 0u; k < 4u; k++) {
        let data = q.data[k + 2];
        let parts = dequantize_q5_1x2_part(k * 4, qh1, data, d_m1);
        result[k] = parts[0];
        result[4 + k] = parts[1];
    }

    // Second block.
    let d_m2 = unpack2x16float(q.data[6]);
    let qh2 = q.data[7];

    for (var k = 0u; k < 4u; k++) {
        let data = q.data[k + 8];
        let parts = dequantize_q5_1x2_part(k * 4, qh2, data, d_m2);
        result[8 + k] = parts[0];
        result[12 + k] = parts[1];
    }

    return result;
}

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L161-L165
struct BlockQ8_K {
    /// Delta
    d: f32,
    /// Quants
    qs: array<u32, 64>, // [i8; 256]
    /// Sum of quants in groups of 16
    bsums: array<u32, 8> // [i16; 256 / 16]
}

fn dequantize_q8_k(block: BlockQ8_K) -> array<vec4<f32>, 64> {
    var result = array<vec4<f32>, 64>();

    for (var j = 0; j < 64; j++) {
        // TODO: bsums is not used? Is this useful
        //       for optimizing matmul?
        let qs = unpack4xI8(block.qs[j]);
        result[j] = vec4<f32>(qs) * block.d;
    }

    return result;
}

struct BlockQ6_Kx2 {
    data: array<u32, 105>
}

// NOTE: this one is a bit more complicated than the other K quantizations
//       because of the nonoptimal alignment of both blocks.
fn dequantize_q6_kx2_ref(block: BlockQ6_Kx2) -> array<vec4<f32>, 128> {
    var result = array<vec4<f32>, 128>();

    // Block A
    // Its data goes from data[0] to half of data[52]
    // It is mostly well aligned with the original BlockQ6_K except for the original value.
    let d_a = unpack2x16float(block.data[52]).x;

    const _0xF = vec4(0xFu);
    const _6 = vec4(6u);
    const _4 = vec4(4u);
    const _3 = vec4(3u);
    const _2 = vec4(2u);
    const _32 = vec4(32);


    for (var i = 0u; i < 2u; i++) {
        const QL0: u32 = 0u;
        const QH0: u32 = 32u;
        const SC0: u32 = 48u;

        let data0 = vec4<f32>(unpack4xI8(block.data[SC0 + i * 2u])) * d_a;
        let data1 = vec4<f32>(unpack4xI8(block.data[SC0 + i * 2u + 1])) * d_a;

        for (var l = 0u; l < 8u; l++) {
            let is = l / 4u; // NOTE: is is either 0 or 1

            let is_shift = is * 8u;
            let is_shift2 = (is + 2u) * 8u;
            let qh = unpack4xU8(block.data[l + QH0 + i * 8u]);
            let ql0 = unpack4xU8(block.data[l + QL0 + i * 16u]);
            let ql32 = unpack4xU8(block.data[l + QL0 + i * 16u + 8u]);

            let q1 = vec4<i32>((ql0 & _0xF) | ((qh & _3) << _4)) - _32;
            let q2 = vec4<i32>((ql32 & _0xF) | (((qh >> _2) & _3) << _4)) - _32;
            let q3 = vec4<i32>((ql0 >> _4) | (((qh >> _4) & _3) << _4)) - _32;
            let q4 = vec4<i32>((ql32 >> _4) | (((qh >> _6) & _3) << _4)) - _32;

            result[l + i * 32u + 0u] = data0[is] * vec4<f32>(q1);
            result[l + i * 32u + 8u] = data0[is + 2] * vec4<f32>(q2);
            result[l + i * 32u + 16u] = data1[is] * vec4<f32>(q3);
            result[l + i * 32u + 24u] = data1[is + 2] * vec4<f32>(q4);
        }
    }

    // Block B
    // Its data goes from half of data[52] to data[104].
    // All values are starting with the u16 leftmost bits of the previous index.
    // So this is a copy-past of BlockB butwith the `-1u` and `>> 16`, `<< 16`
    // schenanigans to reconstruct the properly aligned u32 values.
    let d_b = unpack2x16float(block.data[104]).y;
    for (var i = 0u; i < 2u; i++) {
        const QL0: u32 = 53u + 0u;
        const QH0: u32 = 53u + 32u;
        const SC0: u32 = 53u + 48u;

        let isc0 = SC0 + i * 2u;
        let isc1 = SC0 + i * 2u + 1;
        let data0 = vec4<f32>(unpack4xI8((block.data[isc0 - 1u] >> 16) | (block.data[isc0] << 16))) * d_b;
        let data1 = vec4<f32>(unpack4xI8((block.data[isc1 - 1u] >> 16) | (block.data[isc1] << 16))) * d_b;

        for (var l = 0u; l < 8u; l++) {
            // NOTE: the `* 4u` in these consts is needed because we divide by 4 further down.
            let is = l / 4u; // NOTE: is is either 0 or 1

            let is_shift = is * 8u;
            let is_shift2 = (is + 2u) * 8u;

            let iqh = l + QH0 + i * 8u;
            let iql0 = l + QL0 + i * 16u;
            let iql32 = l + QL0 + i * 16u + 8u;

            let qh = unpack4xU8((block.data[iqh - 1u] >> 16) | (block.data[iqh] << 16));
            let ql0 = unpack4xU8((block.data[iql0 - 1u] >> 16) | (block.data[iql0] << 16));
            let ql32 = unpack4xU8((block.data[iql32 - 1u] >> 16) | (block.data[iql32] << 16));

            let q1 = vec4<i32>((ql0 & _0xF) | ((qh & _3) << _4)) - _32;
            let q2 = vec4<i32>((ql32 & _0xF) | (((qh >> _2) & _3) << _4)) - _32;
            let q3 = vec4<i32>((ql0 >> _4) | (((qh >> _4) & _3) << _4)) - _32;
            let q4 = vec4<i32>((ql32 >> _4) | (((qh >> _6) & _3) << _4)) - _32;

            result[64u + l + i * 32u + 0u] = data0[is] * vec4<f32>(q1);
            result[64u + l + i * 32u + 8u] = data0[is + 2] * vec4<f32>(q2);
            result[64u + l + i * 32u + 16u] = data1[is] * vec4<f32>(q3);
            result[64u + l + i * 32u + 24u] = data1[is + 2] * vec4<f32>(q4);
        }
    }

    return result;
}

// NOTE: this is the same as dequantize_q6_k_ref, bet rearranged to
//       facilitate its integration into gemv where multiple workgroup
//       threads work on the same quantized block.
fn dequantize_q6_kx2(block: BlockQ6_Kx2) -> array<vec4<f32>, 128> {
    var result = array<vec4<f32>, 128>();

    for (var k = 0u; k < 32; k++) {
        let part = dequantize_q6_kx2_workgroup(block, k);
        let base = (k / 16u) * 64u + ((k / 8u) % 2u) * 32u + (k % 8u);
        result[base] = part[0];
        result[base + 8u] = part[1];
        result[base + 16u] = part[2];
        result[base + 24u] = part[3];
    }

    return result;
}


fn dequantize_q6_kx2_workgroup(block: BlockQ6_Kx2, k: u32) -> array<vec4<f32>, 4> {
    const _0xF = vec4(0xFu);
    const _6 = vec4(6u);
    const _4 = vec4(4u);
    const _3 = vec4(3u);
    const _2 = vec4(2u);
    const _32 = vec4(32);

    if k / 16u == 0u {
        // Block A
        // Its data goes from data[0] to half of data[52]
        // It is mostly well aligned with the original BlockQ6_K except for the original value.
        let d_a = unpack2x16float(block.data[52]).x;

        const QL0: u32 = 0u;
        const QH0: u32 = 32u;
        const SC0: u32 = 48u;

        let i = (k / 8u) % 2u;
        let data0 = vec4<f32>(unpack4xI8(block.data[SC0 + i * 2u])) * d_a;
        let data1 = vec4<f32>(unpack4xI8(block.data[SC0 + i * 2u + 1])) * d_a;

        let l = k % 8u;
        let is = l / 4u; // NOTE: is is either 0 or 1

        let is_shift = is * 8u;
        let is_shift2 = (is + 2u) * 8u;
        let qh = unpack4xU8(block.data[l + QH0 + i * 8u]);
        let ql0 = unpack4xU8(block.data[l + QL0 + i * 16u]);
        let ql32 = unpack4xU8(block.data[l + QL0 + i * 16u + 8u]);

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
        let d_b = unpack2x16float(block.data[104]).y;

        const QL0: u32 = 53u + 0u;
        const QH0: u32 = 53u + 32u;
        const SC0: u32 = 53u + 48u;

        let i = (k / 8u) % 2u;
        let l = k % 8u;
        let isc0 = SC0 + i * 2u;
        let isc1 = SC0 + i * 2u + 1;
        let data0 = vec4<f32>(unpack4xI8((block.data[isc0 - 1u] >> 16) | (block.data[isc0] << 16))) * d_b;
        let data1 = vec4<f32>(unpack4xI8((block.data[isc1 - 1u] >> 16) | (block.data[isc1] << 16))) * d_b;

        // NOTE: the `* 4u` in these consts is needed because we divide by 4 further down.
        let is = l / 4u; // NOTE: is is either 0 or 1

        let is_shift = is * 8u;
        let is_shift2 = (is + 2u) * 8u;

        let iqh = l + QH0 + i * 8u;
        let iql0 = l + QL0 + i * 16u;
        let iql32 = l + QL0 + i * 16u + 8u;

        let qh = unpack4xU8((block.data[iqh - 1u] >> 16) | (block.data[iqh] << 16));
        let ql0 = unpack4xU8((block.data[iql0 - 1u] >> 16) | (block.data[iql0] << 16));
        let ql32 = unpack4xU8((block.data[iql32 - 1u] >> 16) | (block.data[iql32] << 16));

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


// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L130-L135
struct BlockQ5_K {
    d_dmin: u32,           // (u16, u16) = super-block scale, super-block scale for quantized mins
    scales: array<u32, 3>, // [u8; 12] = scales and mins, quantized with 6 bits
    qh: array<u32, 8>,     // [u8; 256 / 8] = quants, high bit
    qs: array<u32, 32>,    // [u8; 256 / 2] = quants, low 4 bits
}

fn dequantize_q5_k_ref(block: BlockQ5_K) -> array<vec4<f32>, 64> {
    var result = array<vec4<f32>, 64>();

    let d_dmin = unpack2x16float(block.d_dmin);
    let d = d_dmin.x;
    let min = d_dmin.y;

    for (var j = 0u; j < 4u; j++) {
        let is = j * 2u;
        let iq = j * 8u;
        let u1 = 1u << (j * 2u);
        let u2 = 2u << (j * 2u);

        let qj_prev1 = block.scales[max(is / 4, 1u) - 1];
        let qj1 = block.scales[is / 4];
        let qj_next1 = block.scales[is / 4 + 1];
        let sc_m1 = get_scale_min_k4(is, qj_prev1, qj1, qj_next1);
        let d1 = d * f32(sc_m1.x);
        let m1 = min * f32(sc_m1.y);

        let qj_prev2 = block.scales[max((is + 1) / 4, 1u) - 1];
        let qj2 = block.scales[(is + 1) / 4];
        let qj_next2 = block.scales[(is + 1) / 4 + 1];
        let sc_m2 = get_scale_min_k4(is + 1, qj_prev2, qj2, qj_next2);
        let d2 = d * f32(sc_m2.x);
        let m2 = min * f32(sc_m2.y);

        for (var l = 0u; l < 8; l++) {
            result[j * 16u + l] =
                vec4(
                    f32((block.qs[iq + l] & 0xF) + select(0u, 16u, (block.qh[l] & u1) != 0)),
                    f32(((block.qs[iq + l] >> 8) & 0xF) + select(0u, 16u, ((block.qh[l] >> 8) & u1) != 0)),
                    f32(((block.qs[iq + l] >> 16) & 0xF) + select(0u, 16u, ((block.qh[l] >> 16) & u1) != 0)),
                    f32(((block.qs[iq + l] >> 24) & 0xF) + select(0u, 16u, ((block.qh[l] >> 24) & u1) != 0)),
                ) * d1 - m1;
        }

        for (var l = 0u; l < 8; l++) {
            result[j * 16u + l + 8u] =
                vec4(
                    f32(((block.qs[iq + l] >> 4) & 0xF) + select(0u, 16u, (block.qh[l] & u2) != 0)),
                    f32(((block.qs[iq + l] >> 12) & 0xF) + select(0u, 16u, ((block.qh[l] >> 8) & u2) != 0)),
                    f32(((block.qs[iq + l] >> 20) & 0xF) + select(0u, 16u, ((block.qh[l] >> 16) & u2) != 0)),
                    f32(((block.qs[iq + l] >> 28) & 0xF) + select(0u, 16u, ((block.qh[l] >> 24) & u2) != 0)),
                ) * d2 - m2;
        }
    }

    return result;
}

// NOTE: this is the same as dequantize_q5_k_ref, bet rearranged to
//       facilitate its integration into gemv where multiple workgroup
//       threads work on the same quantized block.
fn dequantize_q5_k(block: BlockQ5_K) -> array<vec4<f32>, 64> {
    var result = array<vec4<f32>, 64>();

    for (var k = 0u; k < 32; k++) {
        let j = k / 8u;
        let part = dequantize_q5_k_workgroup(block, k);
        result[k + j * 8u] = part[0];
        result[k + j * 8u + 8u] = part[1];
    }

    return result;
}

fn dequantize_q5_k_workgroup(block: BlockQ5_K, k: u32) -> array<vec4<f32>, 2> {
    let d_dmin = unpack2x16float(block.d_dmin);
    let d = d_dmin.x;
    let min = d_dmin.y;

    let j = k / 8u;
    let l = k % 8u;
    let is = j * 2u;
    let iq = j * 8u;
    let u1 = 1u << (j * 2u);
    let u2 = 2u << (j * 2u);

    let qj_prev1 = block.scales[max(is / 4, 1u) - 1];
    let qj1 = block.scales[is / 4];
    let qj_next1 = block.scales[is / 4 + 1];
    let sc_m1 = get_scale_min_k4(is, qj_prev1, qj1, qj_next1);
    let d1 = d * f32(sc_m1.x);
    let m1 = min * f32(sc_m1.y);

    let qj_prev2 = block.scales[max((is + 1) / 4, 1u) - 1];
    let qj2 = block.scales[(is + 1) / 4];
    let qj_next2 = block.scales[(is + 1) / 4 + 1];
    let sc_m2 = get_scale_min_k4(is + 1, qj_prev2, qj2, qj_next2);
    let d2 = d * f32(sc_m2.x);
    let m2 = min * f32(sc_m2.y);

    let qs = block.qs[k];
    let qh = block.qh[l];
    let res_a = vec4(
        f32((qs & 0xF) + select(0u, 16u, (qh & u1) != 0)),
        f32(((qs >> 8) & 0xF) + select(0u, 16u, ((qh >> 8) & u1) != 0)),
        f32(((qs >> 16) & 0xF) + select(0u, 16u, ((qh >> 16) & u1) != 0)),
        f32(((qs >> 24) & 0xF) + select(0u, 16u, ((qh >> 24) & u1) != 0)),
    ) * d1 - m1;

    let res_b = vec4(
        f32(((qs >> 4) & 0xF) + select(0u, 16u, (qh & u2) != 0)),
        f32(((qs >> 12) & 0xF) + select(0u, 16u, ((qh >> 8) & u2) != 0)),
        f32(((qs >> 20) & 0xF) + select(0u, 16u, ((qh >> 16) & u2) != 0)),
        f32(((qs >> 28) & 0xF) + select(0u, 16u, ((qh >> 24) & u2) != 0)),
    ) * d2 - m2;

    return array(res_a, res_b);
}

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L109-L113
struct BlockQ4_K {
    d_dmin: u32,           // (u16, u16) = super-block scale, super-block scale for quantized mins
    scales: array<u32, 3>, // [u8; 12] = scales and mins, quantized with 6 bits
    qs: array<u32, 32>,    // [u8; 256 / 2] = 4-bit quants
}


fn dequantize_q4_k_ref(block: BlockQ4_K) -> array<vec4<f32>, 64> {
    var result = array<vec4<f32>, 64>();

    let d_dmin = unpack2x16float(block.d_dmin);
    let d = d_dmin.x;
    let min = d_dmin.y;

    for (var j = 0u; j < 4; j++) {
        let is = j * 2u;
        let qj_prev1 = block.scales[max(is / 4, 1u) - 1];
        let qj1 = block.scales[is / 4];
        let qj_next1 = block.scales[is / 4 + 1];
        let sc_m1 = get_scale_min_k4(is, qj_prev1, qj1, qj_next1);
        let d1 = d * f32(sc_m1.x);
        let m1 = min * f32(sc_m1.y);

        let qj_prev2 = block.scales[max((is + 1) / 4, 1u) - 1];
        let qj2 = block.scales[(is + 1) / 4];
        let qj_next2 = block.scales[(is + 1) / 4 + 1];
        let sc_m2 = get_scale_min_k4(is + 1, qj_prev2, qj2, qj_next2);
        let d2 = d * f32(sc_m2.x);
        let m2 = min * f32(sc_m2.y);

        let iq = j * 8u;

        for (var l = 0u; l < 8; l++) {
            result[j * 16u + l] =
                vec4(
                    f32(block.qs[iq + l] & 0xF),
                    f32((block.qs[iq + l] >> 8) & 0xF),
                    f32((block.qs[iq + l] >> 16) & 0xF),
                    f32((block.qs[iq + l] >> 24) & 0xF),
                ) * d1 - m1;
        }

        for (var l = 0u; l < 8; l++) {
            result[j * 16u + l + 8u] =
                vec4(
                    f32((block.qs[iq + l] >> 4) & 0xF),
                    f32((block.qs[iq + l] >> 12) & 0xF),
                    f32((block.qs[iq + l] >> 20) & 0xF),
                    f32((block.qs[iq + l] >> 28) & 0xF),
                ) * d2 - m2;
        }
    }

    return result;
}

// NOTE: this is the same as dequantize_q4_k_ref, bet rearranged to
//       facilitate its integration into gemv where multiple workgroup
//       threads work on the same quantized block.
fn dequantize_q4_k(block: BlockQ4_K) -> array<vec4<f32>, 64> {
    var result = array<vec4<f32>, 64>();

    for (var k = 0u; k < 32; k++) {
        let j = k / 8u;
        let part = dequantize_q4_k_workgroup(block, k);
        result[k + j * 8u] = part[0];
        result[k + j * 8u + 8u] = part[1];
    }

    return result;
}

// NOTE: this code has been copied to `gemv_quant_q4_k.wgsl` with the
//       `block` argument replaced by an index to a storage buffer.
//       Any change here should be applied to that other file so it can
//       benefit from the improvements.
fn dequantize_q4_k_workgroup(block: BlockQ4_K, k: u32) -> array<vec4<f32>, 2> {
    let d_dmin = unpack2x16float(block.d_dmin);
    let d = d_dmin.x;
    let min = d_dmin.y;

    // 32 threads workgroups
    let j = k / 8u;
    let is = j * 2u;
    let qj_prev1 = block.scales[max(is / 4, 1u) - 1];
    let qj1 = block.scales[is / 4];
    let qj_next1 = block.scales[is / 4 + 1];
    let sc_m1 = get_scale_min_k4(is, qj_prev1, qj1, qj_next1);
    let d1 = d * f32(sc_m1.x);
    let m1 = min * f32(sc_m1.y);

    let qj_prev2 = block.scales[max((is + 1) / 4, 1u) - 1];
    let qj2 = block.scales[(is + 1) / 4];
    let qj_next2 = block.scales[(is + 1) / 4 + 1];
    let sc_m2 = get_scale_min_k4(is + 1, qj_prev2, qj2, qj_next2);
    let d2 = d * f32(sc_m2.x);
    let m2 = min * f32(sc_m2.y);

    // NOTE: from the ref implementation:
    //    iq     == (k / 8u) * 8u
    // => iq + l == (k / 8u) * 8u + k % 8u == k
    let qs = block.qs[k];

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

fn get_scale_min_k4(j: u32, qj_prev_: u32, qj_: u32, qj_next_: u32) -> vec2<u32> {
    let shift = (j % 4) * 8;
    let qj_prev = (qj_prev_ >> shift) & 0x00ff;
    let qj = (qj_ >> shift) & 0x00ff;
    let qj_next = (qj_next_ >> shift) & 0x00ff;

    if j < 4 {
        let d = qj & 63;
        let m = qj_next & 63;
        return vec2(d, m);
    } else {
        let d = (qj_next & 0xf) | ((qj_prev >> 6) << 4);
        let m = (qj_next >> 4) | ((qj >> 6) << 4);
        return vec2(d, m);
    }
}

#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape_q: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_k: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape: RoPEShape;
@group(0) @binding(3)
var<storage, read_write> in_out_q: array<f32>;
@group(0) @binding(4)
var<storage, read_write> in_out_k: array<f32>;



struct RoPEShape {
    head_size: u32,
    kv_dim: u32,
    pos: u32,
}

struct Rotation2 {
    cos: f32,
    sin: f32,
}

fn rot2(angle: f32) -> Rotation2 {
    return Rotation2(cos(angle), sin(angle));
}

fn rotate2(r: Rotation2, vx: f32, vy: f32) -> vec2<f32> {
    return vec2(r.cos * vx - r.sin * vy, r.sin * vx + r.cos * vy);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    let head_dim = f32((i * 2) % shape.head_size);
    // TODO: make the base freq configurable some models use 500K or 1M.
    let theta = pow(10000.0, -head_dim / f32(shape.head_size));
    let m_theta = f32(shape.pos) * theta;
    let rot = rot2(m_theta);

    let iq = Shape::iv(shape_q, i * 2);
    let q_rotated = rotate2(rot, in_out_q[iq], in_out_q[iq + 1]);
    in_out_q[iq] = q_rotated.x;
    in_out_q[iq + 1] = q_rotated.y;

    if (i * 2 < shape.kv_dim) {
        let ik = Shape::iv(shape_k, i * 2);
        let k_rotated = rotate2(rot, in_out_k[ik], in_out_k[ik + 1]);
        in_out_k[ik] = k_rotated.x;
        in_out_k[ik + 1] = k_rotated.y;
    }
}

@compute @workgroup_size(64, 1, 1)
fn main_neox(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    let head_dim = f32((i * 2) % shape.head_size);
    // TODO: make the base freq configurable some models use 500K or 1M.
    let theta = pow(1000000.0, -head_dim / f32(shape.head_size));
    let m_theta = f32(shape.pos) * theta;
    let rot = rot2(m_theta);

    let head_id = (i * 2) / shape.head_size;
    let shift = shape.head_size / 2;

    let iq = Shape::iv(shape_q, i + head_id * shape.head_size / 2);
    let q_rotated = rotate2(rot, in_out_q[iq], in_out_q[iq + shift]);
    in_out_q[iq] = q_rotated.x;
    in_out_q[iq + shift] = q_rotated.y;

    if (i * 2 < shape.kv_dim) {
        let ik = Shape::iv(shape_k, i + head_id * shape.head_size / 2);
        let k_rotated = rotate2(rot, in_out_k[ik], in_out_k[ik + shift]);
        in_out_k[ik] = k_rotated.x;
        in_out_k[ik + shift] = k_rotated.y;
    }
}
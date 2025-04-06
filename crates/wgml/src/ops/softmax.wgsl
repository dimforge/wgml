#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape: Shape::Shape;
@group(0) @binding(1)
var<storage, read_write> in_out_mat: array<f32>;

const WORKGROUP_SIZE: u32 = 64;

var<workgroup> workspace: array<f32, WORKGROUP_SIZE>;
var<workgroup> the_max: f32;
var<workgroup> denominator: f32;

fn reduce_max(thread_id: u32, stride: u32) {
    workgroupBarrier();

    if thread_id < stride {
        workspace[thread_id] = max(workspace[thread_id], workspace[thread_id + stride]);
    }
}

fn reduce_sum(thread_id: u32, stride: u32) {
    workgroupBarrier();

    if thread_id < stride {
        workspace[thread_id] += workspace[thread_id + stride];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let j = workgroup_id.x;
    let thread_id = local_id.x;

    // Compute the MAX
    let data_len = shape.nrows;
    var my_max = -1.0e38;

    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = in_out_mat[Shape::im(shape, i, j)];
        my_max = max(my_max, val_i);
    }

    workspace[thread_id] = my_max;

//    reduce_max(thread_id, 64u);
    reduce_max(thread_id, 32u);
    reduce_max(thread_id, 16u);
    reduce_max(thread_id, 8u);
    reduce_max(thread_id, 4u);
    reduce_max(thread_id, 2u);
    reduce_max(thread_id, 1u);

    if (thread_id == 0) {
        the_max = workspace[0];
    }

    workgroupBarrier();

    // Compute the denominator (sum of exponentials).
    var my_denominator = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = Shape::im(shape, i, j);
        let val_i = in_out_mat[ii];
        let exp_i = exp(val_i - the_max);
        my_denominator += exp_i;
        in_out_mat[ii] = exp_i;
    }

    workspace[thread_id] = my_denominator;

//    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        denominator = workspace[0];
    }

    workgroupBarrier();

    // Divide by the denominator.
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = Shape::im(shape, i, j);
        let val_i = in_out_mat[ii];
        in_out_mat[ii] = val_i / denominator;
    }
}

use magnus::{prelude::*, Ruby, RArray, Value};
use ndarray::Array2;
use std::f64;

// === Flat MatMul ===
fn fast_matmul_flat(
    a: Vec<f64>,
    b: Vec<f64>,
    m: usize,
    k: usize,
    n: usize
) -> Vec<f64> {
    let a_arr = Array2::from_shape_vec((m, k), a).unwrap();
    let b_arr = Array2::from_shape_vec((k, n), b).unwrap();
    let c_arr = a_arr.dot(&b_arr);
    c_arr.into_raw_vec()
}

// === Conv2D ===
fn conv2d_forward(
    input: Vec<f64>,
    weight: Vec<f64>,
    bias: Vec<f64>,
    batch: usize,
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    k: usize,
) -> Vec<f64> {
    let h_out = h - k + 1;
    let w_out = w - k + 1;
    let mut output = vec![0.0; batch * out_c * h_out * w_out];

    for b in 0..batch {
        for oc in 0..out_c {
            for i in 0..h_out {
                for j in 0..w_out {
                    let mut sum = bias[oc];
                    for ic in 0..in_c {
                        for ki in 0..k {
                            for kj in 0..k {
                                let in_idx = ((b * in_c + ic) * h + i + ki) * w + j + kj;
                                let w_idx = ((oc * in_c + ic) * k + ki) * k + kj;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    let out_idx = ((b * out_c + oc) * h_out + i) * w_out + j;
                    output[out_idx] = sum;
                }
            }
        }
    }
    output
}

// === MaxPool2D ===
fn maxpool2d_forward(
    input: Vec<f64>,
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    k: usize,
) -> Vec<f64> {
    let h_out = h / k;
    let w_out = w / k;
    let mut output = vec![0.0; batch * channels * h_out * w_out];

    for b in 0..batch {
        for c in 0..channels {
            for i in 0..h_out {
                for j in 0..w_out {
                    let mut max_val = f64::MIN;
                    for ki in 0..k {
                        for kj in 0..k {
                            let in_idx = ((b * channels + c) * h + i * k + ki) * w + j * k + kj;
                            max_val = max_val.max(input[in_idx]);
                        }
                    }
                    let out_idx = ((b * channels + c) * h_out + i) * w_out + j;
                    output[out_idx] = max_val;
                }
            }
        }
    }
    output
}

// === Инициализация ===
#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), magnus::Error> {
    let module = ruby.define_module("Daimond")?
        .define_module("Rust")?;

    // Только flat-версии (быстрее и проще)
    module.define_singleton_method("fast_matmul_flat", magnus::function!(fast_matmul_flat, 5))?;
    module.define_singleton_method("conv2d_native", magnus::function!(conv2d_forward, 9))?;
    module.define_singleton_method("maxpool2d_native", magnus::function!(maxpool2d_forward, 6))?;

    Ok(())
}
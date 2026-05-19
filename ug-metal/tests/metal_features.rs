use ug::lang::LaunchConfig;
use ug::{Device as _, Result, Slice as _};
use ug_metal::runtime::{Device, Slice};

fn slice_f32(dev: &Device, data: &[f32]) -> Slice {
    let mut s = unsafe { dev.allocate_uninit(ug::DType::F32, data.len()).unwrap() };
    s.copy_host_to_device(data).unwrap();
    s
}

fn zeros_f32(dev: &Device, n: usize) -> Slice {
    let data = vec![0.0f32; n];
    slice_f32(dev, &data)
}

// ── Causal mask kernel ──────────────────────────────────────────────────

const CAUSAL_MASK_3X3_M: &str = r#"
using uint32_t = unsigned int;
kernel void causal_mask_3x3_f32(
    device const float *src,
    device float *dst,
    uint tid [[ thread_position_in_grid ]]
) {
    const uint32_t s = 3;
    const uint32_t n = 9;
    if (tid >= n) return;
    uint32_t i1 = tid / s;
    uint32_t i2 = tid % s;
    dst[tid] = (i2 <= i1) ? src[tid] : -INFINITY;
}
"#;

const CAUSAL_MASK_1X1_M: &str = r#"
kernel void causal_mask_1x1_f32(
    device const float *src,
    device float *dst,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= 1) return;
    dst[tid] = src[tid];
}
"#;

#[test]
fn causal_mask_basic() -> Result<()> {
    let dev = Device::new()?;
    let s = 3u32;
    let n = s * s;
    let cfg = LaunchConfig::new_1d(1, 32);
    let func = dev.compile_metal(CAUSAL_MASK_3X3_M, "causal_mask_3x3_f32", cfg)?;
    let src_data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let mut src = slice_f32(&dev, &src_data);
    let mut dst = zeros_f32(&dev, n as usize);

    dev.run(&func, &mut [&mut src, &mut dst])?;

    let out = dst.to_vec::<f32>()?;
    assert_eq!(out[0], 1.0);
    assert!(out[1].is_infinite() && out[1] < 0.0);
    assert!(out[2].is_infinite() && out[2] < 0.0);
    assert_eq!(out[3], 4.0);
    assert_eq!(out[4], 5.0);
    assert!(out[5].is_infinite() && out[5] < 0.0);
    assert_eq!(out[6], 7.0);
    assert_eq!(out[7], 8.0);
    assert_eq!(out[8], 9.0);
    Ok(())
}

#[test]
fn causal_mask_1x1() -> Result<()> {
    let dev = Device::new()?;
    let cfg = LaunchConfig::new_1d(1, 32);
    let func = dev.compile_metal(CAUSAL_MASK_1X1_M, "causal_mask_1x1_f32", cfg)?;
    let mut src = slice_f32(&dev, &[42.0f32]);
    let mut dst = zeros_f32(&dev, 1);

    dev.run(&func, &mut [&mut src, &mut dst])?;

    let out = dst.to_vec::<f32>()?;
    assert_eq!(out, [42.0]);
    Ok(())
}

// ── BF16/F16 matmul ────────────────────────────────────────────────────

#[test]
fn matmul_f16() -> Result<()> {
    use half::f16;
    let dev = Device::new()?;

    let m = 4;
    let k = 2;
    let n = 3;
    let lhs_data: Vec<f16> = vec![f16::from_f32(1.0); m * k];
    let rhs_data: Vec<f16> = vec![f16::from_f32(2.0); k * n];

    let mut lhs_slice = unsafe { dev.allocate_uninit(ug::DType::F16, m * k)? };
    lhs_slice.copy_host_to_device(&lhs_data)?;
    let mut rhs_slice = unsafe { dev.allocate_uninit(ug::DType::F16, k * n)? };
    rhs_slice.copy_host_to_device(&rhs_data)?;
    let mut dst_slice = unsafe { dev.allocate_uninit(ug::DType::F16, m * n)? };

    let lhs_layout = ug::Layout::from_shape(ug::Shape::from((m, k)));
    let rhs_layout = ug::Layout::from_shape(ug::Shape::from((k, n)));

    dev.matmul(&mut dst_slice, &lhs_slice, &rhs_slice, (1, m, n, k), &lhs_layout, &rhs_layout)?;

    let out = dst_slice.to_vec::<f16>()?;
    // Each element = sum of k elements of 1.0*2.0 = 2*2.0 = 4.0
    for v in &out {
        assert!((v.to_f32() - 4.0).abs() < 0.1, "expected 4.0, got {}", v.to_f32());
    }
    assert_eq!(out.len(), m * n);
    Ok(())
}

#[test]
fn matmul_bf16() -> Result<()> {
    use half::bf16;
    let dev = Device::new()?;

    let m = 4;
    let k = 2;
    let n = 3;
    let lhs_data: Vec<bf16> = vec![bf16::from_f32(1.0); m * k];
    let rhs_data: Vec<bf16> = vec![bf16::from_f32(2.0); k * n];

    let mut lhs_slice = unsafe { dev.allocate_uninit(ug::DType::BF16, m * k)? };
    lhs_slice.copy_host_to_device(&lhs_data)?;
    let mut rhs_slice = unsafe { dev.allocate_uninit(ug::DType::BF16, k * n)? };
    rhs_slice.copy_host_to_device(&rhs_data)?;
    let mut dst_slice = unsafe { dev.allocate_uninit(ug::DType::BF16, m * n)? };

    let lhs_layout = ug::Layout::from_shape(ug::Shape::from((m, k)));
    let rhs_layout = ug::Layout::from_shape(ug::Shape::from((k, n)));

    dev.matmul(&mut dst_slice, &lhs_slice, &rhs_slice, (1, m, n, k), &lhs_layout, &rhs_layout)?;

    let out = dst_slice.to_vec::<bf16>()?;
    for v in &out {
        assert!((v.to_f32() - 4.0).abs() < 0.5, "expected 4.0, got {}", v.to_f32());
    }
    assert_eq!(out.len(), m * n);
    Ok(())
}

// ── Pipeline caching ───────────────────────────────────────────────────

const ADD_KERNEL: &str = r#"
kernel void add_f32(
    device const float* a,
    device float* b,
    uint tid [[ thread_position_in_grid ]]
) {
    b[tid] = a[tid] + 1.0;
}
"#;

#[test]
fn pipeline_caching() -> Result<()> {
    let dev = Device::new()?;
    let n = 64;
    let cfg = LaunchConfig::new_1d(n / 32, 32);
    let func = dev.compile_metal(ADD_KERNEL, "add_f32", cfg)?;

    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut src_s = slice_f32(&dev, &src);
    let mut dst_s = zeros_f32(&dev, n as usize);

    // First call creates the pipeline
    let _pl1 = func.pipeline()?;
    dev.run(&func, &mut [&mut src_s, &mut dst_s])?;
    let out1 = dst_s.to_vec::<f32>()?;

    // Second call should reuse the cached pipeline
    let _pl2 = func.pipeline()?;
    dev.run(&func, &mut [&mut src_s, &mut dst_s])?;
    let out2 = dst_s.to_vec::<f32>()?;

    assert_eq!(out1, out2);
    for (i, v) in out1.iter().enumerate() {
        assert_eq!(*v, i as f32 + 1.0, "mismatch at index {i}");
    }
    Ok(())
}

// ── run_batched ────────────────────────────────────────────────────────

#[test]
fn run_batched_two_kernels() -> Result<()> {
    let dev = Device::new()?;
    let n = 32usize;
    let cfg = LaunchConfig::new_1d(1, 32);
    let func = dev.compile_metal(ADD_KERNEL, "add_f32", cfg)?;

    let src1: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let src2: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
    let mut src1_s = slice_f32(&dev, &src1);
    let mut dst1_s = zeros_f32(&dev, n);
    let mut src2_s = slice_f32(&dev, &src2);
    let mut dst2_s = zeros_f32(&dev, n);

    dev.run_batched(
        &[&func, &func],
        &mut [
            &mut [&mut src1_s, &mut dst1_s],
            &mut [&mut src2_s, &mut dst2_s],
        ],
    )?;

    let out1 = dst1_s.to_vec::<f32>()?;
    let out2 = dst2_s.to_vec::<f32>()?;

    for (i, v) in out1.iter().enumerate() {
        assert_eq!(*v, i as f32 + 1.0, "batch1 mismatch at {i}");
    }
    for (i, v) in out2.iter().enumerate() {
        assert_eq!(*v, (i as f32) * 10.0 + 1.0, "batch2 mismatch at {i}");
    }
    Ok(())
}

#[test]
fn run_batched_empty() -> Result<()> {
    let dev = Device::new()?;
    dev.run_batched(&[], &mut [])?;
    Ok(())
}

#[test]
fn run_batched_single() -> Result<()> {
    let dev = Device::new()?;
    let n = 32usize;
    let cfg = LaunchConfig::new_1d(1, 32);
    let func = dev.compile_metal(ADD_KERNEL, "add_f32", cfg)?;

    let src: Vec<f32> = vec![5.0; n];
    let mut src_s = slice_f32(&dev, &src);
    let mut dst_s = zeros_f32(&dev, n);

    dev.run_batched(&[&func], &mut [&mut [&mut src_s, &mut dst_s]])?;

    let out = dst_s.to_vec::<f32>()?;
    for v in &out {
        assert_eq!(*v, 6.0);
    }
    Ok(())
}

// ── Device::new() (all) ────────────────────────────────────────────────

#[test]
fn device_new_succeeds() -> Result<()> {
    let dev = Device::new()?;
    // Verify we can allocate on it
    let _s = unsafe { dev.allocate_uninit(ug::DType::F32, 16)? };
    Ok(())
}

#[test]
fn device_new_command_queue() -> Result<()> {
    let dev = Device::new()?;
    let _cq = dev.new_command_queue()?;
    Ok(())
}

#[test]
fn device_new_command_buffer() -> Result<()> {
    let dev = Device::new()?;
    let _cb = dev.new_command_buffer()?;
    Ok(())
}

#[test]
fn device_use_grid() {
    assert!(Device::use_grid());
}

// ── synchronize() ──────────────────────────────────────────────────────

#[test]
fn synchronize_empty() -> Result<()> {
    let dev = Device::new()?;
    dev.synchronize()?;
    Ok(())
}

#[test]
fn synchronize_after_run() -> Result<()> {
    let dev = Device::new()?;
    let n = 32usize;
    let cfg = LaunchConfig::new_1d(1, 32);
    let func = dev.compile_metal(ADD_KERNEL, "add_f32", cfg)?;

    let src: Vec<f32> = vec![1.0; n];
    let mut src_s = slice_f32(&dev, &src);
    let mut dst_s = zeros_f32(&dev, n);

    dev.run(&func, &mut [&mut src_s, &mut dst_s])?;
    dev.synchronize()?;

    let out = dst_s.to_vec::<f32>()?;
    for v in &out {
        assert_eq!(*v, 2.0);
    }
    Ok(())
}

#[test]
fn synchronize_flushes_async() -> Result<()> {
    let dev = Device::new()?;
    let n = 32usize;
    let cfg = LaunchConfig::new_1d(1, 32);
    let func = dev.compile_metal(ADD_KERNEL, "add_f32", cfg)?;

    let src: Vec<f32> = vec![10.0; n];
    let mut src_s = slice_f32(&dev, &src);
    let mut dst_s = zeros_f32(&dev, n);

    dev.run_async(&func, &mut [&mut src_s, &mut dst_s])?;
    dev.synchronize()?;

    let out = dst_s.to_vec::<f32>()?;
    for v in &out {
        assert_eq!(*v, 11.0);
    }
    Ok(())
}

// ── Shader compilation error ───────────────────────────────────────────

#[test]
fn shader_compile_error_syntax() {
    let dev = Device::new().unwrap();
    let bad_shader = "this is not valid metal code {{{{";
    let cfg = LaunchConfig::new_1d(1, 32);
    let result = dev.compile_metal(bad_shader, "nonexistent", cfg);
    assert!(result.is_err(), "expected compilation error for invalid shader");
}

#[test]
fn shader_compile_error_missing_function() {
    let dev = Device::new().unwrap();
    let valid_shader = r#"
kernel void real_func(device float* a [[ buffer(0) ]], uint tid [[ thread_position_in_grid ]]) {
    a[tid] = 0.0;
}
"#;
    let cfg = LaunchConfig::new_1d(1, 32);
    let result = dev.compile_metal(valid_shader, "does_not_exist", cfg);
    assert!(result.is_err(), "expected error for missing function name");
}

#[test]
fn shader_compile_error_type_mismatch() {
    let dev = Device::new().unwrap();
    let bad_types = r#"
kernel void bad_types(device float* a [[ buffer(0) ]], uint tid [[ thread_position_in_grid ]]) {
    // Deliberate type error: assigning string to float
    a[tid] = "hello";
}
"#;
    let cfg = LaunchConfig::new_1d(1, 32);
    let result = dev.compile_metal(bad_types, "bad_types", cfg);
    assert!(result.is_err(), "expected compilation error for type mismatch");
}


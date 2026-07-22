//! Utility types and functions for Metal GPU Merkle tree implementation.

use std::ptr;

use metal::MTLSize;

/// Uniforms passed to linear+threadgroup Metal shaders. Must match `LinearThreadgroupUniforms` in the MSL shader.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LinearUniforms {
    pub level: u32,
    pub subtree_digests_len: u32,
    pub subtree_leaves_len: u32,
    pub leaf_size: u32,
    pub leaf_count: u32,
    pub subtree_count: u32,
    pub grid_width: u32,
}

/// Copy raw data from a GPU buffer pointer into a new Vec.
///
/// # Safety
/// Caller must ensure `ptr` points to at least `elts` initialized `T` values.
pub unsafe fn from_buf_raw<T>(ptr: *const T, elts: usize) -> Vec<T> {
    let mut dst = Vec::with_capacity(elts);
    ptr::copy(ptr, dst.as_mut_ptr(), elts);
    dst.set_len(elts);
    dst
}

/// Compute a Metal dispatch grid large enough to cover `count` threads.
pub fn get_size_for_count(count: usize) -> MTLSize {
    const MAX_DIM: usize = 32768;
    if count <= MAX_DIM {
        MTLSize {
            width: count as u64,
            height: 1,
            depth: 1,
        }
    } else {
        MTLSize {
            width: MAX_DIM as u64,
            height: ((count + MAX_DIM - 1) / MAX_DIM) as u64,
            depth: 1,
        }
    }
}

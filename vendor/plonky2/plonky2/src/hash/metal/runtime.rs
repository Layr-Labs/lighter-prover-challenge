//! MetalRuntime core struct and buffer allocation methods.
//! Loads Poseidon2 Merkle shaders for GPU-accelerated tree construction.

use std::sync::Mutex;

use metal::*;
use once_cell::sync::Lazy;

use crate::hash::metal::buffer_pool::{BufferPool, BUFFER_POOL};
use crate::hash::metal::tracking::{track_allocation, TrackedBuffer};

/// Pre-compiled Poseidon2 linear+threadgroup shader library embedded at build time.
const SHADERLIB_POSEIDON2_LINEAR_THREADGROUP: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon2_merkle_hasher_linear_threadgroup.metallib"
));

/// Pre-compiled quotient polynomial shader library embedded at build time.
const SHADERLIB_QUOTIENT_POLY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/quotient_poly.metallib"
));

pub struct MetalRuntime {
    pub device: Mutex<Device>,
    #[allow(dead_code)]
    f_poseidon2_hash_leaves_linear_threadgroup: Function,
    #[allow(dead_code)]
    f_poseidon2_hash_tree_level_linear_threadgroup: Function,
    #[allow(dead_code)]
    f_poseidon2_hash_caps_linear_threadgroup: Function,
    pub(crate) pso_poseidon2_hash_leaves_linear_threadgroup: ComputePipelineState,
    pub(crate) pso_poseidon2_hash_tree_level_linear_threadgroup: ComputePipelineState,
    pub(crate) pso_poseidon2_hash_caps_linear_threadgroup: ComputePipelineState,
    #[allow(dead_code)]
    pub(crate) f_quotient_poly: Function,
    pub(crate) pso_quotient_poly: ComputePipelineState,
    pub(crate) command_queue: CommandQueue,
}

// SAFETY: See plonky2-fork/src/hash/metal/runtime.rs for rationale.
// Pipeline states are created once and read-only. Command queue is only
// accessed from the dedicated GPU dispatch thread.
unsafe impl Sync for MetalRuntime {}

fn create_pso(device: &Device, function: &Function) -> ComputePipelineState {
    device
        .new_compute_pipeline_state_with_function(function)
        .unwrap()
}

pub static RUNTIME: Lazy<MetalRuntime> = Lazy::new(|| {
    let device = Device::system_default().unwrap();
    let command_queue = device.new_command_queue();

    let lib_p2 = device
        .new_library_with_data(SHADERLIB_POSEIDON2_LINEAR_THREADGROUP)
        .unwrap();
    let f_leaves_p2 = lib_p2
        .get_function("poseidon2_hash_leaves_linear_threadgroup", None)
        .unwrap();
    let f_level_p2 = lib_p2
        .get_function("poseidon2_hash_tree_level_linear_threadgroup", None)
        .unwrap();
    let f_caps_p2 = lib_p2
        .get_function("poseidon2_hash_caps_linear_threadgroup", None)
        .unwrap();
    let pso_leaves_p2 = create_pso(&device, &f_leaves_p2);
    let pso_level_p2 = create_pso(&device, &f_level_p2);
    let pso_caps_p2 = create_pso(&device, &f_caps_p2);

    let lib_quotient = device
        .new_library_with_data(SHADERLIB_QUOTIENT_POLY)
        .unwrap();
    let f_quotient = lib_quotient
        .get_function("compute_quotient_poly", None)
        .unwrap();
    let pso_quotient = create_pso(&device, &f_quotient);

    MetalRuntime {
        device: Mutex::new(device),
        command_queue,
        pso_poseidon2_hash_leaves_linear_threadgroup: pso_leaves_p2,
        pso_poseidon2_hash_tree_level_linear_threadgroup: pso_level_p2,
        pso_poseidon2_hash_caps_linear_threadgroup: pso_caps_p2,
        f_poseidon2_hash_leaves_linear_threadgroup: f_leaves_p2,
        f_poseidon2_hash_tree_level_linear_threadgroup: f_level_p2,
        f_poseidon2_hash_caps_linear_threadgroup: f_caps_p2,
        f_quotient_poly: f_quotient,
        pso_quotient_poly: pso_quotient,
    }
});

impl MetalRuntime {
    pub fn get_poseidon2_hash_leaves_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon2_hash_leaves_linear_threadgroup
    }

    pub fn get_poseidon2_hash_tree_level_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon2_hash_tree_level_linear_threadgroup
    }

    pub fn get_poseidon2_hash_caps_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon2_hash_caps_linear_threadgroup
    }

    pub fn get_quotient_poly_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_quotient_poly
    }

    pub fn alloc_aligned_tracked_with_pool_hint(&self, len: usize) -> (TrackedBuffer, bool) {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            return (TrackedBuffer::from_pool(buffer), true);
        }

        let size_class = BufferPool::size_class(aligned_len);
        let buffer = self
            .device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        (TrackedBuffer::new(buffer), false)
    }

    pub fn return_tracked_buffer(&self, tracked: TrackedBuffer) {
        let buffer = tracked.into_inner();
        BUFFER_POOL.put(buffer);
    }

    pub fn alloc_with_data_tracked<T>(&self, data: &[T]) -> TrackedBuffer {
        let len = std::mem::size_of_val(data);
        let buffer = self.device.lock().unwrap().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        TrackedBuffer::new(buffer)
    }

    pub fn alloc_with_data<T>(&self, data: &[T]) -> Buffer {
        let len = std::mem::size_of_val(data);
        track_allocation(len);
        self.device.lock().unwrap().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    #[allow(dead_code)]
    pub fn try_wrap_no_copy<T>(&self, data: &[T]) -> Option<Buffer> {
        let ptr = data.as_ptr() as usize;
        let len = std::mem::size_of_val(data);
        let page_size: usize = 16384;

        if ptr % page_size != 0 {
            return None;
        }

        let aligned_len = (len + page_size - 1) & !(page_size - 1);
        let buffer = self.device.lock().unwrap().new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const std::ffi::c_void,
            aligned_len as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        track_allocation(len);
        Some(buffer)
    }

    #[allow(dead_code)]
    pub fn wrap_or_copy<T>(&self, data: &[T]) -> Buffer {
        if let Some(buf) = self.try_wrap_no_copy(data) {
            buf
        } else {
            self.alloc_with_data(data)
        }
    }

    pub const fn align_to_256(len: usize) -> usize {
        const ALIGNMENT: usize = 256;
        (len + ALIGNMENT - 1) & !(ALIGNMENT - 1)
    }
}

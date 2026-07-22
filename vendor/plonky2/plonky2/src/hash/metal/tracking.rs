//! Buffer allocation tracking for memory analysis.
//!
//! CURRENT_BYTES tracks live allocations (incremented on alloc, decremented on free/return)
//! PEAK_BYTES tracks the maximum CURRENT_BYTES seen
//! ALLOCATION_COUNT tracks total number of allocations

use std::sync::atomic::{AtomicUsize, Ordering};

use metal::Buffer;

static CURRENT_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Get current in-use GPU buffer bytes (allocated minus freed/returned)
#[allow(dead_code)]
pub fn get_current_allocated_bytes() -> usize {
    CURRENT_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

/// Get peak allocated GPU buffer bytes
#[allow(dead_code)]
pub fn get_peak_allocated_bytes() -> usize {
    PEAK_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

/// Get total number of buffer allocations
#[allow(dead_code)]
pub fn get_allocation_count() -> usize {
    ALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Get allocation stats as tuple (current_bytes, peak_bytes, alloc_count)
#[allow(dead_code)]
pub fn get_allocation_stats() -> (usize, usize, usize) {
    (
        CURRENT_ALLOCATED_BYTES.load(Ordering::Relaxed),
        PEAK_ALLOCATED_BYTES.load(Ordering::Relaxed),
        ALLOCATION_COUNT.load(Ordering::Relaxed),
    )
}

/// Reset allocation counters (call before benchmark to get per-run stats)
#[allow(dead_code)]
pub fn reset_allocation_stats() {
    CURRENT_ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    PEAK_ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
}

/// Record a buffer allocation - increments current and updates peak
pub fn track_allocation(bytes: usize) {
    let new_current = CURRENT_ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);

    // Update peak if new current exceeds it
    let mut current_peak = PEAK_ALLOCATED_BYTES.load(Ordering::Relaxed);
    while new_current > current_peak {
        match PEAK_ALLOCATED_BYTES.compare_exchange_weak(
            current_peak,
            new_current,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_peak = actual,
        }
    }
}

/// Record a buffer deallocation - decrements current bytes
pub fn track_deallocation(bytes: usize) {
    CURRENT_ALLOCATED_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// Get allocation stats as a formatted string
#[allow(dead_code)]
pub fn get_allocation_stats_string() -> String {
    let current = get_current_allocated_bytes();
    let peak = get_peak_allocated_bytes();
    let count = get_allocation_count();
    format!(
        "GPU Buffers: {} allocations, {:.2} MB current, {:.2} MB peak",
        count,
        current as f64 / (1024.0 * 1024.0),
        peak as f64 / (1024.0 * 1024.0)
    )
}

/// RAII wrapper for Metal buffers that automatically tracks allocation/deallocation.
/// When a TrackedBuffer is created, it increments the allocation counter.
/// When it's dropped, it decrements the counter.
pub struct TrackedBuffer {
    buffer: Option<Buffer>,
    size: usize,
}

impl TrackedBuffer {
    /// Create a new tracked buffer, recording its allocation
    pub fn new(buffer: Buffer) -> Self {
        let size = buffer.length() as usize;
        track_allocation(size);
        TrackedBuffer {
            buffer: Some(buffer),
            size,
        }
    }

    /// Create a tracked buffer that was retrieved from pool
    /// Re-tracks as "in use" since it's coming out of pool
    pub fn from_pool(buffer: Buffer) -> Self {
        let size = buffer.length() as usize;
        track_allocation(size);
        TrackedBuffer {
            buffer: Some(buffer),
            size,
        }
    }

    /// Get the underlying buffer reference
    pub fn buffer(&self) -> &Buffer {
        self.buffer
            .as_ref()
            .expect("TrackedBuffer already consumed")
    }

    /// Get buffer length
    #[allow(dead_code)]
    pub fn length(&self) -> u64 {
        self.buffer().length()
    }

    /// Get buffer contents pointer
    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.buffer().contents()
    }

    /// Consume and return the inner buffer, tracking deallocation
    pub fn into_inner(mut self) -> Buffer {
        let size = self.size;
        self.size = 0; // Prevent Drop from tracking again
        let buffer = self.buffer.take().expect("TrackedBuffer already consumed");
        track_deallocation(size);
        buffer
    }

    /// Consume and return inner buffer WITHOUT tracking deallocation
    /// Use when buffer will remain in use and tracked elsewhere
    #[allow(dead_code)]
    pub fn into_inner_untracked(mut self) -> Buffer {
        self.size = 0; // Prevent Drop from tracking
        self.buffer.take().expect("TrackedBuffer already consumed")
    }
}

impl Drop for TrackedBuffer {
    fn drop(&mut self) {
        if self.size > 0 {
            track_deallocation(self.size);
        }
    }
}

impl std::ops::Deref for TrackedBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

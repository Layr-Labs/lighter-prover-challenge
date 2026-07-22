//! Buffer pool for reusing Metal buffers to avoid allocation overhead.
//! Groups buffers by size class (power of 2) for efficient reuse.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use metal::Buffer;
use once_cell::sync::Lazy;

/// Buffer pool for reusing Metal buffers to avoid allocation overhead.
/// Groups buffers by size class (power of 2) for efficient reuse.
pub struct BufferPool {
    /// Map from buffer size to list of available buffers
    pools: Mutex<HashMap<usize, Vec<Buffer>>>,
    /// Total bytes currently in pool (for monitoring)
    pool_bytes: AtomicUsize,
    /// Number of pool hits (buffer reused)
    hits: AtomicUsize,
    /// Number of pool misses (new allocation needed)
    misses: AtomicUsize,
}

impl BufferPool {
    pub fn new() -> Self {
        BufferPool {
            pools: Mutex::new(HashMap::new()),
            pool_bytes: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Round size up to next size class (power of 2, minimum 4KB)
    pub fn size_class(size: usize) -> usize {
        const MIN_SIZE: usize = 4096;
        let size = size.max(MIN_SIZE);
        size.next_power_of_two()
    }

    /// Try to get a buffer from the pool
    pub fn get(&self, size: usize) -> Option<Buffer> {
        let size_class = Self::size_class(size);
        let mut pools = self.pools.lock().unwrap();
        if let Some(buffers) = pools.get_mut(&size_class) {
            if let Some(buffer) = buffers.pop() {
                self.pool_bytes.fetch_sub(size_class, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(buffer);
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Return a buffer to the pool for reuse
    pub fn put(&self, buffer: Buffer) {
        let size = buffer.length() as usize;
        let size_class = Self::size_class(size);

        // Limit pool size to prevent unbounded memory growth (max 2GB)
        const MAX_POOL_BYTES: usize = 2 * 1024 * 1024 * 1024;
        if self.pool_bytes.load(Ordering::Relaxed) + size_class > MAX_POOL_BYTES {
            // Drop buffer instead of pooling
            return;
        }

        let mut pools = self.pools.lock().unwrap();
        pools
            .entry(size_class)
            .or_insert_with(Vec::new)
            .push(buffer);
        self.pool_bytes.fetch_add(size_class, Ordering::Relaxed);
    }

    /// Get pool statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.pool_bytes.load(Ordering::Relaxed),
        )
    }

    /// Clear all pooled buffers
    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
        self.pool_bytes.store(0, Ordering::Relaxed);
    }

    /// Reset statistics counters (for per-degree measurement)
    #[allow(dead_code)]
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

pub static BUFFER_POOL: Lazy<BufferPool> = Lazy::new(BufferPool::new);

/// Get buffer pool statistics (hits, misses, bytes_pooled)
#[allow(dead_code)]
pub fn get_buffer_pool_stats() -> (usize, usize, usize) {
    BUFFER_POOL.stats()
}

/// Reset buffer pool statistics (for per-degree measurement)
#[allow(dead_code)]
pub fn reset_buffer_pool_stats() {
    BUFFER_POOL.reset_stats();
}

/// Clear the buffer pool
#[allow(dead_code)]
pub fn clear_buffer_pool() {
    BUFFER_POOL.clear();
}

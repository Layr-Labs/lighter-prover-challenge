//! Dedicated GPU dispatch thread for Metal command buffer operations.
//! Moves all CommandQueue usage to a single thread, fixing thread safety.

use std::sync::mpsc;
use std::thread;

use metal::objc::rc::autoreleasepool;
use metal::Buffer;
use once_cell::sync::Lazy;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::hash_types::HashOut;

type MerkleResult = (Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>);

/// Wrapper asserting Metal `Buffer` is `Send` for `StorageModeShared` on UMA.
pub(crate) struct SendableBuffer(pub Buffer);
unsafe impl Send for SendableBuffer {}

pub(crate) enum GpuJob {
    MerklePoseidon2LinearThreadgroup {
        leaves_buffer: SendableBuffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
        reply: mpsc::Sender<MerkleResult>,
    },
    /// Run an arbitrary closure on the GPU dispatch thread.
    /// Used by the quotient poly dispatch to ensure all Metal operations
    /// happen on the same thread as Merkle (with autoreleasepool).
    RunOnGpuThread {
        work: Box<dyn FnOnce() -> Vec<u64> + Send>,
        reply: mpsc::Sender<Vec<u64>>,
    },
}

pub(crate) struct GpuDispatcher {
    sender: mpsc::Sender<GpuJob>,
}

pub(crate) static GPU_DISPATCHER: Lazy<GpuDispatcher> = Lazy::new(GpuDispatcher::new);

impl GpuDispatcher {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<GpuJob>();

        thread::Builder::new()
            .name("metal-gpu-dispatch".into())
            .spawn(move || {
                autoreleasepool(|| {
                    Self::event_loop(rx);
                });
            })
            .expect("failed to spawn Metal GPU dispatch thread");

        GpuDispatcher { sender: tx }
    }

    fn event_loop(rx: mpsc::Receiver<GpuJob>) {
        use super::runtime::RUNTIME;

        while let Ok(job) = rx.recv() {
            autoreleasepool(|| match job {
                GpuJob::MerklePoseidon2LinearThreadgroup {
                    leaves_buffer,
                    tree_height,
                    leaf_length,
                    cap_height,
                    reply,
                } => {
                    let result = RUNTIME.hash_merkle_tree_poseidon2_linear_threadgroup_buf(
                        leaves_buffer.0,
                        tree_height,
                        leaf_length,
                        cap_height,
                    );
                    let _ = reply.send(result);
                }
                GpuJob::RunOnGpuThread { work, reply } => {
                    let result = work();
                    let _ = reply.send(result);
                }
            });
        }
    }

    pub(crate) fn dispatch_merkle_poseidon2_linear_threadgroup(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> MerkleResult {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(GpuJob::MerklePoseidon2LinearThreadgroup {
                leaves_buffer: SendableBuffer(leaves_buffer),
                tree_height,
                leaf_length,
                cap_height,
                reply: tx,
            })
            .expect("GPU dispatch thread terminated unexpectedly");
        rx.recv()
            .expect("GPU dispatch thread dropped reply channel")
    }

    /// Run an arbitrary closure on the GPU dispatch thread and return the result.
    pub(crate) fn run_on_gpu_thread<F>(&self, work: F) -> Vec<u64>
    where
        F: FnOnce() -> Vec<u64> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(GpuJob::RunOnGpuThread {
                work: Box::new(work),
                reply: tx,
            })
            .expect("GPU dispatch thread terminated unexpectedly");
        rx.recv()
            .expect("GPU dispatch thread dropped reply channel")
    }
}

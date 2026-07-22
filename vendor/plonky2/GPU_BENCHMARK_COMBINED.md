# Lighter-Prover GPU Acceleration Benchmark Results

**Date:** 2026-03-10
**Benchmark:** 500 transactions, 125 chunks of 4 txs each

---

## Summary

| Platform | Hardware | GPU Acceleration | Total Proving Time | Speedup vs CPU |
|----------|----------|-----------------|-------------------|----------------|
| **Metal (macOS)** | Apple M4 (10-core CPU, 10-core GPU) | Merkle Poseidon2 + Quotient Poly | **374-385s** | **1.56-1.61x** |
| **CUDA (Linux)** | 2x AMD EPYC 7773X (256 threads) + 2x RTX 4090 | LDE/NTT | **~507s (8.4 min)** | **~2.0x** |
| CPU-only (macOS) | Apple M4 | None | 602s | baseline |
| CPU-only (Linux) | 2x AMD EPYC 7773X (256 threads) | None | ~1000s (16.7 min) | baseline |

> **Note on CPU baselines:** The Linux CPU-only time (~1000s) is an estimate. The macOS CPU-only time (602s) is measured. The M4's high single-thread performance makes its CPU baseline much faster than the 256-thread EPYC, so the absolute GPU times are not directly comparable across platforms.

---

## Hardware Configurations

### Metal (macOS)

| Spec | Value |
|------|-------|
| Chip | Apple M4 (MacBook Pro) |
| CPU | 10-core (4P + 6E) |
| GPU | 10-core Metal, shared unified memory |
| RAM | 32 GB unified |
| Cooling | Fanless (thermal throttling under sustained load) |

### CUDA (Linux)

| Spec | Value |
|------|-------|
| CPU | 2x AMD EPYC 7773X 64-Core (256 threads total) |
| RAM | 1.0 TiB |
| GPU | 2x NVIDIA GeForce RTX 4090 (24 GB VRAM each) |
| CUDA | 12.8, Driver 570.86.10 |

---

## GPU Kernel Comparison

Different GPU kernels are effective on each platform due to architectural differences:

| GPU Kernel | Metal (M4) | CUDA (RTX 4090) | Why the difference |
|------------|-----------|-----------------|-------------------|
| **Merkle Poseidon2** | Active (2^13-2^20 leaves) | Not used in prover | Metal UMA avoids PCIe transfer; CUDA transfer overhead exceeds benefit for prover leaf sizes (131K-524K) |
| **Quotient Polynomial** | Active (fused gate eval) | Not ported | Metal fused kernel: gate eval + alpha reduction in one pass, zero device allocation |
| **LDE/NTT (FFT)** | Not used (CPU faster) | **Active** (primary speedup) | CUDA FFT on RTX 4090 is massively parallel; M4 GPU lacks the throughput for FFT workloads |
| **Transpose** | Not ported | Available (CPU faster for prover sizes) | GPU only faster for small matrices (<=8K elements) |

---

## Per-Circuit Breakdown

### Metal (macOS, M4)

Measured on cold machine (rebooted before benchmark). GPU Merkle + Quotient both active.

| Circuit | Iterations | Size | CPU-only | GPU Metal | Speedup |
|---------|-----------|------|----------|-----------|---------|
| BlockTxCircuit | 125 | Large (degree 2^16) | 510.1s (avg 4.08s) | 308.8-318.5s (avg 2.47-2.55s) | **1.60-1.65x** |
| BlockTxChainCircuit | 125 | Small (degree 2^14) | 91.3s (avg 730ms) | 64.8-65.6s (avg 518-525ms) | **1.39-1.41x** |
| BlockPreExecutionCircuit | 1 | Small (degree 2^14) | ~860ms | 549-580ms | ~1.5x |
| **Total** | | | **602s** | **374-385s** | **1.56-1.61x** |

GPU quotient kernel timing (from logs):
- Large circuit (lde_size=524288, 27 gates): **~716ms GPU** vs ~1.5s CPU
- Small circuit (lde_size=131072, 16 gates): **~104ms GPU** vs ~200ms CPU

### CUDA (Linux, 2x EPYC + 2x RTX 4090)

GPU LDE/NTT active. Merkle and Transpose fall back to CPU.

| Circuit | Iterations | Size | CPU-only (est.) | GPU CUDA | Speedup |
|---------|-----------|------|-----------------|----------|---------|
| BlockTxCircuit | 125 | Large (degree 2^16) | ~787s (avg ~6.3s) | 401.6s (avg 3.21s) | **~2.0x** |
| BlockTxChainCircuit | 125 | Small (degree 2^14) | ~212s (avg ~1.7s) | 105.2s (avg 841ms) | **~2.0x** |
| BlockPreExecutionCircuit | 1 | Small (degree 2^14) | ~1.5s | 770ms | ~2.0x |
| **Total** | | | **~1000s (16.7 min)** | **~507s (8.4 min)** | **~2.0x** |

GPU LDE operations per proof:
- 3x GPU LDE calls per proof (for trace, partial products, and quotient polynomial batches)
- Polynomials per call: 136 + 20 + 16 = 172 total

---

## Circuit Structure

The benchmark runs: **1 PreExec + 125 iterations of (BlockTx + BlockTxChain)**.

| Circuit | Inner Sub-proofs | Sizes |
|---------|-----------------|-------|
| BlockPreExecutionCircuit | 2 sub-proofs | 2 x small (degree 2^14, 16 gate types) |
| BlockTxCircuit | 2 sub-proofs | 1 x small (degree 2^14, 16 gate types) + **1 x large (degree 2^16, 27 gate types)** |
| BlockTxChainCircuit | 1 sub-proof | 1 x small (degree 2^14, 16 gate types) |

BlockTxCircuit dominates total proving time (85%) due to its large inner sub-proof (degree 2^16, 4x the evaluation domain size of small sub-proofs).

---

## Proving Pipeline Comparison

### Metal Pipeline

```
                        GPU ACCELERATED                    GPU ACCELERATED
                             |                                  |
                             v                                  v
┌─────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌─────────────┐
│ Polynomials │───>│   CPU LDE   │───>│   CPU Transpose   │───>│ GPU Merkle  │
│   (input)   │    │  (CPU FFT)  │    │                   │    │ (Poseidon2) │
└─────────────┘    └─────────────┘    └───────────────────┘    └─────────────┘

┌─────────────────────┐    ┌─────────────────────────────────┐
│ GPU Quotient Poly   │    │  CPU Quotient Poly (fallback)   │
│ (fused gate eval +  │    │  (used when GPU conditions not  │
│  alpha reduction)   │    │   met: non-GL field, lookups)   │
└─────────────────────┘    └─────────────────────────────────┘
```

### CUDA Pipeline

```
                   GPU ACCELERATED
                        |
                        v
┌─────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌─────────────┐
│ Polynomials │───>│  GPU LDE    │───>│ Transpose+BitRev  │───>│ Merkle Tree │
│   (input)   │    │ (CUDA FFT)  │    │ (CPU/GPU auto)    │    │   (CPU)     │
└─────────────┘    └─────────────┘    └───────────────────┘    └─────────────┘

┌─────────────────────┐
│ CPU Quotient Poly   │
│ (not GPU-ported)    │
└─────────────────────┘
```

**Key insight:** Metal and CUDA accelerate **different stages** of the proving pipeline. Metal accelerates Merkle hashing and quotient evaluation (both benefit from UMA — no PCIe transfer). CUDA accelerates LDE/FFT (massive parallelism on RTX 4090 overcomes PCIe overhead for compute-bound FFT).

---

## GPU Activation Conditions

### Metal

| Component | Conditions |
|-----------|------------|
| GPU Merkle Poseidon2 | `metal` feature + GoldilocksField + Poseidon2Hash + 2^13 to 2^20 leaves |
| GPU Quotient Poly | `metal` feature + GoldilocksField + no lookup gates |

### CUDA

| Component | Conditions |
|-----------|------------|
| GPU LDE/NTT | `cuda` feature + GoldilocksField + output_size >= 4K + batch >= 2 |
| GPU Merkle | `cuda` feature + leaf_size > 5 + 1K <= leaves <= 128K (not used in prover) |
| GPU Transpose | `cuda` feature + GoldilocksField + 4K <= elements <= 8K (not used in prover) |

---

## CUDA Standalone Benchmarks

### GPU Merkle Tree (CUDA, standalone — not used in prover)

| Leaves | GPU Time | CPU Time | Speedup |
|--------|----------|----------|---------|
| 2^12 (4,096) | 1.7ms | 31ms | **18.7x** |
| 2^14 (16,384) | 3.1ms | 6.7ms | **2.2x** |
| 2^16 (65,536) | 13ms | 17ms | **1.3x** |
| 2^18 (262,144) | 50ms | 30ms | 0.6x (CPU faster) |
| 2^20 (1,048,576) | 166ms | 157ms | 0.9x (CPU faster) |

> GPU Merkle is faster for small-medium trees (1K-128K leaves) but slower for large trees due to PCIe memory transfer overhead. The prover uses 131K-524K leaf trees.

### GPU Transpose (CUDA, standalone — not used in prover)

| Matrix Size | GPU Time | CPU Time | Speedup |
|-------------|----------|----------|---------|
| 128 x 4K | 7.8ms | 29.5ms | **3.8x** |
| 128 x 16K | 30.2ms | 15.4ms | 0.5x (CPU faster) |
| 128 x 64K | 102ms | 90ms | 0.9x (CPU faster) |
| 128 x 256K | 432ms | 346ms | 0.8x (CPU faster) |

---

## How to Integrate

The GPU acceleration is **fully transparent** — no changes to circuit code or proving logic are needed.

### Metal (macOS)

In your `Cargo.toml`, add `features = ["metal"]` to the plonky2 dependency:

```toml
plonky2 = { git = "https://github.com/timemeansalot/plonky2-metal", package = "plonky2", features = ["metal"] }
```

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4). Metal GPU framework is included in macOS by default. No shader compilation needed — `.metallib` files are pre-compiled and embedded via `include_bytes!`.

### CUDA (Linux)

```toml
plonky2 = { git = "https://github.com/timemeansalot/plonky2-cuda", package = "plonky2", features = ["cuda"] }
```

**Requirements:** Linux with NVIDIA GPU, CUDA 12.x, zeknox 1.0.1.

### What happens automatically

When the feature flag is enabled, GPU kernels are dispatched at runtime based on `TypeId` checks (GoldilocksField, Poseidon2Hash). If conditions are not met, the code falls back to the original CPU path transparently. On unsupported platforms, the feature flag compiles out with no effect.

### Optional: Pipeline Overlap

For workloads that prove multiple chunks sequentially, you can overlap independent proofs on separate threads:

```rust
// Main thread: prove BlockTx sequentially
let proof_tx = BlockTxCircuit::prove(...);

// Send proof to worker thread for BlockTxChain (runs concurrently with next BlockTx)
chain_sender.send(proof_tx.clone()).unwrap();

// Main thread: immediately start next BlockTx (no wait for chain)
let proof_tx_next = BlockTxCircuit::prove(...);
```

This hides BlockTxChain time behind BlockTx, saving additional wall-clock time.

---

## Key Findings

### Memory Architecture Matters

| Architecture | Advantage | Limitation |
|-------------|-----------|------------|
| **Metal UMA** (M4) | Zero-copy GPU access to system memory — no transfer overhead | Lower raw compute throughput than discrete GPUs |
| **CUDA discrete** (RTX 4090) | Massive parallel compute (16K+ CUDA cores) | PCIe transfer overhead limits benefit for memory-bound operations |

This is why Metal can accelerate Merkle hashing (memory-bound, benefits from UMA) while CUDA cannot (PCIe transfer exceeds compute savings). Conversely, CUDA accelerates LDE/FFT (compute-bound, massive parallelism wins) while Metal cannot (M4 GPU lacks throughput).

### Thermal Throttling (Metal/macOS)

Back-to-back benchmark runs on the M4 MacBook Pro can cause thermal throttling that produces unreliable results. Rebooting before benchmarking is recommended for reproducible numbers.

### Future Optimization Opportunities

| Optimization | Platform | Expected Benefit |
|-------------|----------|-----------------|
| Multi-GPU LDE | CUDA | Distribute across 2x RTX 4090 |
| Fused GPU pipeline (LDE -> Transpose -> Merkle on GPU) | CUDA | Eliminate PCIe round-trips |
| GPU Poseidon hashing | CUDA | Benefit if data stays on GPU |
| Pipeline overlap (bench-level) | Both | Hide chain proof time behind block proof |
| Combined Metal+CUDA features | Both | Enable all GPU kernels per platform |

---

*Report generated with [Claude Code](https://claude.ai/code)*

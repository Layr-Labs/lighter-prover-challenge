# 🍎 Metal GPU Benchmark Guide for lighter-prover

## Goal

Run the lighter-prover benchmark **twice** on your Mac and compare:

1. **CPU-only baseline** — upstream plonky2, no GPU acceleration
2. **Metal GPU** — plonky2-metal fork with `features = ["metal"]`, GPU-accelerated Merkle Poseidon2 + Quotient Polynomial

Compare the total proving time and per-circuit averages (BlockTxCircuit, BlockTxChainCircuit) between the two runs. We expect **1.5-1.6x speedup** with Metal GPU on Apple Silicon. Reboot the machine before each run for reproducible cold-start numbers.

## Prerequisites

- **macOS** with **Apple Silicon** (M1/M2/M3/M4)
- **Rust nightly-2025-12-06** (auto-installed via `rust-toolchain` file)
- **Xcode Command Line Tools** (`xcode-select --install`)

## Quick Start

### Step 1: Clone the repos

```bash
# Pick a working directory
mkdir -p ~/coding/plonky2-bench && cd ~/coding/plonky2-bench

# Clone the Metal-enabled plonky2 fork
git clone https://github.com/timemeansalot/plonky2-metal.git

# Clone lighter-prover (requires access to elliottech org)
git clone https://github.com/elliottech/lighter-prover.git
```

### Step 2: Enable Metal in lighter-prover

Edit `lighter-prover/Cargo.toml`:

```toml
# 1. Add features = ["metal"] to the plonky2 dependency (line ~34):
plonky2 = { git = "https://github.com/elliottech/plonky2", rev = "e1c2d35450948b88fca6a7e69e2643c3ecad3caa", package = "plonky2", features = ["metal"] }

# 2. Add [patch] section at the bottom to redirect to the local Metal fork:
[patch."https://github.com/elliottech/plonky2"]
plonky2 = { path = "../plonky2-metal/plonky2" }
```

> **Why the patch?** The upstream `elliottech/plonky2` doesn't have Metal code. The `[patch]` redirects the dependency to your local `plonky2-metal` clone which contains the GPU kernels.

### Step 3: Make sure you have the test data

The benchmark needs `bench_test.json` (~50MB) in the `bench/` directory:

```bash
ls lighter-prover/bench/bench_test.json
```

If the file is missing, ask a teammate to share it (it's not committed to git due to size).

### Step 4: Build and run

```bash
cd lighter-prover/bench

# Option A: Use Makefile
make build-and-run

# Option B: Manual
RUST_LOG=info cargo run --bin bench --release
```

The benchmark takes ~6-7 minutes with GPU, ~10 minutes CPU-only.

### Step 5: Verify GPU is active

Look for these log lines in the output — they confirm Metal GPU kernels are running:

```
quotient dispatch: degree_bits=16, lde_size=524288, is_goldilocks=true, has_lookup=false, num_gates=27
GPU quotient: lde_size=524288, gates=27
GPU quotient: gpu=637ms, total=716ms
```

If you don't see these lines, the GPU path is not active — check that `features = ["metal"]` and `[patch]` are correctly set.

## Expected Results

On Apple M4 (MacBook Pro), cold start:

```
TOTAL BlockPreExecutionCircuit::prove time: ~580ms

TOTAL BlockTxCircuit::prove time:   ~310s
AVERAGE BlockTxCircuit::prove time: ~2.5s

TOTAL BlockTxChainCircuit::prove time: ~65s
AVERAGE BlockTxChainCircuit::prove time: ~520ms
```

CPU-only baseline (without Metal): ~602s total.

## Tips

- **Reboot before benchmarking** for reproducible cold-start numbers. Thermal conditions can affect results.
- **CPU-only baseline** — to run without GPU, edit `lighter-prover/Cargo.toml`:
  1. Remove `features = ["metal"]` from the plonky2 dependency (line ~34)
  2. Remove the entire `[patch."https://github.com/elliottech/plonky2"]` section at the bottom
  3. Rebuild and run:
  ```bash
  RUST_LOG=info cargo run --bin bench --release
  ```
  Compare the total time against the GPU run to measure the speedup.

## Directory Layout

```
~/coding/plonky2-bench/
├── plonky2-metal/          # Metal-enabled plonky2 fork
│   └── plonky2/            # The actual plonky2 crate (patched)
└── lighter-prover/         # Benchmark project
    ├── Cargo.toml           # Edit this to enable Metal
    ├── bench/
    │   ├── bench_test.json  # Test data (~50MB)
    │   ├── src/bin/bench.rs # Benchmark binary
    │   └── Makefile
    └── circuit/             # Circuit definitions
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `error: Metal shaders not found` | Run `xcode-select --install` to install Command Line Tools |
| `Xcode license not accepted` | Run `sudo xcodebuild -license accept` |
| No GPU log lines in output | Verify `features = ["metal"]` and `[patch]` in Cargo.toml |
| Much slower than expected (~5-6s per BlockTx) | Machine is thermally throttled — reboot and try again |
| `bench_test.json not found` | Copy from a teammate or the shared drive |

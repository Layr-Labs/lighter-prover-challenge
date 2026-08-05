# Trusted verifier: what is and is not guaranteed

`lighter-benchmark-verifier` is a prebuilt arm64 executable. It owns the timer,
verifies candidate proofs and their public outputs, and is the only process that
writes a score, so every ranked and local run depends on this file being the
binary the maintainers reviewed. Two checks run against it before any candidate
code is compiled, and they do not prove the same thing.

## The SHA-256 pin is the sole authenticity control

`SHA256SUMS` pins the exact bytes of the published verifier and is the only
control tying this binary to a decision a human made. It is checked by
`setup.sh`, `benchmark.sh`, both workflows, and
`.github/scripts/test-trusted-verifier.sh`.

The pin is only as trustworthy as the commit it lives in: `SHA256SUMS` and the
binary are committed together, so anyone who can land a commit on the default
branch can replace both and every check still passes. What actually protects the
verifier is branch protection, review of any commit touching
`benchmark-tools/trusted/`, and the workflows checking out the trusted default
branch rather than the candidate's ref. Treat any diff here as security-relevant.

## The code signature is tamper-evidence only

`build-trusted-verifier.sh` signs with `codesign --force --sign -` — an **ad-hoc**
signature, reporting `Signature=adhoc` and `TeamIdentifier=not set`. That seals a
hash of the binary's own pages into the binary, so `codesign --verify --strict`
proves the file has not been altered since signing and lets the macOS loader
reject a patched binary at exec time. It proves **nothing** about who produced
it: anyone can ad-hoc sign anything, and an attacker replacing the binary can
re-sign their replacement and still pass.

The step is kept because cheap load-time tamper-evidence is worth having. It is
not an authenticity control, and nothing in this repository should describe it as
one.

## The pinned binary cannot be rebuilt, by anyone

Making the pin independently verifiable would mean rebuilding from the
`REVIEWED_COMMIT` recorded in `build-trusted-verifier.sh` and getting the same
SHA-256. `check-trusted-verifier-reproducibility.sh` attempts exactly that, and
**it fails for a reason that cannot be worked around after the fact.**

The dependency graph contains `const-random 0.1.18` (`circuit`/`plonky2` →
`hashbrown`/`plonky2` → `ahash 0.8.12` → `const-random`), which draws fresh OS
entropy at *compile* time and bakes it into the generated code. On one machine,
one toolchain, identical flags and identical build directory, three builds
minutes apart produced `0426096e…`, `a93ed18d…` and `d624fe48…`. The divergence
starts in `libconst_random`'s rlib and propagates through `ahash`, `hashbrown`,
`plonky2` and `circuit` into `__text`, `__const` and `__cstring`;
`-C codegen-units=1` does not help. The published binary was built without a
fixed seed, so its embedded constants are unrecoverable and the current pin can
never be reproduced, including by its own author. That is why the checker is not
wired into CI: it would fail every run.

Two things are *not* blockers, verified by experiment: the ad-hoc signing step is
deterministic given identical input bytes and basename, and with a fixed
`CONST_RANDOM_SEED` two builds in different build directories were byte-identical.

## Making the next published verifier reproducible

This changes the produced bytes, so it can only take effect the next time the
verifier is legitimately republished and re-reviewed — pair it with any fixture
rotation that already requires republication.

1. Export a fixed, committed `CONST_RANDOM_SEED` in `build-trusted-verifier.sh`.
   On its own this was enough to make two builds byte-identical.
2. Remap the absolute `CARGO_HOME` paths embedded in the binary with
   `--remap-path-prefix`. With a fixed seed plus that remap, two builds under
   different `CARGO_HOME` directories produced the identical binary
   `dfe440d0af481af57a8aa4ca05e6c89068808df0db240fc3f00664fe25de1d8f`.
3. Remap `RUSTUP_HOME` too, or the hash still depends on the builder's home
   directory — 47 distinct `.rustup/toolchains/...` strings survive the
   `CARGO_HOME` remap. **Not verified by experiment**; check before relying on it.
4. Record the exact macOS, Xcode/linker and toolchain versions used, since either
   will also change the bytes.
5. Re-run the checker on a second machine before accepting the new pin, and wire
   it into CI once it is green.

Until that lands, the SHA-256 pin is trusted purely because of review and branch
protection on the commit that introduced it, with no independent way to confirm
the binary corresponds to the source at `REVIEWED_COMMIT`.

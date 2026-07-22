# `metal-lde` extension point

The vendored plonky2 fork already accelerates Poseidon2 Merkle construction and
quotient polynomial evaluation through its `metal` feature.

`metal-lde` is a compile-checked extension point for a future Metal
low-degree-extension implementation. It currently implies `metal` but does not
change proving behavior: LDE remains on the CPU.

The CPU entry points are in `field/src/polynomial/mod.rs`, including
`PolynomialCoeffs::lde`, `lde_onto_coset`, and `coset_fft_with_options`. A GPU
implementation must preserve their field-element output exactly so proofs
remain compatible with the protected CPU verifier.

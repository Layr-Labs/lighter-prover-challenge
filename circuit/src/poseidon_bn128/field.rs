// Portions of this file are derived from succinctx
// Copyright (c) 2023 Succinct Labs
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use ff::PrimeField;

#[derive(PrimeField)]
#[PrimeFieldModulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
pub struct Fr([u64; 4]);

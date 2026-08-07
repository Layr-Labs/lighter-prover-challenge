// Portions of this file are derived from plonky2-keccak256
// Copyright (c) 2023 qope
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use tiny_keccak::{Hasher as KeccakHasher, Keccak};

pub fn keccak(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(input);
    let mut hash = [0u8; 32];
    hasher.finalize(&mut hash);

    hash
}

pub fn u8_array_to_bits(input_u8: &[u8]) -> Vec<bool> {
    input_u8
        .iter()
        .flat_map(|x| u8_to_bits(*x))
        .collect::<Vec<_>>()
}

pub fn u8_to_bits(num: u8) -> Vec<bool> {
    let mut result = Vec::with_capacity(8);
    let mut n = num;
    for _ in 0..8 {
        result.push(n & 1 == 1);
        n >>= 1;
    }
    result
}

pub fn bits_to_u8_array(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8).map(bits_to_u8).collect::<Vec<_>>()
}

pub fn bits_to_u8(bits: &[bool]) -> u8 {
    bits.iter()
        .enumerate()
        .fold(0, |acc, (i, &bit)| acc + ((bit as u8) << i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_conversions() {
        let input_bytes: [u8; 32] = [
            136, 67, 150, 149, 65, 184, 228, 123, 233, 65, 190, 5, 147, 211, 43, 34, 69, 124, 238,
            8, 75, 26, 5, 70, 45, 221, 34, 207, 144, 28, 81, 37,
        ];
        let output = keccak(&input_bytes);

        let expected_output: [u8; 32] = [
            38, 34, 78, 151, 161, 85, 137, 0, 165, 233, 72, 198, 242, 209, 242, 192, 242, 10, 230,
            3, 126, 254, 109, 28, 62, 35, 138, 66, 215, 182, 185, 201,
        ];
        for i in 0..32 {
            assert_eq!(output[i], expected_output[i]);
        }

        let input_bits = u8_array_to_bits(&input_bytes);
        let input_reconstructed = bits_to_u8_array(&input_bits);

        assert_eq!(input_bytes.to_vec(), input_reconstructed);
    }
}

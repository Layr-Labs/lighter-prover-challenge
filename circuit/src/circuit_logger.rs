// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::marker::PhantomData;

use anyhow::Result;
use log::{Level, log};
use num::BigInt;
use num::bigint::Sign;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{IoResult, Read, Write};

use crate::bigint::big_u16::{BigIntU16Target, BigUintU16Target, WitnessBigUintU16};
use crate::bigint::bigint::BigIntTarget;
use crate::bigint::biguint::{BigUintTarget, WitnessBigUint};
use crate::builder::Builder;
use crate::keccak::keccak::KeccakOutputTarget;
use crate::signed::signed_target::{SignedTarget, WitnessSigned};
use crate::uint::u16::gadgets::arithmetic_u16::U16Target;
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::uint::u32::witness::WitnessU32;

#[derive(Debug, Clone, Default)]
pub enum TargetTypes {
    #[default]
    Target,
    TargetArr,
    BoolTarget,
    BigUintTarget,
    BigUintU16Target,
    U32Target,
    SignedTarget,
    BigIntTarget,
    BigIntU16Target,
    KeccakOutputTarget,
}

pub trait CircuitBuilderLogging<F: RichField + Extendable<D>, const D: usize> {
    fn println(&mut self, target: Target, log: &str);
    fn println_arr(&mut self, targets: &[Target], log: &str);
    fn println_bool(&mut self, target: BoolTarget, log: &str);
    fn println_arr_bool(&mut self, targets: &[BoolTarget], log: &str);
    fn println_arr_bool_as_bytes(&mut self, targets: &[BoolTarget], log: &str);
    fn println_biguint(&mut self, target: &BigUintTarget, log: &str);
    fn println_biguint_u16(&mut self, target: &BigUintU16Target, log: &str);
    fn println_bigint(&mut self, target: &BigIntTarget, log: &str);
    fn println_bigint_u16(&mut self, target: &BigIntU16Target, log: &str);
    fn println_hash_out(&mut self, target: &HashOutTarget, log: &str);
    fn println_u32(&mut self, target: U32Target, log: &str);
    fn println_signed_target(&mut self, target: SignedTarget, log: &str);
    fn println_keccak_output(&mut self, target: &KeccakOutputTarget, log: &str);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderLogging<F, D> for Builder<F, D> {
    fn println(&mut self, target: Target, log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: vec![target],
            log,
            target_type: TargetTypes::Target,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_arr(&mut self, targets: &[Target], log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: targets.to_vec(),
            log,
            target_type: TargetTypes::TargetArr,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_bool(&mut self, target: BoolTarget, log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: vec![target.target],
            log,
            target_type: TargetTypes::BoolTarget,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_arr_bool(&mut self, targets: &[BoolTarget], log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: targets.iter().map(|t| t.target).collect::<Vec<Target>>(),
            log,
            target_type: TargetTypes::TargetArr,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_arr_bool_as_bytes(&mut self, targets: &[BoolTarget], log: &str) {
        let log = String::from(log);

        let two = self.two();
        let bits = targets.iter().map(|t| t.target).collect::<Vec<Target>>();
        let bytes = bits
            .chunks(8)
            .map(|chunk| {
                let mut res = chunk[7];
                for i in (0..7).rev() {
                    res = self.mul_add(res, two, chunk[i]);
                }
                res
            })
            .collect::<Vec<_>>();

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: bytes,
            log,
            target_type: TargetTypes::TargetArr,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_biguint(&mut self, variable: &BigUintTarget, log: &str) {
        let variables = variable
            .limbs
            .iter()
            .map(|u32_target| u32_target.0)
            .collect::<Vec<_>>();

        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables,
            log,
            target_type: TargetTypes::BigUintTarget,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_biguint_u16(&mut self, target: &BigUintU16Target, log: &str) {
        let variables = target
            .limbs
            .iter()
            .map(|u16_target| u16_target.0)
            .collect::<Vec<_>>();

        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables,
            log,
            target_type: TargetTypes::BigUintU16Target,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_bigint(&mut self, variable: &BigIntTarget, log: &str) {
        let mut variables = variable
            .abs
            .limbs
            .iter()
            .map(|u32_target| u32_target.0)
            .collect::<Vec<_>>();

        variables.push(variable.sign.target);

        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables,
            log,
            target_type: TargetTypes::BigIntTarget,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_bigint_u16(&mut self, variable: &BigIntU16Target, log: &str) {
        let mut variables = variable
            .abs
            .limbs
            .iter()
            .map(|u16_target| u16_target.0)
            .collect::<Vec<_>>();

        variables.push(variable.sign.target);

        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables,
            log,
            target_type: TargetTypes::BigIntU16Target,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_u32(&mut self, target: U32Target, log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: vec![target.0],
            log,
            target_type: TargetTypes::U32Target,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_signed_target(&mut self, target: SignedTarget, log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: vec![target.target],
            log,
            target_type: TargetTypes::SignedTarget,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }

    fn println_hash_out(&mut self, target: &HashOutTarget, log: &str) {
        self.add_simple_generator(LoggingGenerator {
            variables: target.elements.to_vec(),
            log: String::from(log),
            target_type: TargetTypes::TargetArr,
            _phantom: PhantomData,
        });
    }

    fn println_keccak_output(&mut self, target: &KeccakOutputTarget, log: &str) {
        let log = String::from(log);

        let generator: LoggingGenerator<F, D> = LoggingGenerator {
            variables: target.iter().map(|t| t.0).collect::<Vec<_>>(),
            log,
            target_type: TargetTypes::KeccakOutputTarget,
            _phantom: PhantomData,
        };
        self.add_simple_generator(generator);
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoggingGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub variables: Vec<Target>,
    pub log: String,
    pub target_type: TargetTypes,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for LoggingGenerator<F, D>
{
    fn id(&self) -> String {
        "LoggingGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.variables.clone()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        let log_bytes = self.log.as_bytes();
        dst.write_usize(log_bytes.len())?;
        dst.write_all(log_bytes)?;
        dst.write_usize(self.target_type.clone() as usize)?;
        dst.write_target_vec(&self.variables)
    }

    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Self>
    where
        Self: Sized,
    {
        let log_size = src.read_usize()?;
        let mut log_bytes = vec![0u8; log_size];
        src.read_exact(&mut log_bytes)?;
        let log = String::from_utf8(log_bytes).unwrap();

        let target_type = match src.read_usize()? {
            0 => TargetTypes::Target,
            1 => TargetTypes::TargetArr,
            2 => TargetTypes::BoolTarget,
            3 => TargetTypes::BigUintTarget,
            4 => TargetTypes::BigUintU16Target,
            5 => TargetTypes::U32Target,
            6 => TargetTypes::SignedTarget,
            7 => TargetTypes::BigIntTarget,
            8 => TargetTypes::BigIntU16Target,
            9 => TargetTypes::KeccakOutputTarget,
            _ => panic!("Invalid target type"),
        };

        let variables = src.read_target_vec()?;

        Ok(Self {
            variables,
            log,
            target_type,
            _phantom: PhantomData,
        })
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        _out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        match self.target_type {
            TargetTypes::Target => {
                log!(
                    Level::Info,
                    "[InCircuit] {}: {:?}",
                    self.log,
                    witness.get_target(self.variables[0])
                );
            }
            TargetTypes::TargetArr => {
                log!(
                    Level::Info,
                    "[InCircuit] {}: {:?}",
                    self.log,
                    self.variables
                        .iter()
                        .map(|target| witness.get_target(*target))
                        .collect::<Vec<_>>()
                );
            }
            TargetTypes::BoolTarget => {
                log!(
                    Level::Info,
                    "[InCircuit] {}: {:?}",
                    self.log,
                    witness.get_bool_target(BoolTarget::new_unsafe(self.variables[0]))
                );
            }
            TargetTypes::BigUintTarget => {
                let big_uint_target = BigUintTarget {
                    limbs: self
                        .variables
                        .iter()
                        .map(|target| U32Target(*target))
                        .collect(),
                };
                let biguint = witness.get_biguint_target(big_uint_target);
                log!(Level::Info, "[InCircuit] {}: {:?}", self.log, biguint);
            }
            TargetTypes::BigUintU16Target => {
                let big_uint_target = BigUintU16Target {
                    limbs: self
                        .variables
                        .iter()
                        .map(|target| U16Target(*target))
                        .collect(),
                };
                let biguint = witness.get_biguint_u16_target(big_uint_target);
                log!(Level::Info, "[InCircuit] {}: {:?}", self.log, biguint);
            }
            TargetTypes::U32Target => {
                log!(
                    Level::Info,
                    "[InCircuit] {}: {:?}",
                    self.log,
                    witness.get_u32_target(U32Target(self.variables[0]))
                );
            }
            TargetTypes::SignedTarget => {
                log!(
                    Level::Info,
                    "[InCircuit] {}: {:?}",
                    self.log,
                    witness.get_signed_target(SignedTarget::new_unsafe(self.variables[0])),
                );
            }
            TargetTypes::BigIntTarget => {
                let mut variables = self.variables.clone();
                let sign = variables.pop().unwrap();

                let big_uint_target = BigUintTarget {
                    limbs: variables.iter().map(|target| U32Target(*target)).collect(),
                };
                let biguint = witness.get_biguint_target(big_uint_target);
                let sign = {
                    let t = witness.get_target(sign);
                    if t == F::ONE {
                        Sign::Plus
                    } else if t == F::ZERO {
                        Sign::NoSign
                    } else if t == F::NEG_ONE {
                        Sign::Minus
                    } else {
                        panic!("Invalid Sign!")
                    }
                };

                let bigint = BigInt::from_biguint(sign, biguint);
                log!(Level::Info, "[InCircuit] {}: {:?}", self.log, bigint);
            }
            TargetTypes::BigIntU16Target => {
                let mut variables = self.variables.clone();
                let sign = variables.pop().unwrap();

                let big_uint_target = BigUintU16Target {
                    limbs: variables.iter().map(|target| U16Target(*target)).collect(),
                };
                let biguint = witness.get_biguint_u16_target(big_uint_target);
                let sign = {
                    let t = witness.get_target(sign);
                    if t == F::ONE {
                        Sign::Plus
                    } else if t == F::ZERO {
                        Sign::NoSign
                    } else if t == F::NEG_ONE {
                        Sign::Minus
                    } else {
                        panic!("Invalid Sign!")
                    }
                };

                let bigint = BigInt::from_biguint(sign, biguint);
                log!(Level::Info, "[InCircuit] {}: {:?}", self.log, bigint);
            }
            TargetTypes::KeccakOutputTarget => {
                let bytes = self
                    .variables
                    .iter()
                    .map(|target| {
                        u8::try_from(witness.get_target(*target).to_canonical_u64()).unwrap()
                    })
                    .collect::<Vec<_>>();

                log!(
                    Level::Info,
                    "[InCircuit] {}: {:?}",
                    self.log,
                    hex::encode(bytes),
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
    use num::BigUint;
    use plonky2::field::types::Field64;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use crate::bigint::bigint::{BigIntTarget, SignTarget};
    use crate::bigint::biguint::CircuitBuilderBiguint;
    use crate::builder::Builder;
    use crate::circuit_logger::CircuitBuilderLogging;
    use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

    #[test]
    #[ignore]
    fn test_logger() -> Result<()> {
        let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let pw: PartialWitness<F> = PartialWitness::new();
        let mut builder = Builder::<F, D>::new(config);

        let x = builder.constant(F::from_canonical_i64(11));
        let y = builder.constant(F::from_canonical_i64(22));
        let z = builder.add(x, y);
        builder.println(z, "I am logging z target");

        let x = builder.constant_u32(11);
        let y = builder.constant_u32(22);
        let (z, _) = builder.add_u32(x, y);
        builder.println_u32(z, "I am logging z u32");

        let x = builder.constant_bool(false);
        let y = builder.constant_bool(true);
        let z = builder.and(x, y);
        builder.println_bool(z, "I am logging z bool");

        let x = builder.constant_biguint(&BigUint::from(11u64));
        let y = builder.constant_biguint(&BigUint::from(22u64));
        let z = builder.add_biguint(&x, &y);
        builder.println_biguint(&z, "I am logging z biguint");

        let pos_z = BigIntTarget {
            abs: z.clone(),
            sign: SignTarget::new_unsafe(builder.one()),
        };
        let neg_z = BigIntTarget {
            abs: z.clone(),
            sign: SignTarget::new_unsafe(builder.neg_one()),
        };
        builder.println_bigint(&pos_z, "I am logging pos_z bigint");
        builder.println_bigint(&neg_z, "I am logging neg_z biguint");

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}

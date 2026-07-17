// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::uint::u16::gadgets::arithmetic_u16::U16Target;

pub trait WriteU16 {
    fn write_target_u16(&mut self, x: U16Target) -> IoResult<()>;
}

impl WriteU16 for Vec<u8> {
    #[inline]
    fn write_target_u16(&mut self, x: U16Target) -> IoResult<()> {
        self.write_target(x.0)
    }
}

pub trait ReadU16 {
    fn read_target_u16(&mut self) -> IoResult<U16Target>;
}

impl ReadU16 for Buffer<'_> {
    #[inline]
    fn read_target_u16(&mut self) -> IoResult<U16Target> {
        Ok(U16Target(self.read_target()?))
    }
}

use core::marker::PhantomData;

use crate::field::packed::PackedField;

/// Writes constraints yielded by a gate to a buffer, with a given stride.
/// Permits us to abstract the underlying memory layout. In particular, we can make a matrix of
/// constraints where every column is an evaluation point and every row is a constraint index, with
/// the matrix stored in row-contiguous form.
#[derive(Debug)]
pub struct StridedConstraintConsumer<'a, P: PackedField, const ACCUMULATE: bool = false> {
    // This is a particularly neat way of doing this, more so than a slice. We increase start by
    // stride at every step and terminate when it equals end.
    start: *mut P::Scalar,
    end: *mut P::Scalar,
    stride: usize,
    accumulate_filter: P,
    _phantom: PhantomData<&'a mut [P::Scalar]>,
}

impl<'a, P: PackedField> StridedConstraintConsumer<'a, P, false> {
    pub fn new(buffer: &'a mut [P::Scalar], stride: usize, offset: usize) -> Self {
        assert!(stride >= P::WIDTH);
        assert!(offset < stride);
        assert_eq!(buffer.len() % stride, 0);
        let ptr_range = buffer.as_mut_ptr_range();
        // `wrapping_add` is needed to avoid undefined behavior. Plain `add` causes UB if 'the ...
        // resulting pointer [is neither] in bounds or one byte past the end of the same allocated
        // object'; the UB results even if the pointer is not dereferenced. `end` will be more than
        // one byte past the buffer unless `offset` is 0. The same applies to `start` if the buffer
        // has length 0 and the offset is not 0.
        // We _could_ do pointer arithmetic without `wrapping_add`, but the logic would be
        // unnecessarily complicated.
        let start = ptr_range.start.wrapping_add(offset);
        let end = ptr_range.end.wrapping_add(offset);
        Self {
            start,
            end,
            stride,
            accumulate_filter: P::ZEROS,
            _phantom: PhantomData,
        }
    }
}

impl<'a, P: PackedField> StridedConstraintConsumer<'a, P, true> {
    /// Creates a consumer which applies the point filter while emitting each constraint,
    /// accumulating directly into the destination matrix.
    pub fn new_accumulating(
        buffer: &'a mut [P::Scalar],
        stride: usize,
        offset: usize,
        filter: P,
    ) -> Self {
        assert!(stride >= P::WIDTH);
        assert!(offset + P::WIDTH <= stride);
        assert_eq!(buffer.len() % stride, 0);
        let ptr_range = buffer.as_mut_ptr_range();
        let start = ptr_range.start.wrapping_add(offset);
        let end = ptr_range.end.wrapping_add(offset);
        Self {
            start,
            end,
            stride,
            accumulate_filter: filter,
            _phantom: PhantomData,
        }
    }
}

impl<'a, P: PackedField, const ACCUMULATE: bool>
    StridedConstraintConsumer<'a, P, ACCUMULATE>
{
    /// Emit one constraint.
    pub fn one(&mut self, constraint: P) {
        if !core::ptr::eq(self.start, self.end) {
            // # Safety
            // The checks in `new` guarantee that this points to valid space.
            unsafe {
                let slot = &mut *self.start.cast::<P>();
                if ACCUMULATE {
                    *slot = slot.multiply_accumulate(constraint, self.accumulate_filter);
                } else {
                    *slot = constraint;
                }
            }
            // See the comment in `new`. `wrapping_add` is needed to avoid UB if we've just
            // exhausted our buffer (and hence we're setting `self.start` to point past the end).
            self.start = self.start.wrapping_add(self.stride);
        } else {
            panic!("gate produced too many constraints");
        }
    }

    /// Convenience method that calls `.one()` multiple times.
    pub fn many<I: IntoIterator<Item = P>>(&mut self, constraints: I) {
        constraints
            .into_iter()
            .for_each(|constraint| self.one(constraint));
    }
}

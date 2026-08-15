use core::marker::PhantomData;

use crate::field::packed::PackedField;

/// Writes constraints yielded by a gate to a buffer, with a given stride.
/// Permits us to abstract the underlying memory layout. In particular, we can make a matrix of
/// constraints where every column is an evaluation point and every row is a constraint index, with
/// the matrix stored in row-contiguous form.
#[derive(Debug)]
pub struct StridedConstraintConsumer<'a, P: PackedField> {
    // This is a particularly neat way of doing this, more so than a slice. We increase start by
    // stride at every step and terminate when it equals end.
    start: *mut P::Scalar,
    end: *mut P::Scalar,
    stride: usize,
    filter: Option<P>,
    _phantom: PhantomData<&'a mut [P::Scalar]>,
}

impl<'a, P: PackedField> StridedConstraintConsumer<'a, P> {
    pub fn new(buffer: &'a mut [P::Scalar], stride: usize, offset: usize) -> Self {
        Self::with_filter(buffer, stride, offset, None)
    }

    /// Creates a consumer which accumulates every emitted constraint into the
    /// existing row with the supplied selector filter.
    ///
    /// The accumulator uses the same `multiply_accumulate` operation as the
    /// materialize-then-add path. Keeping it in the consumer lets packed gates
    /// write directly into the quotient buffer instead of a transient scratch
    /// matrix.
    pub fn accumulating(
        buffer: &'a mut [P::Scalar],
        stride: usize,
        offset: usize,
        filter: P,
    ) -> Self {
        Self::with_filter(buffer, stride, offset, Some(filter))
    }

    fn with_filter(
        buffer: &'a mut [P::Scalar],
        stride: usize,
        offset: usize,
        filter: Option<P>,
    ) -> Self {
        assert!(stride >= P::WIDTH);
        assert!(offset <= stride - P::WIDTH);
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
            filter,
            _phantom: PhantomData,
        }
    }

    /// Emit one constraint.
    pub fn one(&mut self, constraint: P) {
        if !core::ptr::eq(self.start, self.end) {
            // # Safety
            // The checks in `with_filter` guarantee that this points to valid
            // space for a complete packed row. Copying through scalar pointers
            // avoids assuming a row-major scalar allocation has P alignment.
            unsafe {
                if let Some(filter) = self.filter {
                    let mut accumulated = P::ZEROS;
                    core::ptr::copy_nonoverlapping(
                        self.start,
                        accumulated.as_slice_mut().as_mut_ptr(),
                        P::WIDTH,
                    );
                    core::ptr::copy_nonoverlapping(
                        accumulated.multiply_accumulate(constraint, filter).as_slice().as_ptr(),
                        self.start,
                        P::WIDTH,
                    );
                } else {
                    core::ptr::copy_nonoverlapping(
                        constraint.as_slice().as_ptr(),
                        self.start,
                        P::WIDTH,
                    );
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

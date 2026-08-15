use core::marker::PhantomData;

use crate::field::packed::PackedField;

/// Writes constraints yielded by a gate to a buffer, with a given stride.
/// Permits us to abstract the underlying memory layout. In particular, we can make a matrix of
/// constraints where every column is an evaluation point and every row is a constraint index, with
/// the matrix stored in row-contiguous form.
///
/// In accumulating mode (`new_accumulating`) the consumer does not overwrite the destination row;
/// instead it loads the packed row, performs `multiply_accumulate(constraint, filter)` in place,
/// and writes the result back. This removes the transient scratch matrix for packed gates whose
/// constraints are emitted directly into the quotient accumulator.
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

    /// Create a consumer that accumulates `constraint * filter` into the strided destination
    /// rows rather than overwriting them. The filter is packed for the current lane group; each
    /// emitted packed constraint is multiplied by this filter and added to the row already at
    /// `buffer[offset + row * stride ..]`.
    pub fn new_accumulating(
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
            filter,
            _phantom: PhantomData,
        }
    }

    /// Emit one constraint.
    pub fn one(&mut self, constraint: P) {
        if !core::ptr::eq(self.start, self.end) {
            if let Some(filter) = self.filter {
                // Load the packed destination row through scalar pointers, accumulate the
                // filtered constraint, and write the packed result back. This avoids a strict
                // alignment requirement on the row-major scalar storage: the packed value is
                // assembled from and written to a `P::WIDTH`-wide scalar slice.
                unsafe {
                    let slice = core::slice::from_raw_parts(self.start, P::WIDTH);
                    let acc = *P::from_slice(slice);
                    let result = acc.multiply_accumulate(constraint, filter);
                    let slice = core::slice::from_raw_parts_mut(self.start, P::WIDTH);
                    slice.copy_from_slice(result.as_slice());
                }
            } else {
                // Safety
                // The checks in `new` guarantee that this points to valid space.
                unsafe {
                    *self.start.cast() = constraint;
                }
            }
            // See the comment in `new`. `wrapping_add` is needed to avoid UB if we've just
            // exhausted our buffer (and hence we are setting `self.start` to point past the end).
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

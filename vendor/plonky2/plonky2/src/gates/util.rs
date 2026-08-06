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
    /// `None` stores each constraint, overwriting the destination. `Some(f)`
    /// folds it in as `dest += f * constraint`, which lets a caller accumulate
    /// filtered constraints straight into the shared quotient buffer instead of
    /// materializing the full `batch_size * num_constraints` matrix first.
    ///
    /// The selector filter is constant for a given evaluation point, so it
    /// lives here rather than being reapplied per constraint, and the mode is
    /// fixed for the consumer's whole lifetime.
    filter: Option<P>,
    _phantom: PhantomData<&'a mut [P::Scalar]>,
}

impl<'a, P: PackedField> StridedConstraintConsumer<'a, P> {
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
            filter: None,
            _phantom: PhantomData,
        }
    }

    /// Like [`Self::new`], but each emitted constraint is folded into the
    /// destination as `dest += filter * constraint` instead of overwriting it.
    ///
    /// The destination must already be initialized (it is the shared quotient
    /// accumulator, which the caller zeroes once per batch), and `stride` /
    /// `offset` mean exactly what they do for [`Self::new`], so the emission
    /// order and layout a gate produces are unchanged.
    pub fn new_accumulating(
        buffer: &'a mut [P::Scalar],
        stride: usize,
        offset: usize,
        filter: P,
    ) -> Self {
        let mut this = Self::new(buffer, stride, offset);
        this.filter = Some(filter);
        this
    }

    /// Emit one constraint.
    #[inline]
    pub fn one(&mut self, constraint: P) {
        if !core::ptr::eq(self.start, self.end) {
            // # Safety
            // The checks in `new` guarantee that this points to valid space. In
            // the accumulating mode we also read the slot first, which is sound
            // because `new_accumulating`'s contract requires an initialized
            // destination.
            unsafe {
                let slot = self.start.cast::<P>();
                match self.filter {
                    None => *slot = constraint,
                    Some(filter) => *slot = *slot + filter * constraint,
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

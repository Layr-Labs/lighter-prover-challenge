#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::ops::Index;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }
}

/// A row-major matrix of Merkle leaves stored in a single contiguous allocation.
///
/// The prover materializes one of these per polynomial commitment, with one row per LDE point and
/// one column per committed polynomial. Storing the rows contiguously (rather than as a
/// `Vec<Vec<F>>`) keeps the transposed LDE in a single allocation, lets the GPU backend consume the
/// leaves as a flat `u64` buffer without a per-row gather, and makes `get_lde_values` a slice index
/// into hot, sequentially laid out memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeafMatrix<F> {
    data: Vec<F>,
    width: usize,
    /// Tracked explicitly: a zero-width matrix still has a meaningful row count, which
    /// `data.len() / width` cannot express.
    rows: usize,
}

impl<F> Default for LeafMatrix<F> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            rows: 0,
        }
    }
}

impl<F> LeafMatrix<F> {
    /// Wraps `data` as a `rows x width` row-major matrix, with `width > 0`.
    pub fn new(data: Vec<F>, width: usize) -> Self {
        assert!(width > 0, "use `with_rows` for a zero-width matrix");
        debug_assert_eq!(data.len() % width, 0);
        let rows = data.len() / width;
        Self { data, width, rows }
    }

    /// Wraps `data` as a row-major matrix with an explicit row count.
    pub fn with_rows(data: Vec<F>, width: usize, rows: usize) -> Self {
        debug_assert_eq!(data.len(), rows * width);
        Self { data, width, rows }
    }

    pub const fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    /// The backing row-major buffer.
    pub fn as_flat(&self) -> &[F] {
        &self.data
    }

    pub fn as_slice(&self) -> LeafSlice<'_, F> {
        LeafSlice {
            data: &self.data,
            width: self.width,
            rows: self.rows,
        }
    }

    pub fn row(&self, index: usize) -> &[F] {
        &self.data[index * self.width..(index + 1) * self.width]
    }
}

impl<F: Clone> LeafMatrix<F> {
    /// Builds a matrix from individual rows. Only used by tests and by callers outside the hot
    /// commitment path; the prover builds the flat buffer directly.
    pub fn from_rows(rows: Vec<Vec<F>>) -> Self {
        Self::from_row_slices(&rows)
    }

    /// Like [`Self::from_rows`], but borrows the rows.
    pub fn from_row_slices(rows: &[Vec<F>]) -> Self {
        let width = rows.first().map_or(0, Vec::len);
        debug_assert!(rows.iter().all(|row| row.len() == width));
        let mut data = Vec::with_capacity(rows.len() * width);
        for row in rows {
            data.extend_from_slice(row);
        }
        Self {
            data,
            width,
            rows: rows.len(),
        }
    }
}

impl<F> Index<usize> for LeafMatrix<F> {
    type Output = [F];

    fn index(&self, index: usize) -> &[F] {
        self.row(index)
    }
}

/// A borrowed view of a contiguous run of rows of a [`LeafMatrix`].
#[derive(Copy, Clone, Debug)]
pub struct LeafSlice<'a, F> {
    data: &'a [F],
    width: usize,
    rows: usize,
}

impl<'a, F> LeafSlice<'a, F> {
    pub const fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub fn row(&self, index: usize) -> &'a [F] {
        &self.data[index * self.width..(index + 1) * self.width]
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.data.split_at(mid * self.width);
        (
            Self {
                data: left,
                width: self.width,
                rows: mid,
            },
            Self {
                data: right,
                width: self.width,
                rows: self.rows - mid,
            },
        )
    }
}

impl<F> Index<usize> for LeafSlice<'_, F> {
    type Output = [F];

    fn index(&self, index: usize) -> &[F] {
        self.row(index)
    }
}

impl<'a, F: Send + Sync> LeafSlice<'a, F> {
    /// Splits into `rows_per_chunk`-row chunks, in parallel. Mirrors `par_chunks_exact`.
    pub fn par_chunks_exact(
        &self,
        rows_per_chunk: usize,
    ) -> impl IndexedParallelIterator<Item = LeafSlice<'a, F>> {
        let width = self.width;
        let data: &'a [F] = self.data;
        // With zero-width rows there is no backing data to split, so chunk the row count instead.
        let stride = rows_per_chunk * width;
        let chunks = if width == 0 {
            self.rows / rows_per_chunk
        } else {
            data.len() / stride
        };
        (0..chunks).into_par_iter().map(move |chunk| LeafSlice {
            data: if width == 0 {
                &data[..0]
            } else {
                &data[chunk * stride..(chunk + 1) * stride]
            },
            width,
            rows: rows_per_chunk,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree, one row per leaf.
    pub leaves: LeafMatrix<F>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: LeafMatrix::default(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: LeafSlice<'_, F>,
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: LeafSlice<'_, F>,
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves.par_chunks_exact(1))
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(&leaf[0]));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = LeafSlice::par_chunks_exact(&leaves, subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves: LeafMatrix<F>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        if let Some((digests, cap)) = H::try_build_merkle_tree(&leaves, cap_height) {
            debug_assert_eq!(digests.len(), 2 * (leaves.len() - (1 << cap_height)));
            debug_assert_eq!(cap.len(), 1 << cap_height);
            return Self {
                leaves,
                digests,
                cap: MerkleCap(cap),
            };
        }

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves.as_slice(), cap_height);

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i]
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.leaves.len(), cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(LeafMatrix::from_row_slices(&leaves), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(LeafMatrix::from_rows(leaves), cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}

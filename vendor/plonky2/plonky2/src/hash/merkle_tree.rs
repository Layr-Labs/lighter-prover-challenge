#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
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

/// Natural-order poly-major column storage, either CPU-owned or retained in a
/// CPU-visible GPU buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ColumnStore<F> {
    Owned(Vec<Vec<F>>),
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    Shared(crate::hash::poseidon2::metal::MetalColumns<F>),
}

impl<F: RichField> ColumnStore<F> {
    pub fn num_cols(&self) -> usize {
        match self {
            ColumnStore::Owned(columns) => columns.len(),
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.cols(),
        }
    }

    pub fn num_rows(&self) -> usize {
        match self {
            ColumnStore::Owned(columns) => columns.first().map_or(0, Vec::len),
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.rows(),
        }
    }

    pub fn col(&self, j: usize) -> &[F] {
        match self {
            ColumnStore::Owned(columns) => &columns[j],
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.col(j),
        }
    }
}

/// Backing storage for the Merkle tree leaves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MerkleLeaves<F> {
    /// One flat row-major buffer: leaf `i` occupies `data[i * width..(i + 1) * width]`.
    Rows { data: Vec<F>, width: usize },
    /// Natural-order poly-major columns: leaf `i` holds
    /// `columns.col(j)[reverse_bits(i, log_rows)]` for each column `j`. This is
    /// the layout LDEs are produced in, so committing to them requires no
    /// transpose.
    Columns {
        columns: ColumnStore<F>,
        log_rows: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: MerkleLeaves<F>,

    /// The number of leaves.
    pub num_leaves: usize,

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
            leaves: MerkleLeaves::Rows {
                data: Vec::new(),
                width: 0,
            },
            num_leaves: 0,
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
    leaves: &[Vec<F>],
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
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
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

pub(crate) fn fill_subtree_flat<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_width: usize,
    num_leaves: usize,
) -> H::Hash {
    debug_assert_eq!(num_leaves, digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(leaves)
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        let half = num_leaves / 2;
        let (left_leaves, right_leaves) = leaves.split_at(half * leaf_width);

        // Sibling leaves are independent; hash them as one interleaved pair so
        // the two permutation dependency chains overlap in the pipeline.
        if num_leaves == 2 {
            let (left_digest, right_digest) = H::hash_or_noop_pair(left_leaves, right_leaves);
            left_digest_mem.write(left_digest);
            right_digest_mem.write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // Same idea one level up: hash the four leaves as one interleaved
        // quad, then compress the two sibling parent nodes as a pair.
        if num_leaves == 4 {
            let (leaf_0, leaf_1) = left_leaves.split_at(leaf_width);
            let (leaf_2, leaf_3) = right_leaves.split_at(leaf_width);
            let (h0, h1, h2, h3) = H::hash_or_noop_quad(leaf_0, leaf_1, leaf_2, leaf_3);
            left_digests_buf[0].write(h0);
            left_digests_buf[1].write(h1);
            right_digests_buf[0].write(h2);
            right_digests_buf[1].write(h3);
            let (left_digest, right_digest) = H::two_to_one_pair(h0, h1, h2, h3);
            left_digest_mem.write(left_digest);
            right_digest_mem.write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // Keep leaf permutations at the accepted four-state boundary, but pair
        // the two independent four-leaf roots before computing the final root.
        if num_leaves == 8 {
            let leaf = |i| &leaves[i * leaf_width..(i + 1) * leaf_width];
            let (h0, h1, h2, h3) =
                H::hash_or_noop_quad(leaf(0), leaf(1), leaf(2), leaf(3));
            digests_buf[0].write(h0);
            digests_buf[1].write(h1);
            digests_buf[4].write(h2);
            digests_buf[5].write(h3);

            let (p0, p1) = H::two_to_one_pair(h0, h1, h2, h3);
            digests_buf[2].write(p0);
            digests_buf[3].write(p1);

            let (h4, h5, h6, h7) =
                H::hash_or_noop_quad(leaf(4), leaf(5), leaf(6), leaf(7));
            digests_buf[8].write(h4);
            digests_buf[9].write(h5);
            digests_buf[12].write(h6);
            digests_buf[13].write(h7);

            let (p2, p3) = H::two_to_one_pair(h4, h5, h6, h7);
            digests_buf[10].write(p2);
            digests_buf[11].write(p3);

            let (left_digest, right_digest) = H::two_to_one_pair(p0, p1, p2, p3);
            digests_buf[6].write(left_digest);
            digests_buf[7].write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // Rayon task creation dominates the tiny subtrees near the leaves. Keep
        // enough parallelism at the upper levels, then recurse synchronously.
        let (left_digest, right_digest) = if num_leaves > 16 {
            plonky2_maybe_rayon::join(
                || fill_subtree_flat::<F, H>(left_digests_buf, left_leaves, leaf_width, half),
                || fill_subtree_flat::<F, H>(right_digests_buf, right_leaves, leaf_width, half),
            )
        } else {
            (
                fill_subtree_flat::<F, H>(left_digests_buf, left_leaves, leaf_width, half),
                fill_subtree_flat::<F, H>(right_digests_buf, right_leaves, leaf_width, half),
            )
        };

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf_flat<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_width: usize,
    num_leaves: usize,
    cap_height: usize,
) {
    // Special case of a tree that's all cap.
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), num_leaves);
        cap_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cap_buf)| {
                cap_buf.write(H::hash_or_noop(&leaves[i * leaf_width..(i + 1) * leaf_width]));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = num_leaves >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    digests_chunks.zip(cap_buf).enumerate().for_each(
        |(subtree_index, (subtree_digests, subtree_cap))| {
            let leaf_start = subtree_index * subtree_leaves_len * leaf_width;
            let leaf_end = (subtree_index + 1) * subtree_leaves_len * leaf_width;
            subtree_cap.write(fill_subtree_flat::<F, H>(
                subtree_digests,
                &leaves[leaf_start..leaf_end],
                leaf_width,
                subtree_leaves_len,
            ));
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
    /// Build a tree from per-leaf vectors. All leaves must have the same width.
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let num_leaves = leaves.len();
        let leaf_width = leaves.first().map_or(0, Vec::len);
        debug_assert!(
            leaves.iter().all(|leaf| leaf.len() == leaf_width),
            "all leaves must have the same width"
        );
        let mut flat = Vec::with_capacity(num_leaves * leaf_width);
        for leaf in &leaves {
            flat.extend_from_slice(leaf);
        }
        Self::from_flat_parts(flat, leaf_width, num_leaves, cap_height)
    }

    /// Build a tree from one flat row-major buffer of `leaves.len() / leaf_width` leaves.
    pub fn new_flat(leaves: Vec<F>, leaf_width: usize, cap_height: usize) -> Self {
        assert!(leaf_width > 0, "flat construction requires nonzero width");
        let num_leaves = leaves.len() / leaf_width;
        assert_eq!(leaves.len(), num_leaves * leaf_width);
        Self::from_flat_parts(leaves, leaf_width, num_leaves, cap_height)
    }

    /// Build a tree directly from the natural-order poly-major LDE columns,
    /// without materializing the transposed leaf matrix. Leaf `i` is
    /// `columns[j][reverse_bits(i, log_rows)]`.
    pub fn new_columns(columns: Vec<Vec<F>>, cap_height: usize) -> Self {
        let num_leaves = columns.first().map_or(0, Vec::len);
        debug_assert!(columns.iter().all(|column| column.len() == num_leaves));
        let log_rows = log2_strict(num_leaves);
        assert!(
            cap_height <= log_rows,
            "cap_height={cap_height} should be at most log2(leaves.len())={log_rows}"
        );

        if let Some((digests, cap)) = H::try_build_merkle_tree_columns(&columns, cap_height) {
            debug_assert_eq!(digests.len(), 2 * (num_leaves - (1 << cap_height)));
            debug_assert_eq!(cap.len(), 1 << cap_height);
            return Self {
                leaves: MerkleLeaves::Columns {
                    columns: ColumnStore::Owned(columns),
                    log_rows,
                },
                num_leaves,
                digests,
                cap: MerkleCap(cap),
            };
        }

        // CPU fallback: materialize the bit-reversed row-major matrix and hash it.
        let flat = crate::util::transpose_to_bitrev_flat(&columns);
        let (digests, cap) =
            Self::cpu_digests(&flat, columns.len(), num_leaves, cap_height);
        Self {
            leaves: MerkleLeaves::Columns {
                columns: ColumnStore::Owned(columns),
                log_rows,
            },
            num_leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    /// Wraps an already-hashed column store (e.g. from the fused GPU NTT +
    /// Merkle pipeline) into a tree.
    pub fn from_prebuilt_columns(
        columns: ColumnStore<F>,
        digests: Vec<H::Hash>,
        cap: Vec<H::Hash>,
    ) -> Self {
        let num_leaves = columns.num_rows();
        let log_rows = log2_strict(num_leaves);
        debug_assert_eq!(digests.len(), 2 * (num_leaves - cap.len()));
        Self {
            leaves: MerkleLeaves::Columns { columns, log_rows },
            num_leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    fn cpu_digests(
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> (Vec<H::Hash>, Vec<H::Hash>) {
        let num_digests = 2 * (num_leaves - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf_flat::<F, H>(
            digests_buf,
            cap_buf,
            leaves,
            leaf_width,
            num_leaves,
            cap_height,
        );

        unsafe {
            // SAFETY: `fill_digests_buf_flat` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        (digests, cap)
    }

    fn from_flat_parts(
        leaves: Vec<F>,
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> Self {
        let log2_leaves_len = log2_strict(num_leaves);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        if let Some((digests, cap)) =
            H::try_build_merkle_tree(&leaves, leaf_width, num_leaves, cap_height)
        {
            debug_assert_eq!(digests.len(), 2 * (num_leaves - (1 << cap_height)));
            debug_assert_eq!(cap.len(), 1 << cap_height);
            return Self {
                leaves: MerkleLeaves::Rows {
                    data: leaves,
                    width: leaf_width,
                },
                num_leaves,
                digests,
                cap: MerkleCap(cap),
            };
        }

        let (digests, cap) = Self::cpu_digests(&leaves, leaf_width, num_leaves, cap_height);
        Self {
            leaves: MerkleLeaves::Rows {
                data: leaves,
                width: leaf_width,
            },
            num_leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    /// The number of field elements per leaf.
    pub fn leaf_width(&self) -> usize {
        match &self.leaves {
            MerkleLeaves::Rows { width, .. } => *width,
            MerkleLeaves::Columns { columns, .. } => columns.num_cols(),
        }
    }

    /// Borrow leaf `i`. Only available for row-major storage.
    pub fn get(&self, i: usize) -> &[F] {
        match &self.leaves {
            MerkleLeaves::Rows { data, width } => &data[i * width..(i + 1) * width],
            MerkleLeaves::Columns { .. } => {
                panic!("MerkleTree::get is unavailable for column-major leaves")
            }
        }
    }

    /// Copy leaf `i` out of either storage layout.
    pub fn leaf_vec(&self, i: usize) -> Vec<F> {
        match &self.leaves {
            MerkleLeaves::Rows { data, width } => data[i * width..(i + 1) * width].to_vec(),
            MerkleLeaves::Columns { columns, log_rows } => {
                let natural = crate::util::reverse_bits(i, *log_rows);
                (0..columns.num_cols())
                    .map(|j| columns.col(j)[natural])
                    .collect()
            }
        }
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.num_leaves, cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::PrimeField64;
    use crate::hash::hash_types::HashOut;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::poseidon2::hash::{Poseidon2Hash, Poseidon2Permutation};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    static PAIR_COMPRESSIONS: AtomicUsize = AtomicUsize::new(0);
    static SCALAR_COMPRESSIONS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct CountingPoseidon2Hash;

    impl Hasher<GoldilocksField> for CountingPoseidon2Hash {
        const HASH_SIZE: usize = <Poseidon2Hash as Hasher<GoldilocksField>>::HASH_SIZE;
        type Hash = HashOut<GoldilocksField>;
        type Permutation = Poseidon2Permutation<GoldilocksField>;

        fn hash_no_pad(input: &[GoldilocksField]) -> Self::Hash {
            <Poseidon2Hash as Hasher<GoldilocksField>>::hash_no_pad(input)
        }

        fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
            SCALAR_COMPRESSIONS.fetch_add(1, Ordering::Relaxed);
            <Poseidon2Hash as Hasher<GoldilocksField>>::two_to_one(left, right)
        }

        fn two_to_one_pair(
            x0: Self::Hash,
            y0: Self::Hash,
            x1: Self::Hash,
            y1: Self::Hash,
        ) -> (Self::Hash, Self::Hash) {
            PAIR_COMPRESSIONS.fetch_add(1, Ordering::Relaxed);
            <Poseidon2Hash as Hasher<GoldilocksField>>::two_to_one_pair(x0, y0, x1, y1)
        }
    }

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    #[test]
    fn eight_leaf_flat_tree_pairs_level_two_and_matches_legacy_layout() {
        type F = GoldilocksField;
        const NUM_LEAVES: usize = 8;
        const NUM_DIGESTS: usize = 2 * (NUM_LEAVES - 1);

        for width in [1, 4, 5, 8, 9, 17, 87] {
            let leaves = random_data::<F>(NUM_LEAVES, width);
            let flat = leaves.iter().flatten().copied().collect::<Vec<_>>();

            let mut expected_digests = Vec::with_capacity(NUM_DIGESTS);
            let expected_root = fill_subtree::<F, Poseidon2Hash>(
                capacity_up_to_mut(&mut expected_digests, NUM_DIGESTS),
                &leaves,
            );
            unsafe { expected_digests.set_len(NUM_DIGESTS) };

            PAIR_COMPRESSIONS.store(0, Ordering::Relaxed);
            SCALAR_COMPRESSIONS.store(0, Ordering::Relaxed);
            let mut actual_digests = Vec::with_capacity(NUM_DIGESTS);
            let actual_root = fill_subtree_flat::<F, CountingPoseidon2Hash>(
                capacity_up_to_mut(&mut actual_digests, NUM_DIGESTS),
                &flat,
                width,
                NUM_LEAVES,
            );
            unsafe { actual_digests.set_len(NUM_DIGESTS) };

            assert_eq!(
                actual_root.elements.map(|value| value.to_noncanonical_u64()),
                expected_root.elements.map(|value| value.to_noncanonical_u64()),
                "root mismatch at width {width}"
            );
            assert_eq!(
                actual_digests
                    .iter()
                    .flat_map(|digest| {
                        digest
                            .elements
                            .iter()
                            .map(|value| value.to_noncanonical_u64())
                    })
                    .collect::<Vec<_>>(),
                expected_digests
                    .iter()
                    .flat_map(|digest| {
                        digest
                            .elements
                            .iter()
                            .map(|value| value.to_noncanonical_u64())
                    })
                    .collect::<Vec<_>>(),
                "digest layout mismatch at width {width}"
            );
            assert_eq!(PAIR_COMPRESSIONS.load(Ordering::Relaxed), 3);
            assert_eq!(SCALAR_COMPRESSIONS.load(Ordering::Relaxed), 1);
        }
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
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
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
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

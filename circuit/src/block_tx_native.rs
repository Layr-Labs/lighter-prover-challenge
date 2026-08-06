// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Native (out-of-circuit) evaluation of the jump-state transition and of the
//! pre-execution outputs the prover orchestration needs before any proof exists.
//!
//! Nothing in this module touches circuit definitions, targets or witness
//! generation; it only recomputes, natively, values that the circuits already
//! compute and expose as public inputs. The prover uses these to cut data
//! dependencies between proofs (all leaf tx proofs can start immediately) and
//! asserts every native value against the corresponding proof's public inputs
//! as soon as that proof exists.

use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::config::Hasher;

use crate::block::Block;
use crate::block_tx::JumpState;
use crate::poseidon2::Poseidon2Hash;
use crate::tx::Tx;
use crate::types::config::F;
use crate::types::constants::TX_TYPE_EMPTY;
use crate::types::state_metadata::StateMetadata;

/// Post-state roots of one active tx slot, recovered from the fixture's global
/// `tx_index` chaining: active txs thread the state/delta roots in `tx_index`
/// order across both circuit paths, so an active tx's new roots are the next
/// active tx's `old_state_root`/`old_account_delta_tree_root`, and the block's
/// `new_state_root`/`new_account_delta_tree_root` for the block's last active
/// tx. Padding (empty) slots never read their entry.
#[derive(Clone, Copy, Debug)]
pub struct TxEndRoots {
    pub new_state_root: HashOut<F>,
    pub new_delta_root: HashOut<F>,
}

/// Recovers [`TxEndRoots`] for every tx slot of every chunk (see the struct
/// docs for the chaining argument). Entries for padding slots are `None`.
/// Panics if active tx indices are not unique — such a fixture cannot satisfy
/// the block circuit's jump cross-checks anyway.
pub fn chunk_end_roots(block: &Block<F>) -> Vec<Vec<Option<TxEndRoots>>> {
    end_roots_from_chunks(
        &block.tx_chunks,
        block.new_state_root,
        block.new_account_delta_tree_root,
    )
}

/// [`chunk_end_roots`] on bare chunks (the only block fields the recovery
/// reads are the final roots).
pub fn end_roots_from_chunks(
    tx_chunks: &[Vec<Tx<F>>],
    block_new_state_root: HashOut<F>,
    block_new_delta_root: HashOut<F>,
) -> Vec<Vec<Option<TxEndRoots>>> {
    // (tx_index, chunk index, slot index) of every active tx, in global order.
    let mut active: Vec<(u64, usize, usize)> = tx_chunks
        .iter()
        .enumerate()
        .flat_map(|(chunk_index, txs)| {
            txs.iter()
                .enumerate()
                .filter(|(_, tx)| tx.tx_type != TX_TYPE_EMPTY)
                .map(move |(slot_index, tx)| (tx.tx_index, chunk_index, slot_index))
        })
        .collect();
    active.sort_unstable_by_key(|&(tx_index, _, _)| tx_index);

    let mut end_roots: Vec<Vec<Option<TxEndRoots>>> = tx_chunks
        .iter()
        .map(|txs| vec![None; txs.len()])
        .collect();
    for position in 0..active.len() {
        let (tx_index, chunk_index, slot_index) = active[position];
        let roots = if let Some(&(next_tx_index, next_chunk, next_slot)) = active.get(position + 1) {
            assert!(
                next_tx_index > tx_index,
                "active txs must have unique tx indices, found #{tx_index} twice"
            );
            let next_tx = &tx_chunks[next_chunk][next_slot];
            TxEndRoots {
                new_state_root: next_tx.old_state_root,
                new_delta_root: next_tx.old_account_delta_tree_root,
            }
        } else {
            TxEndRoots {
                new_state_root: block_new_state_root,
                new_delta_root: block_new_delta_root,
            }
        };
        end_roots[chunk_index][slot_index] = Some(roots);
    }
    end_roots
}

/// Pre-execution outputs recomputed natively from the block witness, i.e.
/// without waiting for the pre-execution proof.
///
/// * `new_state_metadata` mirrors the circuit's metadata update exactly
///   (`block_pre_execution_constraints.rs`, `define_block_pre_execution`): each
///   timestamp is replaced by `created_at` iff the corresponding `calculate_*`
///   flag is set (the circuit itself constrains those flags against the
///   timestamps, so the witness flags are authoritative for a provable block).
/// * `new_state_root` is recovered from the fixture rather than recomputed:
///   the first active tx executes directly on the pre-execution output state,
///   so its `old_state_root` IS that root; a block with no active txs leaves
///   the state untouched, so the block's `new_state_root` is the pre-execution
///   output (both facts are enforced in-circuit by the chain/block jump
///   checks). The prover asserts this value against the pre-execution proof's
///   public inputs once that proof completes.
#[derive(Clone, Debug)]
pub struct NativePreOutput {
    pub new_state_metadata: StateMetadata,
    pub state_metadata_hash: HashOut<F>,
    pub new_state_root: HashOut<F>,
}

pub fn native_pre_output(block: &Block<F>) -> NativePreOutput {
    // Mirrors the three `builder.select(calculate_*, created_at, old)` at the
    // end of `define_block_pre_execution`.
    let new_state_metadata = StateMetadata {
        last_funding_round_timestamp: if block.calculate_funding {
            block.created_at
        } else {
            block.state_metadata.last_funding_round_timestamp
        },
        last_oracle_price_timestamp: if block.calculate_oracle_prices {
            block.created_at
        } else {
            block.state_metadata.last_oracle_price_timestamp
        },
        last_premium_timestamp: if block.calculate_premium {
            block.created_at
        } else {
            block.state_metadata.last_premium_timestamp
        },
    };
    let state_metadata_hash = new_state_metadata.hash();

    let new_state_root = pre_state_root_from_chunks(&block.tx_chunks, block.new_state_root);

    NativePreOutput {
        new_state_metadata,
        state_metadata_hash,
        new_state_root,
    }
}

/// The pre-execution output state root, recovered from the chunks: the first
/// active tx (minimum global `tx_index`) executes directly on it, so its
/// `old_state_root` IS that root; with no active txs the block leaves the
/// state untouched and the block's own `new_state_root` is the answer.
pub fn pre_state_root_from_chunks(
    tx_chunks: &[Vec<Tx<F>>],
    block_new_state_root: HashOut<F>,
) -> HashOut<F> {
    tx_chunks
        .iter()
        .flatten()
        .filter(|tx| tx.tx_type != TX_TYPE_EMPTY)
        .min_by_key(|tx| tx.tx_index)
        .map_or(block_new_state_root, |first_active| {
            first_active.old_state_root
        })
}

impl JumpState<F> {
    /// Native mirror of the in-circuit jump transition over one chunk: exactly
    /// what `BlockTxCircuit::define_tx_loop` folds via `define_jump_step`
    /// (`block_tx_constraints.rs`), evaluated on the witness values the fixture
    /// supplies. `end_roots[i]` must be `Some` for every active `txs[i]` (see
    /// [`chunk_end_roots`]). Panics — with context — wherever the circuit would
    /// be unsatisfiable, so a broken fixture dies here instead of wasting a
    /// proof chain.
    #[must_use]
    pub fn advance(&self, txs: &[Tx<F>], end_roots: &[Option<TxEndRoots>]) -> JumpState<F> {
        assert_eq!(
            txs.len(),
            end_roots.len(),
            "one end-roots entry per tx slot required"
        );
        let mut jump = *self;
        for (tx, roots) in txs.iter().zip(end_roots) {
            jump = jump.step(tx, roots);
        }
        jump
    }

    /// One `define_jump_step`, natively. Comments cite the mirrored constraint.
    fn step(&self, tx: &Tx<F>, end_roots: &Option<TxEndRoots>) -> JumpState<F> {
        let one = F::ONE;
        // set_tx_target writes `from_canonical_u64(tx.tx_index)`; padding txs
        // carry the canonical encoding of the path's last active index
        // (`F::NEG_ONE` before any active tx), assigned in `Block::chunk_txs`.
        let tx_index = F::from_canonical_u64(tx.tx_index);

        // is_padding = is_equal(tx_index, jump.last_active_tx_index)
        let is_padding = tx_index == self.last_active_tx_index;
        let is_active = !is_padding;
        // connect(is_empty, is_padding) with is_empty = is_zero(tx_type)
        let is_empty = tx.tx_type == TX_TYPE_EMPTY;
        assert_eq!(
            is_empty, is_padding,
            "tx #{} (type {}) violates the padding rule: empty={is_empty} but padding={is_padding} \
             (last active index {:?})",
            tx.tx_index, tx.tx_type, self.last_active_tx_index
        );

        // gap / effective_gap / gap_minus_one, with the 32-bit range check.
        let gap = tx_index - self.last_active_tx_index;
        let effective_gap = if is_padding { one } else { gap };
        let gap_minus_one = effective_gap - one;
        assert!(
            gap_minus_one.to_canonical_u64() < (1 << 32),
            "tx #{} jumps backwards or too far: gap-1 = {:?} exceeds 32 bits",
            tx.tx_index,
            gap_minus_one
        );
        let has_gap = gap_minus_one != F::ZERO;
        let jumped = is_active && has_gap;
        let continuous = is_active && !has_gap;

        // conditional_assert_eq(continuous, tx old roots, jump prev roots)
        if continuous {
            assert_eq!(
                tx.old_state_root, self.prev_new_state_root,
                "tx #{} is continuous but its old state root does not extend the jump state",
                tx.tx_index
            );
            assert_eq!(
                tx.old_account_delta_tree_root, self.prev_new_delta_root,
                "tx #{} is continuous but its old delta root does not extend the jump state",
                tx.tx_index
            );
        }

        // Coverage fold: hash_n_to_hash_no_pad over
        // [run_start_prev_index, last_active + 1, run_start_old_state_root,
        //  prev_new_state_root, run_start_old_delta_root, prev_new_delta_root]
        // then hash_two_to_one(coverage_hash, record); selected by
        // fold_coverage = jumped AND run_nonempty.
        let run_nonempty = self.last_active_tx_index != self.run_start_prev_index;
        let coverage_hash = if jumped && run_nonempty {
            let mut record_in = Vec::with_capacity(18);
            record_in.push(self.run_start_prev_index);
            record_in.push(self.last_active_tx_index + one);
            record_in.extend(self.run_start_old_state_root.elements);
            record_in.extend(self.prev_new_state_root.elements);
            record_in.extend(self.run_start_old_delta_root.elements);
            record_in.extend(self.prev_new_delta_root.elements);
            let record = Poseidon2Hash::hash_no_pad(&record_in);
            Poseidon2Hash::two_to_one(self.coverage_hash, record)
        } else {
            self.coverage_hash
        };

        // Claims fold: hash_n_to_hash_no_pad over
        // [last_active, tx_index, prev_new_state_root, tx.old_state_root,
        //  prev_new_delta_root, tx.old_delta_root]
        // then hash_two_to_one(claims_hash, record); selected by `jumped`.
        let claims_hash = if jumped {
            let mut record_in = Vec::with_capacity(18);
            record_in.push(self.last_active_tx_index);
            record_in.push(tx_index);
            record_in.extend(self.prev_new_state_root.elements);
            record_in.extend(tx.old_state_root.elements);
            record_in.extend(self.prev_new_delta_root.elements);
            record_in.extend(tx.old_account_delta_tree_root.elements);
            let record = Poseidon2Hash::hash_no_pad(&record_in);
            Poseidon2Hash::two_to_one(self.claims_hash, record)
        } else {
            self.claims_hash
        };

        // In-circuit, prev_new_*_root select the tx's computed post-state roots
        // when active; natively those are the fixture-recovered end roots.
        let (prev_new_state_root, prev_new_delta_root) = if is_active {
            let roots = end_roots.unwrap_or_else(|| {
                panic!(
                    "active tx #{} has no natively recovered end roots",
                    tx.tx_index
                )
            });
            (roots.new_state_root, roots.new_delta_root)
        } else {
            (self.prev_new_state_root, self.prev_new_delta_root)
        };

        JumpState {
            last_active_tx_index: if is_active {
                tx_index
            } else {
                self.last_active_tx_index
            },
            prev_new_state_root,
            prev_new_delta_root,
            run_start_prev_index: if jumped {
                tx_index - one
            } else {
                self.run_start_prev_index
            },
            run_start_old_state_root: if jumped {
                tx.old_state_root
            } else {
                self.run_start_old_state_root
            },
            run_start_old_delta_root: if jumped {
                tx.old_account_delta_tree_root
            } else {
                self.run_start_old_delta_root
            },
            coverage_hash,
            claims_hash,
            tx_count: self.tx_count + if is_active { one } else { F::ZERO },
        }
    }
}

/// Synthetic ACTIVE-transaction scenarios, shaped exactly like the chunks
/// `Block::chunk_txs` produces for a valid fixture: global `tx_index` order
/// 0..n across both paths, state/delta roots threaded globally through every
/// active tx, per-path chunking at the pinned widths (heavy 4 / light 10) with
/// empty-tx padding only as a tail of each path's last chunk (or one full
/// padding chunk at the `NEG_ONE` sentinel for a path with no txs). Tests use
/// these to exercise the native jump transition on shapes the all-empty smoke
/// fixture cannot reach.
#[cfg(test)]
pub(crate) mod testkit {
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::hash::hash_types::HashOut;
    use plonky2::plonk::config::Hasher;

    use super::TxEndRoots;
    use crate::block::Block;
    use crate::poseidon2::Poseidon2Hash;
    use crate::tx::Tx;
    use crate::types::config::F;
    use crate::types::constants::{TX_HEAVY, TX_LIGHT, TX_TYPE_EMPTY};

    pub(crate) const HEAVY_WIDTH: usize = 4;
    pub(crate) const LIGHT_WIDTH: usize = 10;
    /// Any nonzero tx type: the jump transition only distinguishes empty (0)
    /// from non-empty.
    pub(crate) const ACTIVE_TX_TYPE: u8 = 7;

    /// One tx slot plus the natively recovered end roots (`None` for padding),
    /// i.e. exactly what `JumpState::advance` consumes.
    pub(crate) struct ScenarioTx {
        pub tx: Tx<F>,
        pub end_roots: Option<TxEndRoots>,
    }

    pub(crate) struct Scenario {
        pub initial_state_root: HashOut<F>,
        pub initial_delta_root: HashOut<F>,
        pub final_state_root: HashOut<F>,
        pub final_delta_root: HashOut<F>,
        pub heavy_chunks: Vec<Vec<ScenarioTx>>,
        pub light_chunks: Vec<Vec<ScenarioTx>>,
    }

    /// A real empty tx from the public smoke fixture — the padding template,
    /// exactly as `Block::chunk_txs` would clone it.
    pub(crate) fn empty_tx_template() -> Tx<F> {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let block = Block::<F>::from_json(
                    include_bytes!("../../bench/bench_test.json"),
                    HEAVY_WIDTH,
                    LIGHT_WIDTH,
                )
                .expect("public smoke fixture must parse");
                block
                    .tx_chunks
                    .into_iter()
                    .flatten()
                    .find(|tx| tx.tx_type == TX_TYPE_EMPTY)
                    .expect("smoke fixture must contain an empty padding tx")
            })
            .expect("fixture parse thread must start")
            .join()
            .expect("fixture parse thread must finish")
    }

    /// Deterministic distinct roots for the global state/delta chains.
    fn chain_root(tag: u64, position: u64) -> HashOut<F> {
        Poseidon2Hash::hash_no_pad(&[F::from_canonical_u64(tag), F::from_canonical_u64(position)])
    }

    pub(crate) fn state_root(position: u64) -> HashOut<F> {
        chain_root(0x517A7E, position)
    }

    pub(crate) fn delta_root(position: u64) -> HashOut<F> {
        chain_root(0xDE17A, position)
    }

    /// Builds the scenario for a global path schedule: `paths[k]` is the
    /// circuit type (`TX_HEAVY`/`TX_LIGHT`) of global active tx `k`.
    pub(crate) fn build(paths: &[u8]) -> Scenario {
        let template = empty_tx_template();
        let total = paths.len() as u64;

        let mut heavy: Vec<ScenarioTx> = Vec::new();
        let mut light: Vec<ScenarioTx> = Vec::new();
        let mut last_heavy_index = F::NEG_ONE.to_canonical_u64();
        let mut last_light_index = F::NEG_ONE.to_canonical_u64();
        for (position, &circuit_type) in paths.iter().enumerate() {
            let position = position as u64;
            let mut tx = template.clone();
            tx.tx_type = ACTIVE_TX_TYPE;
            tx.tx_circuit_type = circuit_type;
            tx.tx_index = position;
            tx.old_state_root = state_root(position);
            tx.old_account_delta_tree_root = delta_root(position);
            let scenario_tx = ScenarioTx {
                tx,
                end_roots: Some(TxEndRoots {
                    new_state_root: state_root(position + 1),
                    new_delta_root: delta_root(position + 1),
                }),
            };
            if circuit_type == TX_LIGHT {
                last_light_index = position;
                light.push(scenario_tx);
            } else {
                last_heavy_index = position;
                heavy.push(scenario_tx);
            }
        }

        Scenario {
            initial_state_root: state_root(0),
            initial_delta_root: delta_root(0),
            final_state_root: state_root(total),
            final_delta_root: delta_root(total),
            heavy_chunks: chunk(heavy, HEAVY_WIDTH, TX_HEAVY, last_heavy_index, &template),
            light_chunks: chunk(light, LIGHT_WIDTH, TX_LIGHT, last_light_index, &template),
        }
    }

    /// Mirrors `Block::chunk_txs` for one path: emit full chunks, pad the last
    /// incomplete chunk with the empty template carrying the path's last
    /// active index; a path with no txs still contributes one full padding
    /// chunk at the `NEG_ONE` sentinel.
    fn chunk(
        txs: Vec<ScenarioTx>,
        width: usize,
        circuit_type: u8,
        last_index: u64,
        template: &Tx<F>,
    ) -> Vec<Vec<ScenarioTx>> {
        let mut chunks: Vec<Vec<ScenarioTx>> = Vec::new();
        let mut buf: Vec<ScenarioTx> = Vec::new();
        let had_txs = !txs.is_empty();
        for tx in txs {
            buf.push(tx);
            if buf.len() == width {
                chunks.push(std::mem::take(&mut buf));
            }
        }
        if buf.is_empty() && had_txs {
            return chunks;
        }
        let mut pad = template.clone();
        pad.tx_circuit_type = circuit_type;
        pad.tx_index = last_index;
        while buf.len() < width {
            buf.push(ScenarioTx {
                tx: pad.clone(),
                end_roots: None,
            });
        }
        chunks.push(buf);
        chunks
    }

    pub(crate) fn chunk_txs_and_ends(chunk: &[ScenarioTx]) -> (Vec<Tx<F>>, Vec<Option<TxEndRoots>>) {
        (
            chunk.iter().map(|slot| slot.tx.clone()).collect(),
            chunk.iter().map(|slot| slot.end_roots).collect(),
        )
    }

    /// Global schedules used by both the native and the in-circuit oracle
    /// tests. Shapes covered (none reachable from the all-empty smoke run):
    /// * `interleaved`: multi-chunk on both paths, runs starting/continuing/
    ///   ending, coverage+claims folds onto nonzero accumulators, both paths'
    ///   last chunks tail-padded, heavy path starting mid-block (first-chunk
    ///   `NEG_ONE` sentinel with a jump).
    /// * `all_light`: pure continuous run on light (no jumps at all), heavy
    ///   path entirely absent (full padding chunk at the sentinel).
    /// * `alternating`: every active tx jumps (gap 2) — maximum fold density,
    ///   single-tx runs on both paths, heavy chunks at an exact multiple (no
    ///   tail padding).
    /// * `big_gap`: a wide (26) index gap on the heavy path.
    pub(crate) fn interleaved_paths() -> Vec<u8> {
        let heavy_at = [2usize, 7, 10, 11, 16, 22, 23, 28, 30];
        (0..31)
            .map(|position| {
                if heavy_at.contains(&position) {
                    TX_HEAVY
                } else {
                    TX_LIGHT
                }
            })
            .collect()
    }

    pub(crate) fn all_light_paths() -> Vec<u8> {
        vec![TX_LIGHT; 12]
    }

    pub(crate) fn alternating_paths() -> Vec<u8> {
        (0..16)
            .map(|position| if position % 2 == 0 { TX_LIGHT } else { TX_HEAVY })
            .collect()
    }

    pub(crate) fn big_gap_paths() -> Vec<u8> {
        let mut paths = vec![TX_LIGHT; 27];
        paths[0] = TX_HEAVY;
        paths[26] = TX_HEAVY;
        paths
    }

    pub(crate) fn all_scenario_paths() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("interleaved", interleaved_paths()),
            ("all_light", all_light_paths()),
            ("alternating", alternating_paths()),
            ("big_gap", big_gap_paths()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::HashOut;
    use plonky2::plonk::config::Hasher;

    use super::testkit::{self, ScenarioTx};
    use super::*;
    use crate::block_tx::JumpState;

    /// Interleave the two paths' chunk lists the way `Block::chunk_txs` emits
    /// them (a chunk appears as soon as it fills), so the recovery has to sort
    /// active txs globally rather than rely on chunk order.
    fn interleave_chunks(scenario: &testkit::Scenario) -> Vec<Vec<Tx<F>>> {
        let mut chunks: Vec<(u64, Vec<Tx<F>>)> = Vec::new();
        for chunk in scenario.heavy_chunks.iter().chain(&scenario.light_chunks) {
            let emit_order = chunk
                .iter()
                .filter_map(|slot| slot.end_roots.map(|_| slot.tx.tx_index))
                .max()
                .unwrap_or(u64::MAX);
            chunks.push((
                emit_order,
                chunk.iter().map(|slot| slot.tx.clone()).collect(),
            ));
        }
        chunks.sort_by_key(|&(emit_order, _)| emit_order);
        chunks.into_iter().map(|(_, txs)| txs).collect()
    }

    #[test]
    fn end_roots_recovery_matches_scenario_construction() {
        for (name, paths) in testkit::all_scenario_paths() {
            let scenario = testkit::build(&paths);
            let chunks = interleave_chunks(&scenario);
            let recovered = end_roots_from_chunks(
                &chunks,
                scenario.final_state_root,
                scenario.final_delta_root,
            );

            // Match each recovered entry back to the scenario slot by tx_index.
            let mut expected: std::collections::HashMap<u64, TxEndRoots> = Default::default();
            for slot in scenario
                .heavy_chunks
                .iter()
                .chain(&scenario.light_chunks)
                .flatten()
            {
                if let Some(end_roots) = slot.end_roots {
                    expected.insert(slot.tx.tx_index, end_roots);
                }
            }
            for (chunk, chunk_roots) in chunks.iter().zip(&recovered) {
                for (tx, recovered_roots) in chunk.iter().zip(chunk_roots) {
                    match (tx.tx_type == TX_TYPE_EMPTY, recovered_roots) {
                        (true, None) => {}
                        (false, Some(roots)) => {
                            let want = expected[&tx.tx_index];
                            assert_eq!(
                                roots.new_state_root, want.new_state_root,
                                "{name}: state root of active tx #{}",
                                tx.tx_index
                            );
                            assert_eq!(
                                roots.new_delta_root, want.new_delta_root,
                                "{name}: delta root of active tx #{}",
                                tx.tx_index
                            );
                        }
                        (is_empty, roots) => panic!(
                            "{name}: tx #{} empty={is_empty} but recovery returned {roots:?}",
                            tx.tx_index
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn pre_state_root_recovery_picks_first_active_globally() {
        let scenario = testkit::build(&testkit::interleaved_paths());
        let mut chunks = interleave_chunks(&scenario);
        // Move a later chunk first: recovery must key on tx_index, not layout.
        chunks.rotate_right(1);
        let block_new_state_root = scenario.final_state_root;
        assert_eq!(
            pre_state_root_from_chunks(&chunks, block_new_state_root),
            scenario.initial_state_root,
        );
        // No active txs at all ⇒ the block's new state root.
        let all_empty = testkit::build(&[]);
        let empty_chunks = interleave_chunks(&all_empty);
        assert_eq!(
            pre_state_root_from_chunks(&empty_chunks, block_new_state_root),
            block_new_state_root,
        );
    }

    /// Native mirror of `BlockCircuit::finalize_jump`
    /// (block_constraints.rs:292-338): closes the open coverage run and folds
    /// the trailing claim for the chain that did not observe the block's end.
    fn finalize_native(
        jump: &JumpState<F>,
        total_tx_count: F,
        new_state_root: HashOut<F>,
        new_delta_root: HashOut<F>,
    ) -> (HashOut<F>, HashOut<F>) {
        let prev_plus_one = jump.last_active_tx_index + F::ONE;
        let run_nonempty = jump.last_active_tx_index != jump.run_start_prev_index;
        let coverage_hash = if run_nonempty {
            let mut record_in = Vec::with_capacity(18);
            record_in.push(jump.run_start_prev_index);
            record_in.push(prev_plus_one);
            record_in.extend(jump.run_start_old_state_root.elements);
            record_in.extend(jump.prev_new_state_root.elements);
            record_in.extend(jump.run_start_old_delta_root.elements);
            record_in.extend(jump.prev_new_delta_root.elements);
            let record = Poseidon2Hash::hash_no_pad(&record_in);
            Poseidon2Hash::two_to_one(jump.coverage_hash, record)
        } else {
            jump.coverage_hash
        };

        let at_end = prev_plus_one == total_tx_count;
        let claims_hash = if !at_end {
            let mut record_in = Vec::with_capacity(18);
            record_in.push(jump.last_active_tx_index);
            record_in.push(total_tx_count);
            record_in.extend(jump.prev_new_state_root.elements);
            record_in.extend(new_state_root.elements);
            record_in.extend(jump.prev_new_delta_root.elements);
            record_in.extend(new_delta_root.elements);
            let record = Poseidon2Hash::hash_no_pad(&record_in);
            Poseidon2Hash::two_to_one(jump.claims_hash, record)
        } else {
            jump.claims_hash
        };

        (coverage_hash, claims_hash)
    }

    fn advance_path(initial: JumpState<F>, chunks: &[Vec<ScenarioTx>]) -> JumpState<F> {
        let mut jump = initial;
        for chunk in chunks {
            let (txs, end_roots) = testkit::chunk_txs_and_ends(chunk);
            jump = jump.advance(&txs, &end_roots);
        }
        jump
    }

    /// Runs the native transition over both paths of every scenario, then
    /// checks the block circuit's cross-path jump equations
    /// (`define_jump_checks`, block_constraints.rs:343-412) on the results.
    /// The scenarios are valid-fixture shapes by construction, so any native
    /// fold/select/ordering bug shows up here as a cross-hash mismatch — the
    /// exact equations the final block proof would fail.
    #[test]
    fn native_advance_satisfies_block_level_cross_checks() {
        for (name, paths) in testkit::all_scenario_paths() {
            let scenario = testkit::build(&paths);
            let initial =
                JumpState::initial(scenario.initial_state_root, scenario.initial_delta_root);
            let heavy = advance_path(initial, &scenario.heavy_chunks);
            let light = advance_path(initial, &scenario.light_chunks);

            let total_tx_count = heavy.tx_count + light.tx_count;
            assert_eq!(
                total_tx_count,
                F::from_canonical_usize(paths.len()),
                "{name}: active tx count"
            );

            // At least one chain ends the block; its exposed roots must be the
            // block's final roots.
            let heavy_at_end = heavy.last_active_tx_index + F::ONE == total_tx_count;
            let light_at_end = light.last_active_tx_index + F::ONE == total_tx_count;
            assert!(heavy_at_end || light_at_end, "{name}: no chain at block end");
            let end_jump = if heavy_at_end { &heavy } else { &light };
            assert_eq!(
                end_jump.prev_new_state_root, scenario.final_state_root,
                "{name}: ending chain state root"
            );
            assert_eq!(
                end_jump.prev_new_delta_root, scenario.final_delta_root,
                "{name}: ending chain delta root"
            );

            let (heavy_coverage, heavy_claims) = finalize_native(
                &heavy,
                total_tx_count,
                scenario.final_state_root,
                scenario.final_delta_root,
            );
            let (light_coverage, light_claims) = finalize_native(
                &light,
                total_tx_count,
                scenario.final_state_root,
                scenario.final_delta_root,
            );
            assert_eq!(
                heavy_coverage, light_claims,
                "{name}: heavy coverage != light claims"
            );
            assert_eq!(
                light_coverage, heavy_claims,
                "{name}: light coverage != heavy claims"
            );
        }
    }
}

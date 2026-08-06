//! Precompiled topological schedules for witness generation.
//!
//! For a fixed circuit, the order in which witness generators complete during a successful
//! dynamic-scheduler run of [`generate_partial_witness`] is a valid topological order of the
//! circuit's generator dependency structure, and that structure depends only on the circuit —
//! not on the witness values of a particular proof. Witness generation therefore records the
//! completion order the first time a circuit is proven in a process, and replays it on
//! subsequent runs, dispatching every generator exactly once. This eliminates the dynamic
//! scheduler's per-value watch-map probes, watch counting, and repeated no-op generator
//! dispatches when the same circuit is proven many times.
//!
//! Replays never trust the cache key for correctness:
//! * a recorded schedule is structurally revalidated against the live circuit (generator
//!   count, witness shape, and the exact watch topology) before it is used, so a digest
//!   collision between different circuits can never activate a foreign schedule;
//! * every generator recorded as "ready at completion" is dispatched only after a direct
//!   check that all of its watched representatives are populated;
//! * any divergence aborts the replay and falls back to the dynamic scheduler, which is
//!   correct from any partially populated witness state (each value written so far was
//!   produced by a generator whose watched dependencies were satisfied, i.e. it is a prefix
//!   of a valid execution of the same monotone system).
//!
//! Replayability is decided while recording: a schedule is only replayable if every
//! value-producing dispatch was also a completing dispatch. A replay runs each generator
//! once, so values emitted by earlier, non-final dispatches would otherwise be lost.
//! [`SimpleGenerator`]s always satisfy this (their `run_once` fires exactly once, when every
//! dependency is populated); exotic [`WitnessGenerator`]s that emit values incrementally are
//! detected during recording and their circuits permanently use the dynamic scheduler.
//!
//! [`generate_partial_witness`]: crate::iop::generator::generate_partial_witness
//! [`SimpleGenerator`]: crate::iop::generator::SimpleGenerator
//! [`WitnessGenerator`]: crate::iop::generator::WitnessGenerator

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::sync::Arc;

/// A replayable witness-generation schedule recorded from one successful dynamic run.
#[derive(Debug)]
pub(crate) struct RecordedSchedule {
    /// Number of generators in the circuit the schedule was recorded from.
    num_generators: usize,
    /// Length of the `PartitionWitness::values` vector (one slot per representative-forest
    /// node) the schedule was recorded against.
    num_value_slots: usize,
    /// Generator indices in the order they completed; a permutation of `0..num_generators`.
    order: Vec<u32>,
    /// For each position of `order`, whether every representative watched by that generator
    /// was populated when it completed. Always true for `SimpleGeneratorAdapter`s; general
    /// generators may legitimately finish earlier.
    ready_at_completion: Vec<bool>,
    /// CSR row offsets into `watch_rep_data`, indexed by generator; row `g` is
    /// `watch_rep_data[offsets[g]..offsets[g + 1]]`.
    watch_rep_offsets: Vec<u32>,
    /// Watched representative indices per generator, each row in ascending order. Together
    /// with `watch_rep_offsets` this is the exact inverse of the circuit's
    /// `generator_indices_by_watches` map at recording time.
    watch_rep_data: Vec<u32>,
}

impl RecordedSchedule {
    /// Generator indices in completion order.
    pub(crate) fn order(&self) -> &[u32] {
        &self.order
    }

    /// Whether the generator at `position` of [`Self::order`] completed with every watched
    /// representative populated.
    pub(crate) fn ready_at(&self, position: usize) -> bool {
        self.ready_at_completion[position]
    }

    /// The representative indices watched by `generator_idx`, ascending.
    pub(crate) fn watched_reps(&self, generator_idx: usize) -> &[u32] {
        let start = self.watch_rep_offsets[generator_idx] as usize;
        let end = self.watch_rep_offsets[generator_idx + 1] as usize;
        &self.watch_rep_data[start..end]
    }

    /// Structurally validate this schedule against a live circuit: the generator count, the
    /// witness value-slot count, and the exact watch topology must all match. Iterating the
    /// live map in its canonical (ascending-key) order and advancing a cursor per generator
    /// row makes this an exact equality check of the two watch relations, so a schedule can
    /// only be replayed on a circuit whose dependency structure is identical to the one it
    /// was recorded from.
    pub(crate) fn matches_circuit(
        &self,
        num_generators: usize,
        num_value_slots: usize,
        watch_map: &BTreeMap<usize, Vec<usize>>,
    ) -> bool {
        if self.num_generators != num_generators || self.num_value_slots != num_value_slots {
            return false;
        }

        let mut cursors = vec![0u32; num_generators];
        for (&watch_rep, watchers) in watch_map {
            for &generator_idx in watchers {
                if generator_idx >= num_generators {
                    return false;
                }
                let row = self.watched_reps(generator_idx);
                let cursor = cursors[generator_idx] as usize;
                if cursor >= row.len() || row[cursor] as usize != watch_rep {
                    return false;
                }
                cursors[generator_idx] += 1;
            }
        }

        // Every recorded watch entry must have been matched by a live one.
        cursors
            .iter()
            .enumerate()
            .all(|(generator_idx, &cursor)| cursor as usize == self.watched_reps(generator_idx).len())
    }
}

/// Records the completion order of one dynamic-scheduler run.
#[derive(Debug)]
pub(crate) struct ScheduleRecorder {
    order: Vec<u32>,
    ready_at_completion: Vec<bool>,
    replay_safe: bool,
}

impl ScheduleRecorder {
    pub(crate) fn new(num_generators: usize) -> Self {
        Self {
            order: Vec::with_capacity(num_generators),
            ready_at_completion: Vec::with_capacity(num_generators),
            // Generator indices are stored as `u32`s; refuse to record absurd circuits
            // rather than truncate.
            replay_safe: num_generators <= u32::MAX as usize,
        }
    }

    /// Note one generator dispatch made by the dynamic scheduler.
    ///
    /// A schedule stays replayable only while every value-producing dispatch is a completing
    /// dispatch: a replay runs each generator exactly once, at its completion position, so
    /// values emitted by non-final dispatches would be lost. A non-completing dispatch that
    /// already had every watch populated is likewise disqualifying — the generator would not
    /// be re-dispatched by watch traffic in a replay, so its multi-dispatch behavior cannot
    /// be reproduced.
    #[inline]
    pub(crate) fn record_dispatch(
        &mut self,
        generator_idx: usize,
        all_watches_populated: bool,
        finished: bool,
        produced_values: bool,
    ) {
        if finished {
            self.order.push(generator_idx as u32);
            self.ready_at_completion.push(all_watches_populated);
        } else if produced_values || all_watches_populated {
            self.replay_safe = false;
        }
    }

    /// Consume the recording into a replayable schedule, or `None` if the run is not
    /// replayable. Must only be called after the dynamic run succeeded.
    pub(crate) fn into_schedule(
        self,
        num_generators: usize,
        num_value_slots: usize,
        watch_map: &BTreeMap<usize, Vec<usize>>,
    ) -> Option<Arc<RecordedSchedule>> {
        if !self.replay_safe
            || self.order.len() != num_generators
            || num_value_slots > u32::MAX as usize
        {
            return None;
        }

        // Invert `watch_map` into CSR rows: for each generator, the representatives it
        // watches. Iterating the map in ascending-key order leaves each row ascending, the
        // canonical form `matches_circuit` verifies against.
        let mut row_lengths = vec![0u32; num_generators];
        let mut total_entries = 0usize;
        for watchers in watch_map.values() {
            for &generator_idx in watchers {
                if generator_idx >= num_generators {
                    return None;
                }
                row_lengths[generator_idx] += 1;
                total_entries += 1;
            }
        }
        if total_entries > u32::MAX as usize {
            return None;
        }

        let mut watch_rep_offsets = vec![0u32; num_generators + 1];
        for generator_idx in 0..num_generators {
            watch_rep_offsets[generator_idx + 1] =
                watch_rep_offsets[generator_idx] + row_lengths[generator_idx];
        }

        let mut cursors: Vec<u32> = watch_rep_offsets[..num_generators].to_vec();
        let mut watch_rep_data = vec![0u32; total_entries];
        for (&watch_rep, watchers) in watch_map {
            for &generator_idx in watchers {
                watch_rep_data[cursors[generator_idx] as usize] = watch_rep as u32;
                cursors[generator_idx] += 1;
            }
        }

        Some(Arc::new(RecordedSchedule {
            num_generators,
            num_value_slots,
            order: self.order,
            ready_at_completion: self.ready_at_completion,
            watch_rep_offsets,
            watch_rep_data,
        }))
    }
}

/// Per-circuit scheduling state, filled at most once (first successful recording wins;
/// concurrent recordings of the same circuit are benign races and both produce equivalent
/// schedules). `Some(schedule)` means replayable; `None` means the circuit must always use
/// the dynamic scheduler.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub(crate) struct CircuitScheduleCell {
    recorded: std::sync::OnceLock<Option<Arc<RecordedSchedule>>>,
}

#[cfg(feature = "std")]
impl CircuitScheduleCell {
    /// `None` if nothing has been recorded yet; `Some(None)` if the circuit is known not to
    /// be replayable; `Some(Some(_))` if a schedule is available.
    pub(crate) fn get(&self) -> Option<Option<&Arc<RecordedSchedule>>> {
        self.recorded.get().map(Option::as_ref)
    }

    /// Fill the cell; the first filler wins and later fills are ignored.
    pub(crate) fn fill(&self, schedule: Option<Arc<RecordedSchedule>>) {
        let _ = self.recorded.set(schedule);
    }
}

/// Identity of a circuit for schedule-caching purposes. The digest is what seeds Fiat–Shamir
/// for the circuit, so distinct circuits get distinct keys; even so, correctness never rests
/// on this key — see [`RecordedSchedule::matches_circuit`].
#[cfg(feature = "std")]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CircuitScheduleKey {
    /// Serialized circuit digest.
    pub digest: Vec<u8>,
    /// Concrete `GenericConfig` implementor, keeping digests of different hashers apart.
    pub config_type: &'static str,
    /// Extension degree `D`.
    pub extension_degree: usize,
}

/// Bound on retained schedules, so processes that build many circuits (e.g. test suites)
/// do not accumulate schedules without limit. The proving workload this exists for uses a
/// handful of fixed circuits. Eviction only costs the evicted circuit a re-recording.
#[cfg(feature = "std")]
const MAX_CACHED_SCHEDULES: usize = 64;

#[cfg(feature = "std")]
type CacheMap = std::collections::HashMap<CircuitScheduleKey, (u64, Arc<CircuitScheduleCell>)>;

#[cfg(feature = "std")]
static SCHEDULE_CACHE: std::sync::OnceLock<std::sync::Mutex<(u64, CacheMap)>> =
    std::sync::OnceLock::new();

/// Fetch (or create) the schedule cell for a circuit key, updating its last-used stamp and
/// evicting the least-recently-used entry if the cache is over capacity.
#[cfg(feature = "std")]
pub(crate) fn schedule_cell(key: CircuitScheduleKey) -> Arc<CircuitScheduleCell> {
    let mutex = SCHEDULE_CACHE.get_or_init(Default::default);
    let mut guard = mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (tick, map) = &mut *guard;
    cell_for_key(tick, map, key)
}

#[cfg(feature = "std")]
fn cell_for_key(tick: &mut u64, map: &mut CacheMap, key: CircuitScheduleKey) -> Arc<CircuitScheduleCell> {
    *tick += 1;
    if let Some((stamp, cell)) = map.get_mut(&key) {
        *stamp = *tick;
        return Arc::clone(cell);
    }
    if map.len() >= MAX_CACHED_SCHEDULES {
        if let Some(oldest) = map
            .iter()
            .min_by_key(|(_, (stamp, _))| *stamp)
            .map(|(key, _)| key.clone())
        {
            map.remove(&oldest);
        }
    }
    let cell = Arc::new(CircuitScheduleCell::default());
    map.insert(key, (*tick, Arc::clone(&cell)));
    cell
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn record_completions(recorder: &mut ScheduleRecorder, completions: &[usize]) {
        for &generator_idx in completions {
            recorder.record_dispatch(generator_idx, true, true, true);
        }
    }

    /// Watch map: generator 0 watches reps {2, 5}; generator 1 watches rep {2}; generator 2
    /// watches nothing.
    fn sample_watch_map() -> BTreeMap<usize, Vec<usize>> {
        let mut map = BTreeMap::new();
        map.insert(2usize, vec![0usize, 1]);
        map.insert(5usize, vec![0usize]);
        map
    }

    fn sample_schedule() -> Arc<RecordedSchedule> {
        let mut recorder = ScheduleRecorder::new(3);
        record_completions(&mut recorder, &[2, 1, 0]);
        recorder
            .into_schedule(3, 8, &sample_watch_map())
            .expect("sample recording should be replayable")
    }

    #[test]
    fn schedule_inverts_watch_map_and_validates_structurally() {
        let schedule = sample_schedule();
        assert_eq!(schedule.order(), &[2, 1, 0]);
        assert_eq!(schedule.watched_reps(0), &[2, 5]);
        assert_eq!(schedule.watched_reps(1), &[2]);
        assert_eq!(schedule.watched_reps(2), &[] as &[u32]);
        assert!(schedule.matches_circuit(3, 8, &sample_watch_map()));
    }

    #[test]
    fn mismatched_circuits_are_rejected() {
        let schedule = sample_schedule();

        // Different generator count or witness shape.
        assert!(!schedule.matches_circuit(4, 8, &sample_watch_map()));
        assert!(!schedule.matches_circuit(3, 9, &sample_watch_map()));

        // A live watch entry the recording did not have.
        let mut extra_entry = sample_watch_map();
        extra_entry.insert(6, vec![1]);
        assert!(!schedule.matches_circuit(3, 8, &extra_entry));

        // A recorded watch entry the live circuit does not have.
        let mut missing_entry = sample_watch_map();
        missing_entry.remove(&5);
        assert!(!schedule.matches_circuit(3, 8, &missing_entry));

        // Same shape, different representative.
        let mut different_rep = sample_watch_map();
        let watchers = different_rep.remove(&5).unwrap();
        different_rep.insert(4, watchers);
        assert!(!schedule.matches_circuit(3, 8, &different_rep));

        // Same representative, different watcher.
        let mut different_watcher = sample_watch_map();
        different_watcher.insert(5, vec![1]);
        assert!(!schedule.matches_circuit(3, 8, &different_watcher));
    }

    #[test]
    fn value_producing_non_final_dispatch_disqualifies_replay() {
        let mut recorder = ScheduleRecorder::new(2);
        // Generator 0 emits values without finishing, then both complete.
        recorder.record_dispatch(0, false, false, true);
        record_completions(&mut recorder, &[1, 0]);
        assert!(recorder.into_schedule(2, 8, &BTreeMap::new()).is_none());
    }

    #[test]
    fn ready_non_final_dispatch_disqualifies_replay() {
        let mut recorder = ScheduleRecorder::new(2);
        // Generator 0 was dispatched with every watch populated but did not finish.
        recorder.record_dispatch(0, true, false, false);
        record_completions(&mut recorder, &[1, 0]);
        assert!(recorder.into_schedule(2, 8, &BTreeMap::new()).is_none());
    }

    #[test]
    fn quiet_unready_dispatches_do_not_disqualify_replay() {
        let mut recorder = ScheduleRecorder::new(2);
        recorder.record_dispatch(0, false, false, false);
        record_completions(&mut recorder, &[1, 0]);
        let schedule = recorder
            .into_schedule(2, 8, &BTreeMap::new())
            .expect("quiet unready dispatches are replayable");
        assert_eq!(schedule.order(), &[1, 0]);
    }

    #[test]
    fn incomplete_recording_is_not_replayable() {
        let mut recorder = ScheduleRecorder::new(3);
        record_completions(&mut recorder, &[1, 0]);
        assert!(recorder.into_schedule(3, 8, &BTreeMap::new()).is_none());
    }

    #[test]
    fn cache_evicts_least_recently_used_cell() {
        let mut tick = 0u64;
        let mut map = CacheMap::default();
        let key = |i: usize| CircuitScheduleKey {
            digest: vec![i as u8],
            config_type: "test",
            extension_degree: 2,
        };

        let first = cell_for_key(&mut tick, &mut map, key(0));
        for i in 1..MAX_CACHED_SCHEDULES {
            cell_for_key(&mut tick, &mut map, key(i));
        }
        // Touch key 0 so key 1 is now the least recently used.
        let first_again = cell_for_key(&mut tick, &mut map, key(0));
        assert!(Arc::ptr_eq(&first, &first_again));

        cell_for_key(&mut tick, &mut map, key(MAX_CACHED_SCHEDULES));
        assert_eq!(map.len(), MAX_CACHED_SCHEDULES);
        assert!(map.contains_key(&key(0)));
        assert!(!map.contains_key(&key(1)));
    }
}

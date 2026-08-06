#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Debug;
use core::marker::PhantomData;

use anyhow::{Result, anyhow};

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::iop::topo_schedule::ScheduleRecorder;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Given a `PartitionWitness` that has only inputs set, populates the rest of the witness using the
/// given set of generators.
///
/// For a fixed circuit the generator dependency structure does not depend on witness values, so
/// the first successful run for a circuit records the order in which generators complete — and,
/// per completing generator, the representative slot each of its writes resolved to — and later
/// runs for the same circuit replay that order directly, dispatching each generator exactly once
/// and applying its writes through the precompiled slots instead of rediscovering the execution
/// order through the watch map and re-resolving every write through the representative map. See
/// [`crate::iop::topo_schedule`] for the recording, validation, and fallback rules that keep the
/// replayed execution value-identical to a dynamic one.
pub fn generate_partial_witness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    generate_partial_witness_inner(inputs, prover_data, common_data, true)
}

/// Like [`generate_partial_witness`], but recorded-schedule replays resolve every write through
/// the live representative map instead of the precompiled slot lists. Test-only: lets tests
/// compare the slot fast path against the reference write path on the same recorded schedule.
#[cfg(test)]
pub(crate) fn generate_partial_witness_unprecompiled_writes<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    generate_partial_witness_inner(inputs, prover_data, common_data, false)
}

#[cfg_attr(not(feature = "std"), allow(unused_variables))]
fn generate_partial_witness_inner<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
    use_precompiled_writes: bool,
) -> Result<PartitionWitness<'a, F>> {
    let mut witness = witness_with_inputs(inputs, prover_data, common_data)?;
    let num_generators = prover_data.generators.len();

    #[cfg(feature = "std")]
    {
        use crate::iop::topo_schedule::{self, CircuitScheduleKey};
        use crate::plonk::config::GenericHashOut;

        let cell = topo_schedule::schedule_cell(CircuitScheduleKey {
            digest: prover_data.circuit_digest.to_bytes(),
            config_type: core::any::type_name::<C>(),
            extension_degree: D,
        });
        match cell.get() {
            // A recorded schedule exists and provably describes this circuit's watch
            // topology: replay it.
            Some(Some(schedule))
                if schedule.matches_circuit(
                    num_generators,
                    witness.values.len(),
                    witness.num_wires,
                    witness.degree,
                    &prover_data.generator_indices_by_watches,
                ) =>
            {
                run_recorded_schedule(&mut witness, prover_data, schedule, use_precompiled_writes)?;
            }
            // The circuit is known not to be replayable (or, for `Some(Some(_))`, the cached
            // schedule was recorded from a digest-colliding circuit with a different
            // structure): use the dynamic scheduler.
            Some(_) => {
                let expired = vec![false; num_generators];
                run_dynamic_schedule(&mut witness, prover_data, expired, num_generators, None)?;
            }
            // First run for this circuit: use the dynamic scheduler and record the completion
            // order. Concurrent first runs both record; the first to fill the cell wins.
            None => {
                let mut recorder = ScheduleRecorder::new(num_generators);
                let expired = vec![false; num_generators];
                run_dynamic_schedule(
                    &mut witness,
                    prover_data,
                    expired,
                    num_generators,
                    Some(&mut recorder),
                )?;
                cell.fill(recorder.into_schedule(
                    num_generators,
                    witness.values.len(),
                    witness.num_wires,
                    witness.degree,
                    &prover_data.generator_indices_by_watches,
                ));
            }
        }
    }

    #[cfg(not(feature = "std"))]
    {
        let expired = vec![false; num_generators];
        run_dynamic_schedule(&mut witness, prover_data, expired, num_generators, None)?;
    }

    Ok(witness)
}

/// Populates the witness with the given generators using only the dynamic scheduler, without
/// consulting or filling the schedule cache. Test-only: lets tests compare a replayed execution
/// against a from-scratch dynamic one on the same circuit.
#[cfg(test)]
pub(crate) fn generate_partial_witness_dynamic<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    let mut witness = witness_with_inputs(inputs, prover_data, common_data)?;
    let num_generators = prover_data.generators.len();
    let expired = vec![false; num_generators];
    run_dynamic_schedule(&mut witness, prover_data, expired, num_generators, None)?;
    Ok(witness)
}

/// Creates a `PartitionWitness` for the circuit and sets the input values.
fn witness_with_inputs<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    let mut witness = PartitionWitness::new(
        common_data.config.num_wires,
        common_data.degree(),
        &prover_data.representative_map,
    );

    for (t, v) in inputs.target_values.into_iter() {
        witness.set_target(t, v)?;
    }

    Ok(witness)
}

/// Runs the generators in a previously recorded completion order, dispatching each exactly once.
///
/// The recorded order is a valid topological order of the circuit's generator dependencies (the
/// recording run proves it), so each generator's dependencies are already populated when it is
/// dispatched; a direct check of its watched representatives verifies this before every ready
/// dispatch. A finished dispatch's writes are applied through its recorded write signature
/// (see [`apply_precompiled_writes`]), which validates them before any store and eliminates the
/// per-write representative-map lookup; with `use_precompiled_writes` false (test-only), they
/// are instead resolved through the live representative map, as the dynamic scheduler would.
/// If a dispatch diverges — a dependency is unset, a generator does not finish (e.g. because a
/// different set of input targets was provided than when the schedule was recorded), or its
/// writes do not match the recorded signature — the values produced by the failed dispatch are
/// discarded before any of them reach the witness, and the remaining generators are completed
/// by the dynamic scheduler, which is correct from any partially populated witness state.
#[cfg(feature = "std")]
fn run_recorded_schedule<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &mut PartitionWitness<F>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    schedule: &crate::iop::topo_schedule::RecordedSchedule,
    use_precompiled_writes: bool,
) -> Result<()> {
    let generators = &prover_data.generators;
    let order = schedule.order();
    let mut buffer = GeneratedValues::empty();

    for (position, &generator_idx) in order.iter().enumerate() {
        let generator_idx = generator_idx as usize;
        let ready = schedule.ready_at(position);

        // Defensive dependency check: only claim readiness the generator can rely on after
        // directly verifying that every representative it watches is populated.
        let dependencies_populated = !ready
            || schedule
                .watched_reps(generator_idx)
                .iter()
                .all(|&rep| witness.values[rep as usize].is_some());

        let finished = dependencies_populated
            && generators[generator_idx]
                .0
                .run_with_ready_hint(witness, &mut buffer, ready);

        let applied = finished
            && if use_precompiled_writes {
                apply_precompiled_writes(witness, &mut buffer, schedule.writes_at(position))?
            } else {
                for (t, v) in buffer.target_values.drain(..) {
                    witness.set_target(t, v)?;
                }
                true
            };

        if !applied {
            // This run diverged from the recorded execution. Discard anything the failed
            // dispatch produced — none of it has reached the witness, since signature
            // validation precedes every precompiled store — and finish dynamically (the
            // dynamic scheduler re-runs the diverged generator and regenerates its values
            // legitimately); generators replayed so far each ran with satisfied
            // dependencies, so the current witness is a prefix of a valid execution.
            buffer.target_values.clear();
            let mut expired = vec![false; generators.len()];
            for &done in &order[..position] {
                expired[done as usize] = true;
            }
            let remaining = generators.len() - position;
            return run_dynamic_schedule(witness, prover_data, expired, remaining, None);
        }
    }

    Ok(())
}

/// Applies one replayed dispatch's buffered writes through its recorded write signature.
///
/// Phase one validates, without touching the witness, that the dispatch wrote exactly the
/// recorded targets in the recorded order ([`Target::index`] is pure arithmetic, so this costs
/// no memory traffic beyond streaming the signature). Phase two then stores each value directly
/// into its recorded representative slot with the write-once compare-and-set of
/// [`PartitionWitness::set_slot_checked`] — zero representative-map lookups. Returns `Ok(false)`
/// with the witness untouched and the buffer intact if the signature does not match, so the
/// caller can fall back to dynamic scheduling; a conflicting value in a slot is a hard error,
/// exactly as on the dynamic write path.
#[cfg(feature = "std")]
fn apply_precompiled_writes<F: Field>(
    witness: &mut PartitionWitness<F>,
    buffer: &mut GeneratedValues<F>,
    signature: &[crate::iop::topo_schedule::RecordedWrite],
) -> Result<bool> {
    let writes = &buffer.target_values;
    if writes.len() != signature.len() {
        return Ok(false);
    }
    for (&(target, _), recorded) in writes.iter().zip(signature) {
        if witness.target_index(target) != recorded.target_index as usize {
            return Ok(false);
        }
    }
    for (&(target, value), recorded) in writes.iter().zip(signature) {
        witness.set_slot_checked(recorded.slot as usize, target, value)?;
    }
    buffer.target_values.clear();
    Ok(true)
}

/// Runs all non-expired generators to completion by dynamic scheduling: generators are queued,
/// dispatched, and re-queued as the representatives they watch become populated. Correct from any
/// partially populated witness state. If `recorder` is provided, the completion order of this run
/// is recorded for later replay.
fn run_dynamic_schedule<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &mut PartitionWitness<F>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    mut generator_is_expired: Vec<bool>,
    mut remaining_generators: usize,
    mut recorder: Option<&mut ScheduleRecorder>,
) -> Result<()> {
    let generators = &prover_data.generators;
    let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

    // A simple generator can run once all of the distinct representatives it watches have values.
    // Derive those unresolved counts from the existing watcher index so this remains local witness
    // state and does not add anything to serialized prover data.
    let mut unresolved_watches = vec![0usize; generators.len()];
    for (&watch, watchers) in generator_indices_by_watches {
        if witness.values[watch].is_none() {
            for &generator_idx in watchers {
                if !generator_is_expired[generator_idx] {
                    unresolved_watches[generator_idx] += 1;
                }
            }
        }
    }

    // Build a list of "pending" generators which are queued to be run. Initially, all non-expired
    // generators are queued.
    let mut pending_generator_indices: Vec<_> = (0..generators.len())
        .filter(|&i| !generator_is_expired[i])
        .collect();

    let mut buffer = GeneratedValues::empty();

    // Keep running generators until we fail to make progress.
    while !pending_generator_indices.is_empty() {
        let mut next_pending_generator_indices = Vec::new();

        for &generator_idx in &pending_generator_indices {
            if generator_is_expired[generator_idx] {
                continue;
            }

            let all_watches_populated = unresolved_watches[generator_idx] == 0;
            let finished = generators[generator_idx].0.run_with_ready_hint(
                witness,
                &mut buffer,
                all_watches_populated,
            );
            if let Some(recorder) = recorder.as_deref_mut() {
                recorder.record_dispatch(
                    generator_idx,
                    all_watches_populated,
                    finished,
                    !buffer.target_values.is_empty(),
                );
                if finished {
                    // Record the completing dispatch's write signature: each written target's
                    // canonical index and the representative slot it resolves to, in write
                    // order. Replays validate against, and write through, this signature.
                    recorder.record_completed_writes(buffer.target_values.iter().map(
                        |&(t, _)| {
                            let target_index = witness.target_index(t);
                            (target_index, witness.representative_map[target_index])
                        },
                    ));
                }
            }
            if finished {
                generator_is_expired[generator_idx] = true;
                remaining_generators -= 1;
            }

            // Merge any generated values into our witness, and get a list of newly-populated
            // targets' representatives.
            let mut new_target_reps = Vec::with_capacity(buffer.target_values.len());
            for (t, v) in buffer.target_values.drain(..) {
                let reps = witness.set_target_returning_rep(t, v)?;
                new_target_reps.extend(reps);
            }

            // Enqueue unfinished generators that were watching one of the newly populated targets.
            for watch in new_target_reps {
                let opt_watchers = generator_indices_by_watches.get(&watch);
                if let Some(watchers) = opt_watchers {
                    for &watching_generator_idx in watchers {
                        if !generator_is_expired[watching_generator_idx] {
                            debug_assert_ne!(unresolved_watches[watching_generator_idx], 0);
                            unresolved_watches[watching_generator_idx] -= 1;
                            next_pending_generator_indices.push(watching_generator_idx);
                        }
                    }
                }
            }
        }

        pending_generator_indices = next_pending_generator_indices;
    }

    if remaining_generators != 0 {
        return Err(anyhow!("{} generators weren't run", remaining_generators));
    }

    Ok(())
}

/// A generator participates in the generation of the witness.
pub trait WitnessGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    /// Targets to be "watched" by this generator. Whenever a target in the watch list is populated,
    /// the generator will be queued to run.
    fn watch_list(&self) -> Vec<Target>;

    /// Run this generator, returning a flag indicating whether the generator is finished. If the
    /// flag is true, the generator will never be run again, otherwise it will be queued for another
    /// run next time a target in its watch list is populated.
    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool;

    /// Scheduler entry point carrying a hint that every watched representative is populated.
    ///
    /// General generators may produce values before all watches are populated, so the default
    /// implementation preserves their existing [`Self::run`] behavior. Generators which require
    /// every watch can override this to avoid rediscovering readiness.
    #[doc(hidden)]
    fn run_with_ready_hint(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
        _all_watches_populated: bool,
    ) -> bool {
        self.run(witness, out_buffer)
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

/// A wrapper around an `Box<WitnessGenerator>` which implements `PartialEq`
/// and `Eq` based on generator IDs.
pub struct WitnessGeneratorRef<F: RichField + Extendable<D>, const D: usize>(
    pub Box<dyn WitnessGenerator<F, D>>,
);

impl<F: RichField + Extendable<D>, const D: usize> WitnessGeneratorRef<F, D> {
    pub fn new<G: WitnessGenerator<F, D>>(generator: G) -> WitnessGeneratorRef<F, D> {
        WitnessGeneratorRef(Box::new(generator))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PartialEq for WitnessGeneratorRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Eq for WitnessGeneratorRef<F, D> {}

impl<F: RichField + Extendable<D>, const D: usize> Debug for WitnessGeneratorRef<F, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0.id())
    }
}

/// Values generated by a generator invocation.
#[derive(Debug)]
pub struct GeneratedValues<F: Field> {
    pub target_values: Vec<(Target, F)>,
}

impl<F: Field> From<Vec<(Target, F)>> for GeneratedValues<F> {
    fn from(target_values: Vec<(Target, F)>) -> Self {
        Self { target_values }
    }
}

impl<F: Field> WitnessWrite<F> for GeneratedValues<F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        self.target_values.push((target, value));

        Ok(())
    }
}

impl<F: Field> GeneratedValues<F> {
    pub fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    pub fn empty() -> Self {
        Vec::new().into()
    }

    pub fn singleton_wire(wire: Wire, value: F) -> Self {
        Self::singleton_target(Target::Wire(wire), value)
    }

    pub fn singleton_target(target: Target, value: F) -> Self {
        vec![(target, value)].into()
    }

    pub fn singleton_extension_target<const D: usize>(
        et: ExtensionTarget<D>,
        value: F::Extension,
    ) -> Result<Self>
    where
        F: RichField + Extendable<D>,
    {
        let mut witness = Self::with_capacity(D);
        witness.set_extension_target(et, value)?;

        Ok(witness)
    }
}

/// A generator which runs once after a list of dependencies is present in the witness.
pub trait SimpleGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    fn dependencies(&self) -> Vec<Target>;

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()>;

    fn adapter(self) -> SimpleGeneratorAdapter<F, Self, D>
    where
        Self: Sized,
    {
        SimpleGeneratorAdapter {
            inner: self,
            _phantom: PhantomData,
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct SimpleGeneratorAdapter<
    F: RichField + Extendable<D>,
    SG: SimpleGenerator<F, D> + ?Sized,
    const D: usize,
> {
    _phantom: PhantomData<F>,
    inner: SG,
}

impl<F: RichField + Extendable<D>, SG: SimpleGenerator<F, D>, const D: usize> WitnessGenerator<F, D>
    for SimpleGeneratorAdapter<F, SG, D>
{
    fn id(&self) -> String {
        self.inner.id()
    }

    fn watch_list(&self) -> Vec<Target> {
        self.inner.dependencies()
    }

    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
        if witness.contains_all(&self.inner.dependencies()) {
            self.inner.run_once(witness, out_buffer).is_ok()
        } else {
            false
        }
    }

    fn run_with_ready_hint(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
        all_watches_populated: bool,
    ) -> bool {
        all_watches_populated && self.inner.run_once(witness, out_buffer).is_ok()
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.inner.serialize(dst, common_data)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            inner: SG::deserialize(src, common_data)?,
            _phantom: PhantomData,
        })
    }
}

/// A generator which copies one wire to another.
#[derive(Debug, Default)]
pub struct CopyGenerator {
    pub(crate) src: Target,
    pub(crate) dst: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for CopyGenerator {
    fn id(&self) -> String {
        "CopyGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.src]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let value = witness.get_target(self.src);
        out_buffer.set_target(self.dst, value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.src)?;
        dst.write_target(self.dst)
    }

    fn deserialize(source: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let src = source.read_target()?;
        let dst = source.read_target()?;
        Ok(Self { src, dst })
    }
}

/// A generator for including a random value
#[derive(Debug, Default)]
pub struct RandomValueGenerator {
    pub(crate) target: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for RandomValueGenerator {
    fn id(&self) -> String {
        "RandomValueGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        Vec::new()
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let random_value = F::rand();
        out_buffer.set_target(self.target, random_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.target)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let target = src.read_target()?;
        Ok(Self { target })
    }
}

/// A generator for testing if a value equals zero
#[derive(Debug, Default)]
pub struct NonzeroTestGenerator {
    pub(crate) to_test: Target,
    pub(crate) dummy: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for NonzeroTestGenerator {
    fn id(&self) -> String {
        "NonzeroTestGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.to_test]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let to_test_value = witness.get_target(self.to_test);

        let dummy_value = if to_test_value == F::ZERO {
            F::ONE
        } else {
            to_test_value.inverse()
        };

        out_buffer.set_target(self.dummy, dummy_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.to_test)?;
        dst.write_target(self.dummy)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let to_test = src.read_target()?;
        let dummy = src.read_target()?;
        Ok(Self { to_test, dummy })
    }
}

/// Generator used to fill an extra constant.
#[derive(Debug, Clone, Default)]
pub struct ConstantGenerator<F: Field> {
    pub row: usize,
    pub constant_index: usize,
    pub wire_index: usize,
    pub constant: F,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn set_constant(&mut self, c: F) {
        self.constant = c;
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ConstantGenerator<F> {
    fn id(&self) -> String {
        "ConstantGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        out_buffer.set_target(Target::wire(self.row, self.wire_index), self.constant)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.constant_index)?;
        dst.write_usize(self.wire_index)?;
        dst.write_field(self.constant)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let constant_index = src.read_usize()?;
        let wire_index = src.read_usize()?;
        let constant = src.read_field()?;
        Ok(Self {
            row,
            constant_index,
            wire_index,
            constant,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    const D: usize = 2;
    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;

    #[derive(Debug)]
    struct CountingSimpleGenerator {
        dependencies: Vec<Target>,
        output: Target,
        dependency_calls: Arc<AtomicUsize>,
        run_calls: Arc<AtomicUsize>,
    }

    impl SimpleGenerator<F, D> for CountingSimpleGenerator {
        fn id(&self) -> String {
            "CountingSimpleGenerator".to_string()
        }

        fn dependencies(&self) -> Vec<Target> {
            self.dependency_calls.fetch_add(1, Ordering::Relaxed);
            self.dependencies.clone()
        }

        fn run_once(
            &self,
            witness: &PartitionWitness<F>,
            out_buffer: &mut GeneratedValues<F>,
        ) -> Result<()> {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            let value = self
                .dependencies
                .iter()
                .map(|&target| witness.get_target(target))
                .sum();
            out_buffer.set_target(self.output, value)
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<()> {
            unreachable!("test generator is never serialized")
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<Self> {
            unreachable!("test generator is never deserialized")
        }
    }

    #[derive(Debug)]
    struct IncrementalGenerator {
        trigger: Target,
        early_output: Target,
        final_output: Target,
        run_calls: Arc<AtomicUsize>,
    }

    impl WitnessGenerator<F, D> for IncrementalGenerator {
        fn id(&self) -> String {
            "IncrementalGenerator".to_string()
        }

        fn watch_list(&self) -> Vec<Target> {
            vec![self.trigger]
        }

        fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            if let Some(value) = witness.try_get_target(self.trigger) {
                out_buffer.set_target(self.final_output, value).unwrap();
                true
            } else {
                out_buffer
                    .set_target(self.early_output, F::from_canonical_u64(7))
                    .unwrap();
                false
            }
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<()> {
            unreachable!("test generator is never serialized")
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<Self> {
            unreachable!("test generator is never deserialized")
        }
    }

    #[test]
    fn simple_generator_uses_representative_readiness_without_rescanning_dependencies() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let initial = builder.add_virtual_target();
        let first = builder.constant(F::from_canonical_u64(3));
        let first_alias = builder.add_virtual_target();
        builder.connect(first, first_alias);
        let second = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        let dependency_calls = Arc::new(AtomicUsize::new(0));
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_simple_generator(CountingSimpleGenerator {
            dependencies: vec![initial, initial, first, first_alias, second, second],
            output,
            dependency_calls: Arc::clone(&dependency_calls),
            run_calls: Arc::clone(&run_calls),
        });
        builder.generate_copy(first, second);
        builder.register_public_input(output);

        let circuit = builder.build::<C>();
        let dependency_calls_after_build = dependency_calls.load(Ordering::Relaxed);
        let mut inputs = PartialWitness::new();
        inputs
            .set_target(initial, F::from_canonical_u64(5))
            .unwrap();
        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        assert_eq!(witness.get_target(output), F::from_canonical_u64(22));
        assert_eq!(run_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            dependency_calls.load(Ordering::Relaxed),
            dependency_calls_after_build
        );
    }

    /// A general `WitnessGenerator` computing `output = input + 1` that counts every scheduler
    /// dispatch and every actual execution. Uses the default `run_with_ready_hint`, so dispatch
    /// counts are comparable between the dynamic scheduler and a recorded-schedule replay.
    #[derive(Debug)]
    struct ChainSumGenerator {
        input: Target,
        output: Target,
        dispatches: Arc<AtomicUsize>,
        executions: Arc<AtomicUsize>,
    }

    impl WitnessGenerator<F, D> for ChainSumGenerator {
        fn id(&self) -> String {
            "ChainSumGenerator".to_string()
        }

        fn watch_list(&self) -> Vec<Target> {
            vec![self.input]
        }

        fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
            self.dispatches.fetch_add(1, Ordering::Relaxed);
            match witness.try_get_target(self.input) {
                Some(value) => {
                    self.executions.fetch_add(1, Ordering::Relaxed);
                    out_buffer.set_target(self.output, value + F::ONE).unwrap();
                    true
                }
                None => false,
            }
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<()> {
            unreachable!("test generator is never serialized")
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<Self> {
            unreachable!("test generator is never deserialized")
        }
    }

    /// Adds a chain of `length` `ChainSumGenerator`s computing `c_{i+1} = c_i + 1` from `from`,
    /// registered in reverse topological order so the dynamic scheduler needs one no-op dispatch
    /// plus one requeued dispatch for every generator except the chain head. Returns the
    /// dispatch/execution counters (in registration order) and the final chain target.
    fn add_reversed_chain(
        builder: &mut CircuitBuilder<F, D>,
        from: Target,
        length: usize,
    ) -> (Vec<Arc<AtomicUsize>>, Vec<Arc<AtomicUsize>>, Target) {
        let mut targets = vec![from];
        for _ in 0..length {
            targets.push(builder.add_virtual_target());
        }

        let mut dispatches = Vec::new();
        let mut executions = Vec::new();
        for i in (0..length).rev() {
            let dispatch_count = Arc::new(AtomicUsize::new(0));
            let execution_count = Arc::new(AtomicUsize::new(0));
            builder.add_generators(vec![WitnessGeneratorRef::new(ChainSumGenerator {
                input: targets[i],
                output: targets[i + 1],
                dispatches: Arc::clone(&dispatch_count),
                executions: Arc::clone(&execution_count),
            })]);
            dispatches.push(dispatch_count);
            executions.push(execution_count);
        }

        (dispatches, executions, targets[length])
    }

    fn counter_values(counters: &[Arc<AtomicUsize>]) -> Vec<usize> {
        counters
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    fn reset_counters(counters: &[Arc<AtomicUsize>]) {
        for counter in counters {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// Every built circuit re-randomizes its unused public-input wires on each witness
    /// generation (`RandomValueGenerator`s added by `randomize_unused_pi_wires`), so even two
    /// dynamic runs differ on those slots. The slots on which two independent dynamic
    /// reference runs agree are exactly the value-deterministic ones; scheduling changes must
    /// reproduce all of them bit for bit.
    fn deterministic_slots(
        reference_a: &PartitionWitness<F>,
        reference_b: &PartitionWitness<F>,
    ) -> Vec<bool> {
        assert_eq!(reference_a.values.len(), reference_b.values.len());
        reference_a
            .values
            .iter()
            .zip(&reference_b.values)
            .map(|(a, b)| a == b)
            .collect()
    }

    fn assert_agree_on_deterministic_slots(
        mask: &[bool],
        expected: &PartitionWitness<F>,
        actual: &PartitionWitness<F>,
    ) {
        assert_eq!(expected.values.len(), mask.len());
        assert_eq!(actual.values.len(), mask.len());
        for (slot, keep) in mask.iter().enumerate() {
            if *keep {
                assert_eq!(
                    expected.values[slot], actual.values[slot],
                    "witness slot {slot} diverged",
                );
            }
        }
    }

    #[test]
    fn recorded_schedule_dispatches_each_generator_exactly_once() {
        const CHAIN_LENGTH: usize = 5;

        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        // Unique constant, registered public, so this circuit's digest collides with no other
        // test's circuit in the process-wide schedule cache.
        let salt = builder.constant(F::from_canonical_u64(0x70_60_50_40_30_20_10_01));
        builder.register_public_input(salt);
        let entry = builder.add_virtual_target();
        let (dispatches, executions, chain_out) =
            add_reversed_chain(&mut builder, entry, CHAIN_LENGTH);
        builder.register_public_input(chain_out);
        let circuit = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs.set_target(entry, F::from_canonical_u64(10)).unwrap();

        // Two dynamic reference runs to identify the value-deterministic witness slots. These
        // bypass the schedule cache entirely.
        let reference_a =
            generate_partial_witness_dynamic(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        let reference_b =
            generate_partial_witness_dynamic(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        let mask = deterministic_slots(&reference_a, &reference_b);
        reset_counters(&dispatches);
        reset_counters(&executions);

        // First generation runs the dynamic scheduler and records the completion order. In the
        // dynamic scheduler every chain generator is dispatched once from the initial queue (a
        // no-op for all but the chain head, whose input is already set) and every non-head
        // generator is dispatched a second time when the representative it watches is populated:
        // 2 * CHAIN_LENGTH - 1 dispatches in total.
        let first_witness =
            generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        assert_eq!(
            counter_values(&dispatches).iter().sum::<usize>(),
            2 * CHAIN_LENGTH - 1,
        );
        assert_eq!(counter_values(&executions), vec![1; CHAIN_LENGTH]);

        // Second generation replays the recorded order: every generator is dispatched exactly
        // once, and executes on that single dispatch.
        reset_counters(&dispatches);
        reset_counters(&executions);
        let second_witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();
        assert_eq!(counter_values(&dispatches), vec![1; CHAIN_LENGTH]);
        assert_eq!(counter_values(&executions), vec![1; CHAIN_LENGTH]);

        // Same inputs, deterministic generators: the replayed execution must reproduce the
        // dynamic execution's witness bit for bit on every value-deterministic slot.
        assert_agree_on_deterministic_slots(&mask, &reference_a, &first_witness);
        assert_agree_on_deterministic_slots(&mask, &reference_a, &second_witness);
        assert_eq!(
            second_witness.get_target(chain_out),
            F::from_canonical_u64(10 + CHAIN_LENGTH as u64),
        );
    }

    #[test]
    fn recorded_schedule_matches_dynamic_witness_values() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt = builder.constant(F::from_canonical_u64(0x0D1F_F2E4_5A17_0002));
        builder.register_public_input(salt);
        let x = builder.add_virtual_target();
        // Mix in real gate generators (arithmetic) below the counting chain.
        let x_squared = builder.mul(x, x);
        let y = builder.add(x_squared, salt);
        let (dispatches, _executions, chain_out) = add_reversed_chain(&mut builder, y, 3);
        builder.register_public_input(chain_out);
        let circuit = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs.set_target(x, F::from_canonical_u64(7)).unwrap();

        // Reference executions: dynamic scheduling only, no schedule cache involved. Two runs
        // identify the value-deterministic slots (unused public-input wires are re-randomized
        // on every generation, by design).
        let dynamic_witness = generate_partial_witness_dynamic(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let dynamic_witness_b = generate_partial_witness_dynamic(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let mask = deterministic_slots(&dynamic_witness, &dynamic_witness_b);

        // Recording run, then a replayed run.
        let recording_witness =
            generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        reset_counters(&dispatches);
        let replayed_witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        // The single dispatch per chain generator proves the second run replayed the recorded
        // schedule rather than scheduling dynamically.
        assert_eq!(counter_values(&dispatches), vec![1; 3]);

        // Differential check: dynamic, recording, and replayed executions agree on every
        // value-deterministic witness slot.
        assert_agree_on_deterministic_slots(&mask, &dynamic_witness, &recording_witness);
        assert_agree_on_deterministic_slots(&mask, &dynamic_witness, &replayed_witness);
    }

    #[test]
    fn multi_dispatch_value_writers_keep_dynamic_scheduling() {
        // `IncrementalGenerator` emits a value on a non-final dispatch, so its completion order
        // cannot be replayed (a replay would skip the early write). The recording must classify
        // the circuit as non-replayable and keep using the dynamic scheduler.
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt = builder.constant(F::from_canonical_u64(0x0D1F_F2E4_5A17_0003));
        builder.register_public_input(salt);
        let trigger = builder.constant(F::from_canonical_u64(11));
        let early_output = builder.add_virtual_target();
        let final_output = builder.add_virtual_target();
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_generators(vec![WitnessGeneratorRef::new(IncrementalGenerator {
            trigger,
            early_output,
            final_output,
            run_calls: Arc::clone(&run_calls),
        })]);
        builder.register_public_inputs(&[early_output, final_output]);
        let circuit = builder.build::<C>();

        // Reference runs for the value-deterministic slot mask (each dynamic run dispatches
        // the generator twice).
        let reference_a = generate_partial_witness_dynamic(
            PartialWitness::new(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let reference_b = generate_partial_witness_dynamic(
            PartialWitness::new(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let mask = deterministic_slots(&reference_a, &reference_b);
        assert_eq!(run_calls.load(Ordering::Relaxed), 4);

        let first_witness =
            generate_partial_witness(PartialWitness::new(), &circuit.prover_only, &circuit.common)
                .unwrap();
        assert_eq!(run_calls.load(Ordering::Relaxed), 6);

        // A second generation must again dispatch the generator twice (dynamic scheduling); a
        // replay would have dispatched it only once and lost the early write.
        let second_witness =
            generate_partial_witness(PartialWitness::new(), &circuit.prover_only, &circuit.common)
                .unwrap();
        assert_eq!(run_calls.load(Ordering::Relaxed), 8);

        assert_agree_on_deterministic_slots(&mask, &reference_a, &first_witness);
        assert_agree_on_deterministic_slots(&mask, &reference_a, &second_witness);
        assert_eq!(second_witness.get_target(early_output), F::from_canonical_u64(7));
        assert_eq!(second_witness.get_target(final_output), F::from_canonical_u64(11));
    }

    #[test]
    fn mismatched_generator_sets_fall_back_to_dynamic_scheduling() {
        const SHARED_SALT: u64 = 0x0D1F_F2E4_5A17_0004;

        // Circuit A: salt + a 2-generator counting chain.
        let mut builder_a =
            CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt_a = builder_a.constant(F::from_canonical_u64(SHARED_SALT));
        builder_a.register_public_input(salt_a);
        let entry_a = builder_a.add_virtual_target();
        let (dispatches_a, _, _) = add_reversed_chain(&mut builder_a, entry_a, 2);
        let circuit_a = builder_a.build::<C>();

        let mut inputs_a = PartialWitness::new();
        inputs_a
            .set_target(entry_a, F::from_canonical_u64(3))
            .unwrap();

        // Record circuit A's schedule.
        generate_partial_witness(inputs_a.clone(), &circuit_a.prover_only, &circuit_a.common)
            .unwrap();

        // Circuit B: the same instance (identical gates, constants, and public-input count, so
        // in practice the same circuit digest — generators leave no trace in the digest), but a
        // different generator set: a 3-generator chain. If it shares circuit A's cache entry the
        // structural validation must reject A's schedule and fall back to dynamic scheduling; if
        // the digests happen to differ it simply records its own schedule. Either way this first
        // generation must be fully dynamic and produce exactly the dynamic scheduler's witness.
        let mut builder_b =
            CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt_b = builder_b.constant(F::from_canonical_u64(SHARED_SALT));
        builder_b.register_public_input(salt_b);
        let entry_b = builder_b.add_virtual_target();
        let (dispatches_b, executions_b, chain_out_b) =
            add_reversed_chain(&mut builder_b, entry_b, 3);
        let circuit_b = builder_b.build::<C>();

        let mut inputs_b = PartialWitness::new();
        inputs_b
            .set_target(entry_b, F::from_canonical_u64(5))
            .unwrap();

        let dynamic_witness_b = generate_partial_witness_dynamic(
            inputs_b.clone(),
            &circuit_b.prover_only,
            &circuit_b.common,
        )
        .unwrap();
        let dynamic_witness_b2 = generate_partial_witness_dynamic(
            inputs_b.clone(),
            &circuit_b.prover_only,
            &circuit_b.common,
        )
        .unwrap();
        let mask_b = deterministic_slots(&dynamic_witness_b, &dynamic_witness_b2);
        reset_counters(&dispatches_b);
        reset_counters(&executions_b);

        let witness_b =
            generate_partial_witness(inputs_b, &circuit_b.prover_only, &circuit_b.common).unwrap();

        // 2 * 3 - 1 dispatches: the dynamic scheduler ran, not a (foreign) replay.
        assert_eq!(counter_values(&dispatches_b).iter().sum::<usize>(), 5);
        assert_eq!(counter_values(&executions_b), vec![1; 3]);
        assert_agree_on_deterministic_slots(&mask_b, &dynamic_witness_b, &witness_b);
        assert_eq!(
            witness_b.get_target(chain_out_b),
            F::from_canonical_u64(5 + 3),
        );

        // Circuit A must still replay its own recorded schedule afterwards.
        reset_counters(&dispatches_a);
        generate_partial_witness(inputs_a, &circuit_a.prover_only, &circuit_a.common).unwrap();
        assert_eq!(counter_values(&dispatches_a), vec![1; 2]);
    }

    #[test]
    fn precompiled_slot_writes_match_unprecompiled_replay() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt = builder.constant(F::from_canonical_u64(0x0D1F_F2E4_5A17_0005));
        builder.register_public_input(salt);
        let x = builder.add_virtual_target();
        // Mix real gate generators (arithmetic) with the counting chain.
        let x_squared = builder.mul(x, x);
        let y = builder.add(x_squared, salt);
        let (dispatches, _executions, chain_out) = add_reversed_chain(&mut builder, y, 4);
        builder.register_public_input(chain_out);
        let circuit = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs.set_target(x, F::from_canonical_u64(9)).unwrap();

        // Value-deterministic slot mask from two dynamic reference runs (unused public-input
        // wires are re-randomized on every generation, by design).
        let reference_a = generate_partial_witness_dynamic(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let reference_b = generate_partial_witness_dynamic(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let mask = deterministic_slots(&reference_a, &reference_b);

        // Recording run.
        generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common).unwrap();

        // Replay the recorded schedule twice: once with writes resolved through the live
        // representative map (the reference write path) and once through the precompiled slot
        // lists. A single dispatch per chain generator proves both runs actually replayed.
        reset_counters(&dispatches);
        let unprecompiled = generate_partial_witness_unprecompiled_writes(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        assert_eq!(counter_values(&dispatches), vec![1; 4]);

        reset_counters(&dispatches);
        let precompiled =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();
        assert_eq!(counter_values(&dispatches), vec![1; 4]);

        // The slot fast path must write exactly the slots the live representative map would:
        // identical values on every value-deterministic slot, and identical slot population
        // (None/Some agreement) everywhere, including the randomized slots.
        assert_agree_on_deterministic_slots(&mask, &unprecompiled, &precompiled);
        assert_eq!(unprecompiled.values.len(), precompiled.values.len());
        for (slot, (a, b)) in unprecompiled
            .values
            .iter()
            .zip(&precompiled.values)
            .enumerate()
        {
            assert_eq!(a.is_some(), b.is_some(), "slot {slot} population diverged");
        }
        assert_agree_on_deterministic_slots(&mask, &reference_a, &precompiled);
    }

    /// A run-once generator whose write signature is controlled by an external mode: mode 0
    /// writes `out_a`; mode 1 writes `out_b` (same write count, different target); mode 2
    /// writes `out_a` then `out_b` (different write count). Within a mode its writes are
    /// deterministic, so a diverged replay's dynamic re-run reproduces them exactly.
    #[derive(Debug)]
    struct FickleWriteGenerator {
        input: Target,
        out_a: Target,
        out_b: Target,
        mode: Arc<AtomicUsize>,
        run_calls: Arc<AtomicUsize>,
    }

    impl SimpleGenerator<F, D> for FickleWriteGenerator {
        fn id(&self) -> String {
            "FickleWriteGenerator".to_string()
        }

        fn dependencies(&self) -> Vec<Target> {
            vec![self.input]
        }

        fn run_once(
            &self,
            witness: &PartitionWitness<F>,
            out_buffer: &mut GeneratedValues<F>,
        ) -> Result<()> {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            let value = witness.get_target(self.input) + F::ONE;
            match self.mode.load(Ordering::Relaxed) {
                0 => out_buffer.set_target(self.out_a, value),
                1 => out_buffer.set_target(self.out_b, value),
                _ => {
                    out_buffer.set_target(self.out_a, value)?;
                    out_buffer.set_target(self.out_b, value.double())
                }
            }
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<()> {
            unreachable!("test generator is never serialized")
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<Self> {
            unreachable!("test generator is never deserialized")
        }
    }

    #[test]
    fn diverging_write_signatures_fall_back_to_dynamic_scheduling() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt = builder.constant(F::from_canonical_u64(0x0D1F_F2E4_5A17_0006));
        builder.register_public_input(salt);
        let input = builder.add_virtual_target();
        let out_a = builder.add_virtual_target();
        let out_b = builder.add_virtual_target();
        let mode = Arc::new(AtomicUsize::new(0));
        let run_calls = Arc::new(AtomicUsize::new(0));
        builder.add_simple_generator(FickleWriteGenerator {
            input,
            out_a,
            out_b,
            mode: Arc::clone(&mode),
            run_calls: Arc::clone(&run_calls),
        });
        let circuit = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs
            .set_target(input, F::from_canonical_u64(20))
            .unwrap();

        // Recording run in mode 0: the schedule records a one-write signature for `out_a`.
        run_calls.store(0, Ordering::Relaxed);
        let recording =
            generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        assert_eq!(recording.get_target(out_a), F::from_canonical_u64(21));
        assert_eq!(run_calls.load(Ordering::Relaxed), 1);

        // Undisturbed replay: the signature matches and the generator executes exactly once.
        run_calls.store(0, Ordering::Relaxed);
        let replayed =
            generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        assert_eq!(replayed.get_target(out_a), F::from_canonical_u64(21));
        assert_eq!(run_calls.load(Ordering::Relaxed), 1);

        // Same write count, different target: the signature check must reject the dispatch
        // before anything reaches the witness, and the dynamic fallback must re-run the
        // generator (two executions in this generation) and produce the mode-1 witness.
        mode.store(1, Ordering::Relaxed);
        run_calls.store(0, Ordering::Relaxed);
        let diverged =
            generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common)
                .unwrap();
        assert_eq!(run_calls.load(Ordering::Relaxed), 2);
        assert_eq!(diverged.get_target(out_b), F::from_canonical_u64(21));
        assert!(diverged.try_get_target(out_a).is_none());

        // The fallback's witness must match a fully dynamic execution of the same mode on
        // every value-deterministic slot.
        let dynamic_a = generate_partial_witness_dynamic(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let dynamic_b = generate_partial_witness_dynamic(
            inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let mask = deterministic_slots(&dynamic_a, &dynamic_b);
        assert_agree_on_deterministic_slots(&mask, &dynamic_a, &diverged);

        // Different write count diverges as well.
        mode.store(2, Ordering::Relaxed);
        run_calls.store(0, Ordering::Relaxed);
        let diverged =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();
        assert_eq!(run_calls.load(Ordering::Relaxed), 2);
        assert_eq!(diverged.get_target(out_a), F::from_canonical_u64(21));
        assert_eq!(diverged.get_target(out_b), F::from_canonical_u64(42));
    }

    #[test]
    fn replayed_write_conflicts_error_like_the_dynamic_path() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let salt = builder.constant(F::from_canonical_u64(0x0D1F_F2E4_5A17_0007));
        builder.register_public_input(salt);
        let entry = builder.add_virtual_target();
        let (dispatches, _executions, chain_out) = add_reversed_chain(&mut builder, entry, 1);
        let circuit = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs.set_target(entry, F::from_canonical_u64(10)).unwrap();

        // Record, and confirm the circuit replays (single dispatch for the chain generator).
        generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common).unwrap();
        reset_counters(&dispatches);
        generate_partial_witness(inputs.clone(), &circuit.prover_only, &circuit.common).unwrap();
        assert_eq!(counter_values(&dispatches), vec![1]);

        // Pre-set the chain output to a value conflicting with what the generator computes
        // (10 + 1): the write-once check must reject the generation on the dynamic path and
        // on the replayed slot fast path alike.
        let mut conflicting = inputs;
        conflicting
            .set_target(chain_out, F::from_canonical_u64(999))
            .unwrap();
        assert!(
            generate_partial_witness_dynamic(
                conflicting.clone(),
                &circuit.prover_only,
                &circuit.common,
            )
            .is_err()
        );
        assert!(
            generate_partial_witness(conflicting, &circuit.prover_only, &circuit.common).is_err()
        );
    }

    #[test]
    fn readiness_hint_preserves_incremental_witness_generator_fallback() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let trigger = builder.constant(F::from_canonical_u64(11));
        let early_output = builder.add_virtual_target();
        let final_output = builder.add_virtual_target();
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_generators(vec![WitnessGeneratorRef::new(IncrementalGenerator {
            trigger,
            early_output,
            final_output,
            run_calls: Arc::clone(&run_calls),
        })]);
        builder.register_public_inputs(&[early_output, final_output]);

        let circuit = builder.build::<C>();
        let witness =
            generate_partial_witness(PartialWitness::new(), &circuit.prover_only, &circuit.common)
                .unwrap();

        assert_eq!(witness.get_target(early_output), F::from_canonical_u64(7));
        assert_eq!(witness.get_target(final_output), F::from_canonical_u64(11));
        assert_eq!(run_calls.load(Ordering::Relaxed), 2);
    }
}

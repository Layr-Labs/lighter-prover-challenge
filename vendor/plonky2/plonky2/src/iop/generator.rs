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
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Given a `PartitionWitness` that has only inputs set, populates the rest of the witness using the
/// given set of generators.
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
    PendingPartitionWitness::start(inputs, prover_data, common_data)?.finish()
}

/// A resumable [`generate_partial_witness`]: the generator worklist can be run to quiescence on a
/// subset of the input targets, resumed as further inputs become available, and completed once all
/// inputs have been supplied.
///
/// [`Self::start`] seeds the initial inputs and runs the worklist to quiescence, [`Self::feed`]
/// populates additional input targets and resumes the same worklist, and [`Self::finish`] performs
/// the completeness check and returns the populated [`PartitionWitness`].
pub struct PendingPartitionWitness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    witness: PartitionWitness<'a, F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    unresolved_watches: Vec<usize>,
    generator_is_expired: Vec<bool>,
    remaining_generators: usize,
    pending_generator_indices: Vec<usize>,
    buffer: GeneratedValues<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Debug
    for PendingPartitionWitness<'_, F, C, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PendingPartitionWitness")
            .field("remaining_generators", &self.remaining_generators)
            .finish_non_exhaustive()
    }
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PendingPartitionWitness<'a, F, C, D>
{
    /// Seeds the given input targets and runs the generator worklist to quiescence, without
    /// requiring that every generator has run.
    pub fn start(
        inputs: PartialWitness<F>,
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Result<Self> {
        let config = &common_data.config;
        let generators = &prover_data.generators;
        let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

        let mut witness = PartitionWitness::new(
            config.num_wires,
            common_data.degree(),
            &prover_data.representative_map,
        );

        for (t, v) in inputs.target_values.into_iter() {
            witness.set_target(t, v)?;
        }

        // A simple generator can run once all of the distinct representatives it watches have
        // values. Derive those unresolved counts from the existing watcher index so this remains
        // local witness state and does not add anything to serialized prover data.
        let mut unresolved_watches = vec![0usize; generators.len()];
        for (&watch, watchers) in generator_indices_by_watches {
            if !witness.is_representative_set(watch) {
                for &generator_idx in watchers {
                    unresolved_watches[generator_idx] += 1;
                }
            }
        }

        // Build a list of "pending" generators which are queued to be run. Initially, all
        // generators are queued.
        let pending_generator_indices: Vec<_> = (0..generators.len()).collect();

        // We also track a list of "expired" generators which have already returned false.
        let generator_is_expired = vec![false; generators.len()];
        let remaining_generators = generators.len();

        let mut pending = Self {
            witness,
            prover_data,
            unresolved_watches,
            generator_is_expired,
            remaining_generators,
            pending_generator_indices,
            buffer: GeneratedValues::empty(),
        };
        pending.run_generator_worklist()?;

        Ok(pending)
    }

    /// Populates additional input targets, wakes the not-yet-expired generators watching each newly
    /// populated representative, and runs the generator worklist to quiescence again.
    pub fn feed(&mut self, inputs: PartialWitness<F>) -> Result<()> {
        let prover_data = self.prover_data;
        let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

        let mut new_target_reps = Vec::with_capacity(inputs.target_values.len());
        for (t, v) in inputs.target_values.into_iter() {
            let reps = self.witness.set_target_returning_rep(t, v)?;
            new_target_reps.extend(reps);
        }

        // Enqueue unfinished generators that were watching one of the newly populated targets.
        for watch in new_target_reps {
            let opt_watchers = generator_indices_by_watches.get(&watch);
            if let Some(watchers) = opt_watchers {
                for &watching_generator_idx in watchers {
                    if !self.generator_is_expired[watching_generator_idx] {
                        debug_assert_ne!(self.unresolved_watches[watching_generator_idx], 0);
                        self.unresolved_watches[watching_generator_idx] -= 1;
                        self.pending_generator_indices.push(watching_generator_idx);
                    }
                }
            }
        }

        self.run_generator_worklist()
    }

    /// Runs the generator worklist to quiescence, checks that every generator has run, and returns
    /// the populated witness.
    pub fn finish(mut self) -> Result<PartitionWitness<'a, F>> {
        self.run_generator_worklist()?;

        if self.remaining_generators != 0 {
            return Err(anyhow!(
                "{} generators weren't run",
                self.remaining_generators
            ));
        }

        Ok(self.witness)
    }

    fn run_generator_worklist(&mut self) -> Result<()> {
        let prover_data = self.prover_data;
        let generators = &prover_data.generators;
        let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

        // Keep running generators until we fail to make progress.
        let mut pending_generator_indices = core::mem::take(&mut self.pending_generator_indices);
        while !pending_generator_indices.is_empty() {
            let mut next_pending_generator_indices = Vec::new();

            for &generator_idx in &pending_generator_indices {
                if self.generator_is_expired[generator_idx] {
                    continue;
                }

                let finished = generators[generator_idx].0.run_with_ready_hint(
                    &self.witness,
                    &mut self.buffer,
                    self.unresolved_watches[generator_idx] == 0,
                );
                if finished {
                    self.generator_is_expired[generator_idx] = true;
                    self.remaining_generators -= 1;
                }

                // Merge any generated values into our witness, and get a list of newly-populated
                // targets' representatives.
                let mut new_target_reps = Vec::with_capacity(self.buffer.target_values.len());
                for (t, v) in self.buffer.target_values.drain(..) {
                    let reps = self.witness.set_target_returning_rep(t, v)?;
                    new_target_reps.extend(reps);
                }

                // Enqueue unfinished generators that were watching one of the newly populated
                // targets.
                for watch in new_target_reps {
                    let opt_watchers = generator_indices_by_watches.get(&watch);
                    if let Some(watchers) = opt_watchers {
                        for &watching_generator_idx in watchers {
                            if !self.generator_is_expired[watching_generator_idx] {
                                debug_assert_ne!(
                                    self.unresolved_watches[watching_generator_idx],
                                    0
                                );
                                self.unresolved_watches[watching_generator_idx] -= 1;
                                next_pending_generator_indices.push(watching_generator_idx);
                            }
                        }
                    }
                }
            }

            pending_generator_indices = next_pending_generator_indices;
        }
        self.pending_generator_indices = pending_generator_indices;

        Ok(())
    }
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
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::prover::prove_with_partition_witness;
    use crate::util::timing::TimingTree;

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

    fn square_circuit() -> (CircuitData<F, C, D>, Target, Target) {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let input = builder.add_virtual_target();
        let square = builder.mul(input, input);
        builder.register_public_input(square);
        (builder.build::<C>(), input, square)
    }

    fn random_value_representatives(
        witness: &PartitionWitness<F>,
        circuit: &CircuitData<F, C, D>,
    ) -> HashSet<usize> {
        let mut representatives = HashSet::new();
        for generator in &circuit.prover_only.generators {
            if generator.0.id() == "RandomValueGenerator" {
                let mut serialized = Vec::new();
                generator
                    .0
                    .serialize(&mut serialized, &circuit.common)
                    .unwrap();
                let mut buffer = Buffer::new(&serialized);
                let target = buffer.read_target().unwrap();
                representatives.insert(witness.representative_map[witness.target_index(target)]);
            }
        }
        representatives
    }

    #[test]
    fn pending_partition_witness_matches_single_shot_witness_generation() {
        let config = CircuitConfig::standard_recursion_config();

        let mut inner_builder = CircuitBuilder::<F, D>::new(config.clone());
        let inner_input = inner_builder.add_virtual_target();
        let mut inner_value = inner_input;
        for _ in 0..256 {
            inner_value = inner_builder.mul(inner_value, inner_input);
        }
        inner_builder.register_public_input(inner_value);
        let inner = inner_builder.build::<C>();

        let mut first_inputs = PartialWitness::new();
        first_inputs
            .set_target(inner_input, F::from_canonical_u64(3))
            .unwrap();
        let first_proof = inner.prove(first_inputs).unwrap();
        let mut second_inputs = PartialWitness::new();
        second_inputs
            .set_target(inner_input, F::from_canonical_u64(5))
            .unwrap();
        let second_proof = inner.prove(second_inputs).unwrap();

        let mut builder = CircuitBuilder::<F, D>::new(config);
        let first_proof_target = builder.add_virtual_proof_with_pis(&inner.common);
        let second_proof_target = builder.add_virtual_proof_with_pis(&inner.common);
        let inner_verifier_data = builder.constant_verifier_data(&inner.verifier_only);
        builder.verify_proof::<C>(&first_proof_target, &inner_verifier_data, &inner.common);
        builder.verify_proof::<C>(&second_proof_target, &inner_verifier_data, &inner.common);
        let outer = builder.build::<C>();

        let mut single_shot_inputs = PartialWitness::new();
        single_shot_inputs
            .set_proof_with_pis_target(&first_proof_target, &first_proof)
            .unwrap();
        single_shot_inputs
            .set_proof_with_pis_target(&second_proof_target, &second_proof)
            .unwrap();
        let single_shot_witness =
            generate_partial_witness(single_shot_inputs, &outer.prover_only, &outer.common)
                .unwrap();

        let mut early_inputs = PartialWitness::new();
        early_inputs
            .set_proof_with_pis_target(&first_proof_target, &first_proof)
            .unwrap();
        let mut pending =
            PendingPartitionWitness::start(early_inputs, &outer.prover_only, &outer.common)
                .unwrap();
        pending.feed(PartialWitness::new()).unwrap();
        let mut late_inputs = PartialWitness::new();
        late_inputs
            .set_proof_with_pis_target(&second_proof_target, &second_proof)
            .unwrap();
        pending.feed(late_inputs).unwrap();
        let pending_witness = pending.finish().unwrap();

        let unconstrained = random_value_representatives(&single_shot_witness, &outer);
        // The sparse bitmap store keeps its value slots private; compare the logical
        // per-representative Option view through the guarded accessor instead. One value slot
        // exists per representative-map entry.
        assert_eq!(
            single_shot_witness.representative_map.len(),
            pending_witness.representative_map.len()
        );
        for representative in 0..single_shot_witness.representative_map.len() {
            if unconstrained.contains(&representative) {
                continue;
            }
            assert_eq!(
                single_shot_witness.representative_value(representative),
                pending_witness.representative_value(representative),
                "witness values diverge at representative {representative}"
            );
        }

        let single_shot_proof = prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            single_shot_witness,
            &mut TimingTree::default(),
        )
        .unwrap();
        let pending_proof = prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            pending_witness,
            &mut TimingTree::default(),
        )
        .unwrap();
        outer.verify(single_shot_proof).unwrap();
        outer.verify(pending_proof).unwrap();
    }

    #[test]
    fn pending_finish_without_input_reports_unrun_generators() {
        let (circuit, _input, _square) = square_circuit();

        let pending = PendingPartitionWitness::start(
            PartialWitness::new(),
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let error = pending
            .finish()
            .expect_err("finishing without the arithmetic input must fail");
        assert!(
            error.to_string().contains("generators weren't run"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn pending_feed_rejects_contradictory_input_values() {
        let (circuit, input, _square) = square_circuit();

        let mut initial_inputs = PartialWitness::new();
        initial_inputs
            .set_target(input, F::from_canonical_u64(3))
            .unwrap();
        let mut pending = PendingPartitionWitness::start(
            initial_inputs,
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let mut conflicting_inputs = PartialWitness::new();
        conflicting_inputs
            .set_target(input, F::from_canonical_u64(4))
            .unwrap();
        let error = pending
            .feed(conflicting_inputs)
            .expect_err("feeding a contradictory input value must fail");
        assert!(
            error.to_string().contains("set twice with different values"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn pending_feed_rejects_conflicts_with_generated_values() {
        let (circuit, input, square) = square_circuit();

        let mut initial_inputs = PartialWitness::new();
        initial_inputs
            .set_target(input, F::from_canonical_u64(3))
            .unwrap();
        let mut pending = PendingPartitionWitness::start(
            initial_inputs,
            &circuit.prover_only,
            &circuit.common,
        )
        .unwrap();
        let mut late_inputs = PartialWitness::new();
        late_inputs
            .set_target(square, F::from_canonical_u64(10))
            .unwrap();
        let error = pending
            .feed(late_inputs)
            .expect_err("feeding a value conflicting with a generated value must fail");
        assert!(
            error.to_string().contains("set twice with different values"),
            "unexpected error: {error}"
        );
    }
}

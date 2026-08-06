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

/// Runs the given pending generators, and transitively any generator watching a newly populated
/// representative, until no further progress can be made.
fn run_generator_worklist<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &mut PartitionWitness<F>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    unresolved_watches: &mut [usize],
    generator_is_expired: &mut [bool],
    remaining_generators: &mut usize,
    mut pending_generator_indices: Vec<usize>,
) -> Result<()> {
    let generators = &prover_data.generators;
    let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

    let mut buffer = GeneratedValues::empty();

    // Keep running generators until we fail to make progress.
    while !pending_generator_indices.is_empty() {
        let mut next_pending_generator_indices = Vec::new();

        for &generator_idx in &pending_generator_indices {
            if generator_is_expired[generator_idx] {
                continue;
            }

            let finished = generators[generator_idx].0.run_with_ready_hint(
                witness,
                &mut buffer,
                unresolved_watches[generator_idx] == 0,
            );
            if finished {
                generator_is_expired[generator_idx] = true;
                *remaining_generators -= 1;
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

    Ok(())
}

/// Resumable witness generation: [`Self::start`] seeds an initial set of inputs and runs every
/// generator that can already make progress, each [`Self::feed`] sets newly available inputs and
/// resumes only the generators watching them, and [`Self::finish`] performs the same completeness
/// check as [`generate_partial_witness`].
///
/// Generators are deterministic functions of their watched values, so splitting the same inputs
/// across `start`/`feed` calls in any order yields a witness identical to the single-shot path.
pub struct PendingPartitionWitness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    witness: PartitionWitness<'a, F>,
    unresolved_watches: Vec<usize>,
    generator_is_expired: Vec<bool>,
    remaining_generators: usize,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
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
    /// Seeds `inputs` and runs generators to quiescence. Unlike [`generate_partial_witness`],
    /// generators whose watched values are still missing are left pending rather than being an
    /// error.
    pub fn start(
        inputs: PartialWitness<F>,
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Result<Self> {
        let generators = &prover_data.generators;
        let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

        let mut witness = PartitionWitness::new(
            common_data.config.num_wires,
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
            if witness.values[watch].is_none() {
                for &generator_idx in watchers {
                    unresolved_watches[generator_idx] += 1;
                }
            }
        }

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();

        // Initially, all generators are queued.
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut unresolved_watches,
            &mut generator_is_expired,
            &mut remaining_generators,
            (0..generators.len()).collect(),
        )?;

        Ok(Self {
            witness,
            unresolved_watches,
            generator_is_expired,
            remaining_generators,
            prover_data,
        })
    }

    /// Sets newly available inputs and resumes witness generation. Only the unfinished watchers of
    /// newly populated representatives are queued; every other unfinished generator was already
    /// run to quiescence and cannot make progress without new values.
    pub fn feed(&mut self, inputs: PartialWitness<F>) -> Result<()> {
        let generator_indices_by_watches = &self.prover_data.generator_indices_by_watches;

        let mut pending_generator_indices = Vec::new();
        for (t, v) in inputs.target_values.into_iter() {
            if let Some(watch) = self.witness.set_target_returning_rep(t, v)? {
                if let Some(watchers) = generator_indices_by_watches.get(&watch) {
                    for &watching_generator_idx in watchers {
                        if !self.generator_is_expired[watching_generator_idx] {
                            debug_assert_ne!(self.unresolved_watches[watching_generator_idx], 0);
                            self.unresolved_watches[watching_generator_idx] -= 1;
                            pending_generator_indices.push(watching_generator_idx);
                        }
                    }
                }
            }
        }

        run_generator_worklist(
            &mut self.witness,
            self.prover_data,
            &mut self.unresolved_watches,
            &mut self.generator_is_expired,
            &mut self.remaining_generators,
            pending_generator_indices,
        )
    }

    /// Returns the fully populated witness, or an error if some generators still couldn't run.
    pub fn finish(self) -> Result<PartitionWitness<'a, F>> {
        if self.remaining_generators != 0 {
            return Err(anyhow!(
                "{} generators weren't run",
                self.remaining_generators
            ));
        }

        Ok(self.witness)
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::noop::NoopGate;
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

    #[test]
    fn pending_partition_witness_matches_single_shot_for_recursive_circuit() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();

        // Inner circuit: expose x^2 as a public input.
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let x = builder.add_virtual_target();
        let x_squared = builder.mul(x, x);
        builder.register_public_input(x_squared);
        for _ in 0..1_000 {
            builder.add_gate(NoopGate, vec![]);
        }
        let inner = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs.set_target(x, F::from_canonical_u64(3))?;
        let inner_proof_a = inner.prove(inputs)?;
        let mut inputs = PartialWitness::new();
        inputs.set_target(x, F::from_canonical_u64(5))?;
        let inner_proof_b = inner.prove(inputs)?;

        // Outer circuit: verify two independent inner proofs, mirroring a chain step's
        // tx-proof/cyclic-proof pair.
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let proof_target_a = builder.add_virtual_proof_with_pis(&inner.common);
        let proof_target_b = builder.add_virtual_proof_with_pis(&inner.common);
        let verifier_data = builder.constant_verifier_data(&inner.verifier_only);
        builder.verify_proof::<C>(&proof_target_a, &verifier_data, &inner.common);
        builder.verify_proof::<C>(&proof_target_b, &verifier_data, &inner.common);
        builder.register_public_inputs(&proof_target_a.public_inputs);
        builder.register_public_inputs(&proof_target_b.public_inputs);
        let outer = builder.build::<C>();

        let mut single_shot_inputs = PartialWitness::new();
        single_shot_inputs.set_proof_with_pis_target(&proof_target_a, &inner_proof_a)?;
        single_shot_inputs.set_proof_with_pis_target(&proof_target_b, &inner_proof_b)?;
        let single_shot = generate_partial_witness(
            single_shot_inputs.clone(),
            &outer.prover_only,
            &outer.common,
        )?;
        // Every witness position is deterministic except the outputs of the circuit's
        // `RandomValueGenerator`s (unused public-input-gate wires): a second single-shot run
        // isolates exactly those positions.
        let single_shot_repeat =
            generate_partial_witness(single_shot_inputs, &outer.prover_only, &outer.common)?;
        let num_random_generators = outer
            .prover_only
            .generators
            .iter()
            .filter(|generator| generator.0.id() == "RandomValueGenerator")
            .count();

        let mut early_inputs = PartialWitness::new();
        early_inputs.set_proof_with_pis_target(&proof_target_a, &inner_proof_a)?;
        let mut pending =
            PendingPartitionWitness::start(early_inputs, &outer.prover_only, &outer.common)?;
        // A feed with no new targets must be a no-op.
        pending.feed(PartialWitness::new())?;
        let mut late_inputs = PartialWitness::new();
        late_inputs.set_proof_with_pis_target(&proof_target_b, &inner_proof_b)?;
        pending.feed(late_inputs)?;
        let two_phase = pending.finish()?;

        let mut nondeterministic_positions = 0usize;
        for ((single, repeat), split) in single_shot
            .values
            .iter()
            .zip(&single_shot_repeat.values)
            .zip(&two_phase.values)
        {
            if single == repeat {
                assert_eq!(single, split);
            } else {
                nondeterministic_positions += 1;
            }
        }
        assert!(
            nondeterministic_positions <= num_random_generators,
            "{nondeterministic_positions} nondeterministic positions exceed the {num_random_generators} random generators"
        );

        let single_shot_proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            single_shot,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        let two_phase_proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            two_phase,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        outer.verify(single_shot_proof)?;
        outer.verify(two_phase_proof)
    }

    #[test]
    fn pending_partition_witness_finish_and_feed_errors() -> Result<()> {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let product = builder.mul(a, b);
        builder.register_public_input(product);
        let circuit = builder.build::<C>();

        let mut early_inputs = PartialWitness::new();
        early_inputs.set_target(a, F::from_canonical_u64(3))?;

        // Finishing before all inputs are fed reports the unrun generators.
        let pending = PendingPartitionWitness::start(
            early_inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )?;
        let error = pending.finish().unwrap_err();
        assert!(
            error.to_string().contains("generators weren't run"),
            "unexpected finish error: {error:?}"
        );

        // Feeding a value contradicting an already-set target fails.
        let mut pending = PendingPartitionWitness::start(
            early_inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )?;
        let mut contradictory_inputs = PartialWitness::new();
        contradictory_inputs.set_target(a, F::from_canonical_u64(4))?;
        assert!(pending.feed(contradictory_inputs).is_err());

        // Feeding the missing input completes witness generation.
        let mut pending =
            PendingPartitionWitness::start(early_inputs, &circuit.prover_only, &circuit.common)?;
        let mut late_inputs = PartialWitness::new();
        late_inputs.set_target(b, F::from_canonical_u64(5))?;
        pending.feed(late_inputs)?;
        let witness = pending.finish()?;
        assert_eq!(witness.get_target(product), F::from_canonical_u64(15));

        Ok(())
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

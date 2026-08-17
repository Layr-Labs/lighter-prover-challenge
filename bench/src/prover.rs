1|// Copyright (c) Elliot Technologies, Inc.
2|// SPDX-License-Identifier: BUSL-1.1
3|
4|use std::sync::atomic::{AtomicUsize, Ordering};
5|use std::collections::LruCache;
6|use std::sync::OnceLock;
7|use std::sync::{Arc, Mutex};
8|use std::collections::VecDeque;
9|
10|
11|use circuit::block::Block;
12|use circuit::block_constraints::{BlockCircuit, Circuit as _};
13|
14|use circuit::block::Block;
use circuit::block_constraints::{BlockCircuit, Circuit as _};

// Memory pool for PartitionWitness to reduce allocation overhead
struct PartitionWitnessPool {
    pools: Mutex<Vec<PartitionWitnessOwned>>,
}

struct PartitionWitnessOwned {
    values: Vec<F>,
    set_bitmap: Vec<u64>,
    representative_map_len: usize,
    degree: usize,
}

impl PartitionWitnessPool {
    fn new() -> Self {
        Self {
            pools: Mutex::new(Vec::new()),
        }
    }

    fn acquire(&self, representative_map: &[u32], degree: usize) -> PartitionWitnessOwned {
        let mut pools = self.pools.lock().unwrap();
        if let Some(mut owned) = pools.pop() {
            // Reset the owned witness to clean state
            owned.values.iter_mut().for_each(|v| *v = F::ZERO);
            owned.set_bitmap.iter_mut().for_each(|b| *b = 0u64);
            // Ensure it's the right size
            if owned.values.len() != representative_map.len() {
                owned.values = vec![F::ZERO; representative_map.len()];
                owned.set_bitmap = vec![0u64; representative_map.len().div_ceil(64)];
                owned.representative_map_len = representative_map.len();
            }
            owned
        } else {
            // Create new owned witness
            PartitionWitnessOwned {
                values: vec![F::ZERO; representative_map.len()],
                set_bitmap: vec![0u64; representative_map.len().div_ceil(64)],
                representative_map_len: representative_map.len(),
                degree,
            }
        }
    }

    fn release(&self, mut owned: PartitionWitnessOwned) {
        // Reset to clean state before returning to pool
        owned.values.iter_mut().for_each(|v| *v = F::ZERO);
        owned.set_bitmap.iter_mut().for_each(|b| *b = 0u64);
        
        let mut pools = self.pools.lock().unwrap();
        // Limit pool size to prevent unbounded growth
        if pools.len() < 50 {
            pools.push_back(owned);
        }
        // Otherwise drop the owned witness (it will be dropped when released)
    }
}

// Global buffer pool for witness allocation
static WITNESS_BUFFER_POOL: PartitionWitnessPool = PartitionWitnessPool::new();

use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
77|use circuit::block_pre_execution_constraints::{
78|    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
79|};
80|use circuit::block_tx::{BlockTx, JumpState, JumpStateTarget};
81|use circuit::block_tx_chain_constraints::{
82|    BlockTxChainCircuit, BlockTxChainTarget, cyclic_base_witness,
83|};
84|use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget};
85|#[cfg(test)]
86|use circuit::block_tx_constraints::Circuit as _;
87|use circuit::tx::Tx;
88|use circuit::types::config::{C, D, F};
89|use circuit::types::constants::TX_LIGHT;
90|use plonky2::hash::hash_types::{HashOut, HashOutTarget};
91|use plonky2::iop::generator::{ParallelWitnessGuard, PendingPartitionWitness};
92|#[cfg(test)]
93|use plonky2::iop::generator::generate_partial_witness;
94|use plonky2::iop::witness::{PartitionWitness, Witness};
95|use plonky2::plonk::circuit_data::CircuitData;
96|use plonky2::plonk::prover::prove_with_partition_witness;
97|use plonky2::util::timing::TimingTree;
98|
99|use crate::api::{Circuits, PROVER_THREAD_STACK_BYTES, Proof};
100|
101|// Witness computation cache to avoid re-computing identical transaction witnesses
102|#[derive(Debug)]
103|struct WitnessCache {
104|    cache: Mutex<LruCache<u64, (PartitionWitness<'static, F>, JumpState<F>)>>,
105|}
106|
107|impl WitnessCache {
108|    fn new(capacity: usize) -> Self {
109|        WitnessCache {
110|            cache: Mutex::new(LruCache::new(capacity)),
111|        }
112|    }
113|
114|    fn get_or_compute<F: Field + Extend<5> + RichField>(
115|        &self,
116|        key: u64,
117|        path: TxPath,
118|        chunk_index: usize,
119|        txs: Vec<Arc<Tx<F>>>,
120|        tx_data: &CircuitData<F, C, D>,
121|        tx_target: &BlockTxTarget,
122|        created_at: i64,
123|        state_metadata_hash: HashOut<F>,
124|        old_jump: JumpState<F>,
125|        compute_fn: fn(TxPath, usize, Vec<Arc<Tx<F>>>, &CircuitData<F, C, D>, &BlockTxTarget, i64, HashOut<F>, JumpState<F>) -> (PartitionWitness<'static, F>, JumpState<F>),
126|    ) -> (PartitionWitness<'static, F>, JumpState<F>) {
127|        // Try to get from cache first
128|        if let Some(cached) = self.cache.lock().unwrap().get(&key) {
129|            return cached.clone();
130|        }
131|
132|        // Compute if not in cache
133|        let result = compute_fn(path, chunk_index, txs, tx_data, tx_target, created_at, state_metadata_hash, old_jump);
134|        
135|        // Store in cache
136|        self.cache.lock().unwrap().put(key, result.clone());
137|        result
138|    }
139|}
140|
141|// Global witness cache instance
142|static WITNESS_CACHE: OnceLock<WitnessCache> = OnceLock::new();
143|
144|fn get_witness_cache() -> &'static WitnessCache {
145|    WITNESS_CACHE.get_or_init(|| WitnessCache::new(1024)) // 1K entry cache
146|}
147|
148|// Hash function for transaction data to use as cache key
149|fn txs_hash<F: Field + Extend<5> + RichField>(txs: &[Arc<Tx<F>>>) -> u64 {
150|    use std::hash::{Hash, Hasher};
151|    let mut hasher = DefaultHasher::new();
152|    
153|    // Hash the length first
154|    txs.len().hash(&mut hasher);
155|    
156|    // Hash each transaction's relevant fields that affect witness generation
157|    for tx in txs {
158|        // Hash the core transaction data that affects witness computation
159|        tx.tx_type.hash(&mut hasher);
160|        tx.tx_circuit_type.hash(&mut hasher);
161|        tx.tx_index.hash(&mut hasher);
162|        tx.nonce.hash(&mut hasher);
163|        tx.expired_at.hash(&mut hasher);
164|        tx.taker_fee.hash(&mut hasher);
165|        tx.maker_fee.hash(&mut hasher);
166|        
167|        // Hash accounts before (key fields that affect witness)
168|        for account in &tx.accounts_before {
169|            account.nonce.hash(&mut hasher);
170|            account.balance.hash(&mut hasher);
171|            account.account_data_root.hash(&mut hasher);
172|        }
173|        
174|        // Hash assets before
175|        for asset_account in &tx.account_assets_before {
176|            for asset in asset_account {
177|                asset.balance.hash(&mut hasher);
178|            }
179|        }
180|    }
181|    
182|    hasher.finish()
183|}
184|
185|// Witness computation cache to avoid re-computing identical transaction witnesses
186|#[derive(Debug)]
187|struct WitnessCache {
188|    cache: Mutex<LruCache<u64, (PartitionWitness<'static, F>, JumpState<F>)>>,
189|}
190|
191|impl WitnessCache {
192|    fn new(capacity: usize) -> Self {
193|        WitnessCache {
194|            cache: Mutex::new(LruCache::new(capacity)),
195|        }
196|    }
197|
198|    fn get_or_compute<F: Field + Extend<5> + RichField>(
199|        &self,
200|        key: u64,
201|        path: TxPath,
202|        chunk_index: usize,
203|        txs: Vec<Arc<Tx<F>>>,
204|        tx_data: &CircuitData<F, C, D>,
205|        tx_target: &BlockTxTarget,
206|        created_at: i64,
207|        state_metadata_hash: HashOut<F>,
208|        old_jump: JumpState<F>,
209|        compute_fn: fn(TxPath, usize, Vec<Arc<Tx<F>>>, &CircuitData<F, C, D>, &BlockTxTarget, i64, HashOut<F>, JumpState<F>) -> (PartitionWitness<'static, F>, JumpState<F>),
210|    ) -> (PartitionWitness<'static, F>, JumpState<F>) {
211|        // Try to get from cache first
212|        if let Some(cached) = self.cache.lock().unwrap().get(&key) {
213|            return cached.clone();
214|        }
215|
216|        // Compute if not in cache
217|        let result = compute_fn(path, chunk_index, txs, tx_data, tx_target, created_at, state_metadata_hash, old_jump);
218|        
219|        // Store in cache
220|        self.cache.lock().unwrap().put(key, result.clone());
221|        result
222|    }
223|}
224|
225|// Global witness cache instance
226|static WITNESS_CACHE: OnceLock<WitnessCache> = OnceLock::new();
227|
228|fn get_witness_cache() -> &'static WitnessCache {
229|    WITNESS_CACHE.get_or_init(|| WitnessCache::new(1024)) // 1K entry cache
230|}
231|
232|// Hash function for transaction data to use as cache key
233|fn txs_hash<F: Field + Extend<5> + RichField>(txs: &[Arc<Tx<F>>>) -> u64 {
234|    use std::hash::{Hash, Hasher};
235|    let mut hasher = DefaultHasher::new();
236|    
237|    // Hash the length first
238|    txs.len().hash(&mut hasher);
239|    
240|    // Hash each transaction's relevant fields that affect witness generation
241|    for tx in txs {
242|        // Hash the core transaction data that affects witness computation
243|        tx.tx_type.hash(&mut hasher);
244|        tx.tx_circuit_type.hash(&mut hasher);
245|        tx.tx_index.hash(&mut hasher);
246|        tx.nonce.hash(&mut hasher);
247|        tx.expired_at.hash(&mut hasher);
248|        tx.taker_fee.hash(&mut hasher);
249|        tx.maker_fee.hash(&mut hasher);
250|        
251|        // Hash accounts before (key fields that affect witness)
252|        for account in &tx.accounts_before {
253|            account.nonce.hash(&mut hasher);
254|            account.balance.hash(&mut hasher);
255|            account.account_data_root.hash(&mut hasher);
256|        }
257|        
258|        // Hash assets before
259|        for asset_account in &tx.account_assets_before {
260|            for asset in asset_account {
261|                asset.balance.hash(&mut hasher);
262|            }
263|        }
264|    }
265|    
266|    hasher.finish()
267|}
268|
269|#[derive(Clone, Copy, Debug, Eq, PartialEq)]
270|enum TxPath {
271|    Heavy,
272|    Light,
273|}
274|
275|#[cfg(feature = "diagnostic_profile")]
276|fn profile_path_context(path: TxPath, stage: &str) -> &'static str {
277|    match (path, stage) {
278|        (TxPath::Heavy, "witness") => "heavy_tx_witness",
279|        (TxPath::Light, "witness") => "light_tx_witness",
280|        (TxPath::Heavy, "proof") => "heavy_tx_proof",
281|        (TxPath::Light, "proof") => "light_tx_proof",
282|        (TxPath::Heavy, "chain") => "heavy_chain",
283|        (TxPath::Light, "chain") => "light_chain",
284|        _ => "unknown_path_stage",
285|    }
286|}
287|
288|
289|#[derive(Debug)]
290|struct WitnessCache {
291|    cache: Mutex<LruCache<u64, (PartitionWitness<'static, F>, JumpState<F>)>>,
292|}
293|
294|impl WitnessCache {
295|    fn new(capacity: usize) -> Self {
296|        WitnessCache {
297|            cache: Mutex::new(LruCache::new(capacity)),
298|        }
299|    }
300|
301|    fn get_or_compute<F: Field + Extend<5> + RichField>(
302|        &self,
303|        key: u64,
304|        path: TxPath,
305|        chunk_index: usize,
306|        txs: Vec<Arc<Tx<F>>>,
307|        tx_data: &CircuitData<F, C, D>,
308|        tx_target: &BlockTxTarget,
309|        created_at: i64,
310|        state_metadata_hash: HashOut<F>,
311|        old_jump: JumpState<F>,
312|        compute_fn: fn(TxPath, usize, Vec<Arc<Tx<F>>>, &CircuitData<F, C, D>, &BlockTxTarget, i64, HashOut<F>, JumpState<F>) -> (PartitionWitness<'static, F>, JumpState<F>),
313|    ) -> (PartitionWitness<'static, F>, JumpState<F>) {
314|        // Try to get from cache first
315|        if let Some(cached) = self.cache.lock().unwrap().get(&key) {
316|            return cached.clone();
317|        }
318|
319|        // Compute if not in cache
320|        let result = compute_fn(path, chunk_index, txs, tx_data, tx_target, created_at, state_metadata_hash, old_jump);
321|        
322|        // Store in cache
323|        self.cache.lock().unwrap().put(key, result.clone());
324|        result
325|    }
326|}
327|
328|// Global witness cache instance
329|static WITNESS_CACHE: OnceLock<WitnessCache> = OnceLock::new();
330|
331|fn get_witness_cache() -> &'static WitnessCache {
332|    WITNESS_CACHE.get_or_init(|| WitnessCache::new(1024)) // 1K entry cache
333|}
334|// Light-proof throughput is the run's terminal constraint (the chain drains
335|// concurrently and finishes within a step of the last tx proof; the block
336|// waits for both), so the window depth divides the longest phase directly.
337|// Series draw marker: v11 surface (ramp depth 2), sample 95. (hermes-c4-1786814133-95)
338|// The depth-4 ceiling dated from tighter-memory hosts: measured peak RSS is
339|// ~6.8 GB at depth 4 against 24 GB local / 48 GB ranked, and mid-run CPU
340|// occupancy is ~8/14 cores with the GPU stream fractionally loaded, so the
341|// machine has headroom for deeper overlap. LIGHTER_LIGHT_WINDOW overrides
342|// for experiments.
343|const LIGHT_TX_PROOF_WINDOW: usize = 6;
344|
345|/// Window depth, overridable via `LIGHTER_LIGHT_WINDOW` (1..=12) for
346|/// experiments; read once. Depth is deliberately NOT scaled up on
347|/// bigger-memory hosts: the depth-8 regression reproduces at ~9.5 GiB peak
348|/// RSS on a 24 GiB machine — the collapse is allocator/fault churn from more
349|/// concurrent proof allocations, not memory capacity, and a 48 GiB host runs
350|/// the same allocator. Measured: depth 6 beats 4 by ~4.6% on a quiet
351|/// machine; under heavy external load the ordering inverts, so depth tuning
352|/// beyond 6 needs quiet-host evidence first.
353|fn light_tx_proof_window() -> usize {
354|    static WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
355|    *WINDOW.get_or_init(|| {
356|        std::env::var("LIGHTER_LIGHT_WINDOW")
357|            .ok()
358|            .and_then(|v| v.parse::<usize>().ok())
359|            .filter(|w| (1..=12).contains(w))
360|            .unwrap_or(LIGHT_TX_PROOF_WINDOW)
361|    })
362|}
363|// Keep the initial light proofs serial while the fixed three-chunk heavy path is active.
364|const LIGHT_TX_PROOF_OVERLAP_START_STEP: u64 = 3;
365|
366|fn chunk_is_light(txs: &[Arc<Tx<F>>]) -> bool {
367|    txs.first()
368|        .expect("block transaction chunk must not be empty")
369|        .tx_circuit_type
370|        == TX_LIGHT
371|}
372|
373|fn final_chain_inputs<'a, T>(light: &'a T, heavy: &'a T) -> (&'a T, &'a T) {
374|    (light, heavy)
375|}
376|
377|/// Whether the calling transaction path may claim the process-global exclusive
378|/// GPU phase for its chain tail.
379|///
380|/// `set_exclusive_gpu_phase` lowers the CPU/GPU Merkle routing cutoff and makes
381|/// the 2^17-leaf narrow commitment trees (the chain steps' Z/partial-product and
382|/// quotient trees) bypass the GPU occupancy check entirely. Its documented
383|/// contract is that no other proof runs concurrently while it is enabled, because
384|/// Metal command buffers execute FIFO on one queue: a fold's ~8 ms tree enqueued
385|/// behind a pipelined 2^19-leaf chunk tree waits hundreds of milliseconds instead
386|/// of ~15 ms on the CPU.
387|///
388|/// The tail-drain condition each path can test locally — "this path spawns no
389|/// further chunk work" — is *not* that contract. The heavy path has three chunks
390|/// and the light path forty-nine, so the heavy path reaches its drain while the
391|/// light pipeline is at full saturation. Claiming the exclusive phase there
392|/// disables occupancy-conditional routing process-wide for the light pipeline and
393|/// simultaneously force-routes this path's own fold trees behind the light
394|/// pipeline's chunk trees — it hurts both sides. The claim is legitimate only for
395|/// the path that is the last one still proving, which this counter identifies.
396|///
397|/// Routing is a scheduling heuristic: either outcome hashes the identical tree,
398|/// so a stale read here is benign and no proof byte depends on the answer.
399|fn claims_exclusive_gpu_phase(active_paths: &AtomicUsize) -> bool {
400|    active_paths.load(Ordering::Acquire) == 1
401|}
402|
403|/// Marks the calling thread as latency-critical to the macOS scheduler.
404|///
405|/// The 49 sequential chain folds are the whole critical path of a block
406|/// bundle: every serial section of a fold (witness feed, opening
407|/// evaluation, FRI reduce, transcript work) runs on a chain-step thread
408|/// while the global worker pool is saturated by transaction proving that
409|/// hides behind the spine anyway. At default QoS those serial sections
410|/// compete for cores on equal terms with hideable bulk work and are
411|/// eligible for efficiency-core placement; per-statement profiling of the
412|/// fold pipeline shows episodic multi-hundred-millisecond stalls between
413|/// instrumented spans under exactly this contention. `USER_INTERACTIVE`
414|/// asks the scheduler to keep the fold thread on a performance core and
415|/// schedule it ahead of default-QoS pool workers. This changes thread
416|/// scheduling only: no work is added, moved, or reordered, and proof
417|/// bytes are untouched. On non-macOS targets this is a no-op.
418|#[cfg(target_os = "macos")]
419|fn mark_spine_thread_latency_critical() {
420|    // `QOS_CLASS_USER_INTERACTIVE` is 0x21 in <sys/qos.h>.
421|    #[allow(non_camel_case_types)]
422|    type qos_class_t = u32;
423|    unsafe extern "C" {
424|        fn pthread_set_qos_class_self_np(qos_class: qos_class_t, relative_priority: i32) -> i32;
425|    }
426|    // Best-effort: a nonzero return leaves the thread at its previous QoS,
427|    // which is exactly the pre-change behavior.
428|    unsafe {
429|        let _ = pthread_set_qos_class_self_np(0x21, 0);
430|    }
431|}
432|
433|#[cfg(not(target_os = "macos"))]
434|fn mark_spine_thread_latency_critical() {}
435|
436|/// Marks the calling thread `QOS_CLASS_USER_INITIATED` (0x19). Used for the
437|/// transaction-proof threads: they hold the single GPU buffer set across
438|/// submit/wait/readback, so a default-QoS holder woken by the condvar can be
439|/// preempted while every other tree build queues behind it — a classic
440|/// priority inversion at the pipeline's one serialized station. Best-effort.
441|#[cfg(target_os = "macos")]
442|fn mark_thread_user_initiated() {
443|    #[allow(non_camel_case_types)]
444|    type qos_class_t = u32;
445|    unsafe extern "C" {
446|        fn pthread_set_qos_class_self_np(qos_class: qos_class_t, relative_priority: i32) -> i32;
447|    }
448|    unsafe {
449|        let _ = pthread_set_qos_class_self_np(0x19, 0);
450|    }
451|}
452|
453|#[cfg(not(target_os = "macos"))]
454|fn mark_thread_user_initiated() {}
455|
456|/// Marks the calling thread `QOS_CLASS_UTILITY` (0x11) so background page
457|/// walks prefer E-cores instead of competing with the light pipeline's
458|/// P-core work. Best-effort.
459|#[cfg(target_os = "macos")]
460|fn mark_thread_utility() {
461|    #[allow(non_camel_case_types)]
462|    type qos_class_t = u32;
463|    unsafe extern "C" {
464|        fn pthread_set_qos_class_self_np(qos_class: qos_class_t, relative_priority: i32) -> i32;
465|    }
466|    unsafe {
467|        let _ = pthread_set_qos_class_self_np(0x11, 0);
468|    }
469|}
470|
471|#[cfg(not(target_os = "macos"))]
472|fn mark_thread_utility() {}
473|
474|enum ChainState<'scope> {
475|    Ready(Proof),
476|    InFlight(std::thread::ScopedJoinHandle<'scope, Proof>),
477|}
478|
479|impl ChainState<'_> {
480|    fn wait(self) -> Proof {
481|        #[cfg(feature = "diagnostic_profile")]
482|        let _wait = plonky2::util::profile::span("wait", "chain_predecessor_join");
483|        match self {
484|            ChainState::Ready(proof) => proof,
485|            ChainState::InFlight(handle) => handle
486|                .join()
487|                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
488|        }
489|    }
490|}
491|
492|#[allow(clippy::too_many_arguments)]
493|fn chain_step_proof(
494|    path: TxPath,
495|    chain_target: &BlockTxChainTarget,
496|    chain_data: &CircuitData<F, C, D>,
497|    chain_step: u64,
498|    previous: Option<ChainState<'_>>,
499|    base_proof: &Proof,
500|    dummy_proof: &Proof,
501|    tx_proof: &Proof,
502|) -> Proof {
503|    mark_spine_thread_latency_critical();
504|    #[cfg(feature = "diagnostic_profile")]
505|    let _profile_context = plonky2::util::profile::enter_context(
506|        profile_path_context(path, "chain"),
507|        chain_step,
508|        &[("chain_step", chain_step), ("path", path as u64)],
509|    );
510|    #[cfg(feature = "diagnostic_profile")]
511|    let _profile_span = plonky2::util::profile::span("orchestration", "chain_step");
512|    let result = (|| {
513|        // Phase 1: run every generator that does not depend on the previous chain proof while
514|        // that proof may still be in flight. Inputs are written directly into
515|        // the partition's representative slots — no PartialWitness map, no
516|        // per-path template clone, no replay pass.
517|        let mut pending = PendingPartitionWitness::start_seeded(
518|            &chain_data.prover_only,
519|            &chain_data.common,
520|            |seeder| {
521|                BlockTxChainCircuit::witness_inputs_early_into(
522|                    chain_target,
523|                    chain_data,
524|                    chain_step,
525|                    dummy_proof,
526|                    tx_proof,
527|                    seeder,
528|                )
529|            },
530|        )?;
531|
532|        // Phase 2: wait for the previous chain proof, feed it directly, and prove.
533|        let previous_proof = previous.map(ChainState::wait);
534|        pending.feed_seeded(|feeder| {
535|            BlockTxChainCircuit::witness_inputs_cyclic_into(
536|                chain_target,
537|                previous_proof.as_ref().unwrap_or(base_proof),
538|                feeder,
539|            )
540|        })?;
541|        BlockTxChainCircuit::prove_prepared(pending, chain_data)
542|    })();
543|    // This step is no longer part of the runnable backlog (see the matching
544|    // spine_backlog_add(1) at all spawn sites).
545|    plonky2::hash::poseidon2::spine_backlog_add(-1);
546|    result.unwrap_or_else(|error| {
547|        panic!("{path:?} block transaction chain step #{chain_step} failed: {error:?}")
548|    })
549|}
550|
551|fn hash_from_witness(witness: &impl Witness<F>, target: &HashOutTarget) -> HashOut<F> {
552|    HashOut {
553|        elements: target.elements.map(|element| witness.get_target(element)),
554|    }
555|}
556|
557|fn jump_from_witness(witness: &impl Witness<F>, target: &JumpStateTarget) -> JumpState<F> {
558|    JumpState {
559|        last_active_tx_index: witness.get_target(target.last_active_tx_index),
560|        prev_new_state_root: hash_from_witness(witness, &target.prev_new_state_root),
561|        prev_new_delta_root: hash_from_witness(witness, &target.prev_new_delta_root),
562|        run_start_prev_index: witness.get_target(target.run_start_prev_index),
563|        run_start_old_state_root: hash_from_witness(witness, &target.run_start_old_state_root),
564|        run_start_old_delta_root: hash_from_witness(witness, &target.run_start_old_delta_root),
565|        coverage_hash: hash_from_witness(witness, &target.coverage_hash),
566|        claims_hash: hash_from_witness(witness, &target.claims_hash),
567|        tx_count: witness.get_target(target.tx_count),
568|    }
569|}
570|
571|#[allow(clippy::too_many_arguments)]
572|fn generate_tx_witness<'a>(
573|    path: TxPath,
574|    chunk_index: usize,
575|    txs: Vec<Arc<Tx<F>>>,
576|    tx_data: &'a CircuitData<F, C, D>,
577|    tx_target: &BlockTxTarget,
578|    created_at: i64,
579|    state_metadata_hash: HashOut<F>,
580|    old_jump: JumpState<F>,
581|) -> (PartitionWitness<'a, F>, JumpState<F>) {
582|    #[cfg(feature = "diagnostic_profile")]
583|    let _profile_context = plonky2::util::profile::enter_context(
584|        profile_path_context(path, "witness"),
585|        chunk_index as u64,
586|        &[("chunk_index", chunk_index as u64), ("path", path as u64)],
587|    );
588|    #[cfg(feature = "diagnostic_profile")]
589|    let _profile_span = plonky2::util::profile::span("orchestration", "generate_tx_witness");
590|    let block_tx = BlockTx {
591|        created_at,
592|        state_metadata_hash,
593|        old_jump,
594|        txs,
595|    };
596|    // Write witness values directly into the partition's representative
597|    // slots (array-indexed), bypassing the PartialWitness hash map and its
598|    // per-target hashing for the ~10^5 inputs of every transaction chunk,
599|    // while maintaining the same unresolved-watch counters.
600|    let partition_witness = PendingPartitionWitness::start_seeded(
601|        &tx_data.prover_only,
602|        &tx_data.common,
603|        |seeder| BlockTxCircuit::generate_witness_into(&block_tx, tx_target, seeder),
604|    )
605|    .and_then(PendingPartitionWitness::finish)
606|    .unwrap_or_else(|error| {
607|        panic!("{path:?} block transaction chunk #{chunk_index} witness generation failed: {error:?}")
608|    });
609|    let new_jump = jump_from_witness(&partition_witness, &tx_target.new_jump);
610|    (partition_witness, new_jump)
611|}
612|
613|fn prove_tx_witness(
614|    path: TxPath,
615|    chunk_index: usize,
616|    tx_data: &CircuitData<F, C, D>,
617|    partition_witness: PartitionWitness<'_, F>,
618|) -> Proof {
619|    #[cfg(feature = "diagnostic_profile")]
620|    let _profile_context = plonky2::util::profile::enter_context(
621|        profile_path_context(path, "proof"),
622|        chunk_index as u64,
623|        &[("chunk_index", chunk_index as u64), ("path", path as u64)],
624|    );
625|    #[cfg(feature = "diagnostic_profile")]
626|    let _profile_span = plonky2::util::profile::span("orchestration", "prove_tx_witness");
627|    let proof = prove_with_partition_witness::<F, C, D>(
628|        &tx_data.prover_only,
629|        &tx_data.common,
630|        partition_witness,
631|        &mut TimingTree::default(),
632|    )
633|    .unwrap_or_else(|error| {
634|        panic!("{path:?} block transaction chunk #{chunk_index} proof failed: {error:?}")
635|    });
636|    #[cfg(debug_assertions)]
637|    tx_data
638|        .verify(proof.clone())
639|        .expect("transaction proof self-check failed");
640|    proof
641|}
642|
643|#[allow(clippy::too_many_arguments)]
644|fn prove_path(
645|    path: TxPath,
646|    chunks: Vec<(usize, Vec<Arc<Tx<F>>>)>,
647|    circuits: &Circuits,
648|    block_number: u64,
649|    created_at: i64,
650|    old_account_delta_tree_root: HashOut<F>,
651|    pre_output: &BlockPreExecWitness<F>,
652|    state_metadata_hash: HashOut<F>,
653|    active_paths: &AtomicUsize,
654|) -> Proof {
655|    assert!(
656|        !chunks.is_empty(),
657|        "{path:?} transaction path must contain at least one chunk"
658|    );
659|    #[cfg(feature = "diagnostic_profile")]
660|    let _profile_context = plonky2::util::profile::enter_context(
661|        match path {
662|            TxPath::Heavy => "heavy_path",
663|            TxPath::Light => "light_path",
664|        },
665|        0,
666|        &[("chunks", chunks.len() as u64), ("path", path as u64)],
667|    );
668|    #[cfg(feature = "diagnostic_profile")]
669|    let _profile_span = plonky2::util::profile::span("orchestration", "prove_path");
670|    // The heavy pair's shared guards are held for exactly as long as this path
671|    // may read them — from here to the `return`, which is after its chain proof
672|    // exists — so the exclusive acquisition in
673|    // `Circuits::release_heavy_circuit_extensions` is a proof that the heavy
674|    // path is finished with them. Shared guards never block one another, so
675|    // this neither serializes the two paths nor delays the concurrent block
676|    // circuit construction, which takes its own shared guard.
677|    let heavy_tx_guard;
678|    let heavy_chain_guard;
679|    let light_tx_guard;
680|    let light_chain_guard;
681|    let (tx_data, tx_target, chain_data, chain_target, dummy_proof) = match path {
682|        TxPath::Light => {
683|            light_tx_guard = circuits
684|                .light_tx_data
685|                .read()
686|                .unwrap_or_else(|poisoned| poisoned.into_inner());
687|            light_chain_guard = circuits
688|                .light_chain_data
689|                .read()
690|                .unwrap_or_else(|poisoned| poisoned.into_inner());
691|            (
692|                &*light_tx_guard,
693|                &circuits.light_tx_target,
694|                &*light_chain_guard,
695|                &circuits.light_chain_target,
696|                &circuits.dummy_light_proof,
697|            )
698|        }
699|        TxPath::Heavy => {
700|            heavy_tx_guard = circuits
701|                .heavy_tx_data
702|                .read()
703|                .unwrap_or_else(|poisoned| poisoned.into_inner());
704|            heavy_chain_guard = circuits
705|                .heavy_chain_data
706|                .read()
707|                .unwrap_or_else(|poisoned| poisoned.into_inner());
708|            (
709|                &*heavy_tx_guard,
710|                &circuits.heavy_tx_target,
711|                &*heavy_chain_guard,
712|                &circuits.heavy_chain_target,
713|                &circuits.dummy_heavy_proof,
714|            )
715|        }
716|    };
717|
718|    let base_proof = cyclic_base_witness(
719|        dummy_proof,
720|        block_number,
721|        created_at,
722|        pre_output.new_state_root,
723|        pre_output.new_validium_root,
724|        old_account_delta_tree_root,
725|    );
726|    let mut jump = JumpState::initial(pre_output.new_state_root, old_account_delta_tree_root);
727|    let mut chunks = chunks.into_iter();
728|    let (mut current_chunk_index, first_txs) =
729|        chunks.next().expect("transaction path must not be empty");
730|    let (mut current_witness, next_jump) = generate_tx_witness(
731|        path,
732|        current_chunk_index,
733|        first_txs,
734|        tx_data,
735|        tx_target,
736|        created_at,
737|        state_metadata_hash,
738|        jump,
739|    );
740|    jump = next_jump;
741|
742|
743|    let chain_proof = std::thread::scope(|scope| {
744|        let base = &base_proof;
745|        let mut chain: Option<ChainState<'_>> = None;
746|        let mut pending_tx: Option<(u64, Proof)> = None;
747|        let mut in_flight = std::collections::VecDeque::new();
748|        let mut current_step = 0u64;
749|
750|        loop {
751|            if let Some((chain_step, tx_proof)) = pending_tx.take() {
752|                // The predecessor handle moves into the chain thread, which waits for it only
753|                // after its tx-proof-side witness generation: the path thread never blocks here.
754|                let previous = chain.take();
755|                // This step is now runnable (its tx proof exists); while the
756|                // count of runnable-but-unproven steps is high, the chain is
757|                // the laggard and its GPU trees take priority (see
758|                // spine_backlog_add). Decremented inside chain_step_proof.
759|                plonky2::hash::poseidon2::spine_backlog_add(1);
760|                let handle = std::thread::Builder::new()
761|                    .name(format!("{path:?}-chain-step-{chain_step}"))
762|                    .stack_size(PROVER_THREAD_STACK_BYTES)
763|                    .spawn_scoped(scope, move || {
764|                        chain_step_proof(
765|                            path,
766|                            chain_target,
767|                            chain_data,
768|                            chain_step,
769|                            previous,
770|                            base,
771|                            dummy_proof,
772|                            &tx_proof,
773|                        )
774|                    })
775|                    .expect("chain step pipeline thread must start");
776|                chain = Some(ChainState::InFlight(handle));
777|            }
778|
779|            let witness = current_witness;
780|            let proof_handle = std::thread::Builder::new()
781|                .name(format!("{path:?}-tx-proof-{current_step}"))
782|                .stack_size(PROVER_THREAD_STACK_BYTES)
783|                .spawn_scoped(scope, move || {
784|                    // These threads hold the single GPU buffer set across
785|                    // submit/wait/readback; see mark_thread_user_initiated.
786|                    mark_thread_user_initiated();
787|                    prove_tx_witness(path, current_chunk_index, tx_data, witness)
788|                })
789|                .expect("transaction proof pipeline thread must start");
790|
791|            let next_witness = chunks.next().map(|(chunk_index, txs)| {
792|                let (witness, next_jump) = generate_tx_witness(
793|                    path,
794|                    chunk_index,
795|                    txs,
796|                    tx_data,
797|                    tx_target,
798|                    created_at,
799|                    state_metadata_hash,
800|                    jump,
801|                );
802|                jump = next_jump;
803|                (chunk_index, witness)
804|            });
805|
806|            in_flight.push_back((current_step, proof_handle));
807|            #[cfg(feature = "diagnostic_profile")]
808|            plonky2::util::profile::counter(
809|                "scheduler",
810|                "tx_in_flight",
811|                in_flight.len() as u64,
812|            );
813|            let max_in_flight =
814|                if path == TxPath::Light && current_step >= LIGHT_TX_PROOF_OVERLAP_START_STEP {
815|                    light_tx_proof_window()
816|                } else if path == TxPath::Light {
817|                    // Ramp: while the heavy path's three chunks run, the old
818|                    // depth-1 throttle left the GPU 38% idle and the buffer
819|                    // set held only 50% (measured). Depth 2 fills that idle
820|                    // without exceeding the single set's capacity; the full
821|                    // window still waits for the heavy path's step-3 horizon.
822|                    2
823|                } else {
824|                    1
825|                };
826|            if in_flight.len() >= max_in_flight {
827|                let (proof_step, proof_handle) = in_flight
828|                    .pop_front()
829|                    .expect("transaction proof window must not be empty");
830|                #[cfg(feature = "diagnostic_profile")]
831|                let _join_wait = plonky2::util::profile::span("wait", "tx_proof_window_join");
832|                let tx_proof = proof_handle
833|                    .join()
834|                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
835|                pending_tx = Some((proof_step, tx_proof));
836|            }
837|            current_step += 1;
838|
839|            match next_witness {
840|                Some((chunk_index, witness)) => {
841|                    current_chunk_index = chunk_index;
842|                    current_witness = witness;
843|                }
844|                None => break,
845|            }
846|        }
847|
848|        if let Some((chain_step, tx_proof)) = pending_tx.take() {
849|            // This post-loop step is runnable and decrements the global
850|            // backlog in `chain_step_proof`, exactly like both spawn loops.
851|            plonky2::hash::poseidon2::spine_backlog_add(1);
852|            let previous = chain.take();
853|            let handle = std::thread::Builder::new()
854|                .name(format!("{path:?}-chain-step-{chain_step}"))
855|                .stack_size(PROVER_THREAD_STACK_BYTES)
856|                .spawn_scoped(scope, move || {
857|                    chain_step_proof(
858|                        path,
859|                        chain_target,
860|                        chain_data,
861|                        chain_step,
862|                        previous,
863|                        base,
864|                        dummy_proof,
865|                        &tx_proof,
866|                    )
867|                })
868|                .expect("chain step pipeline thread must start");
869|            chain = Some(ChainState::InFlight(handle));
870|        }
871|        // Past this point the pipeline spawns no new chunk work: the drain
872|        // below is the strictly sequential chain tail, so its mid-size
873|        // commitment trees can use the mostly idle GPU exactly like the
874|        // pre-execution and final block phases — but only once this path is the
875|        // last one proving, since the switch is process-global (see
876|        // [`claims_exclusive_gpu_phase`]).
877|        let mut exclusive_drain = false;
878|        #[cfg(feature = "diagnostic_profile")]
879|        plonky2::util::profile::counter(
880|            "scheduler",
881|            "drain_tx_in_flight",
882|            in_flight.len() as u64,
883|        );
884|        while let Some((chain_step, proof_handle)) = in_flight.pop_front() {
885|            let tx_proof = {
886|                #[cfg(feature = "diagnostic_profile")]
887|                let _join_wait = plonky2::util::profile::span("wait", "tx_proof_drain_join");
888|                proof_handle
889|                    .join()
890|                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
891|            };
892|            // Claim the exclusive phase only once the window has flushed AND
893|            // the last chunk proof has retired (the join above): before this
894|            // point the exclusive routing's lower GPU cutoff and occupancy
895|            // bypass would queue still-running chunk proofs' trees against
896|            // the drain's. Covers the sibling retiring mid-drain (relaxed).
897|            if !exclusive_drain
898|                && in_flight.is_empty()
899|                && claims_exclusive_gpu_phase(active_paths)
900|            {
901|                exclusive_drain = true;
902|                plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
903|            }
904|            // Spawn the drained step exactly like the pipelined phase so its
905|            // phase-1 witness (which needs only the tx proof) overlaps the
906|            // predecessor's prove instead of serializing behind it. Only one
907|            // proof's GPU work is in flight at a time either way — phase 1 is
908|            // pure CPU generator execution — so the exclusive-drain contract
909|            // above is unchanged.
910|            let previous = chain.take();
911|            plonky2::hash::poseidon2::spine_backlog_add(1);
912|            let handle = std::thread::Builder::new()
913|                .name(format!("{path:?}-chain-drain-{chain_step}"))
914|                .stack_size(PROVER_THREAD_STACK_BYTES)
915|                .spawn_scoped(scope, move || {
916|                    chain_step_proof(
917|                        path,
918|                        chain_target,
919|                        chain_data,
920|                        chain_step,
921|                        previous,
922|                        base,
923|                        dummy_proof,
924|                        &tx_proof,
925|                    )
926|                })
927|                .expect("chain drain thread must start");
928|            chain = Some(ChainState::InFlight(handle));
929|        }
930|        let chain_proof = chain
931|            .map(ChainState::wait)
932|            .expect("transaction path must produce a chain proof");
933|        if exclusive_drain {
934|            plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
935|        }
936|        chain_proof
937|    });
938|    // This path has produced its last proof. Retiring it here — after the scope,
939|    // so every thread it spawned has joined — is what lets the sibling path's
940|    // drain observe that it is alone and claim the exclusive GPU phase.
941|    active_paths.fetch_sub(1, Ordering::Release);
942|    if path == TxPath::Heavy {
943|        // The heavy path retires far ahead of the light path; use the slack
944|        // to pre-fault the final block's wires column store (larger than the
945|        // Metal pool cap, so otherwise freshly zero-faulted ~2 GiB inside the
946|        // run's most serial window). Detached: only populates a stash the
947|        // block's allocation consults; a size miss falls through unchanged.
948|        std::thread::Builder::new()
949|            .name("block-store-prewarm".to_owned())
950|            .spawn(|| {
951|                // Page-walking ~2 GiB at default QoS competes with the light
952|                // pipeline for P-cores; utility class prefers the E-cores,
953|                // whose memory-bound fault service is nearly as fast.
954|                mark_thread_utility();
955|                // Final block: 2^18 rows << 3 rate bits = 2^21 LDE rows, one
956|                // u64 per wire column. Kept in sync with CIRCUIT_CONFIG's
957|                // num_wires; a drift just misses the stash harmlessly.
958|                const BLOCK_WIRES_STORE_BYTES: u64 =
959|                    (circuit::types::config::CIRCUIT_CONFIG.num_wires as u64) * (1 << 21) * 8;
960|                plonky2::hash::poseidon2::prewarm_large_column_store(BLOCK_WIRES_STORE_BYTES);
961|            })
962|            .ok();
963|    }
964|    chain_proof
965|}
966|
967|/// Proves the block pre-execution circuit. The startup-overlap path must NOT
968|/// set the exclusive GPU phase, because the remaining circuit loads are still
969|/// using the GPU normally; the serial path sets it around the call.
970|///
971|/// Measured, so nobody re-mines this (10 interleaved runs, one binary, the
972|/// switch runtime-gated, census taken in `gpu_worthwhile`):
973|/// `Circuits::load_remaining_embedded` recomputes each blob's
974|/// `constants_sigmas_commitment` through `PolynomialBatch::from_values`, so
975|/// four commitment trees — 2 x (2^19 leaves x 88 cols) for the transaction
976|/// circuits and 2 x (2^17 x 86) for the chain circuits, all either above the
977|/// routing cutoff or wider than 64 and therefore GPU-bound unconditionally —
978|/// are hashing on the GPU *inside* the pre-execution window (8 of the window's
979|/// routing decisions). With `MAX_BUFFER_SETS == 1` those builds serialize, and
980|/// this proof's own narrow trees observe `GPU_JOBS_IN_FLIGHT` at 1-2 for 5-7 of
981|/// their 9 routing decisions. So "no other proof runs concurrently" holds here
982|/// while "the GPU stream is idle" does not, and only the latter is the switch's
983|/// real contract. Enabling it does change routing — the 2^17 width-20
984|/// Zs/partial-products tree goes 1/10 -> 10/10 GPU and the width-16 quotient
985|/// tree 6/10 -> 10/10 — but each flipped tree then queues FIFO behind a
986|/// 2^19-leaf load build, and the phase inflated from a 325 ms median to 425 ms.
987|/// It buys nothing even when it wins: this proof finishes a median 187 ms before
988|/// the loads it hides behind, so the join waits on the loads, not on it.
989|/// Enabling the switch only spends that slack (median 187 ms -> 126 ms) and put
990|/// the proof on the critical path in 1 of the 5 runs that had it enabled.
991|pub(crate) fn prove_pre_execution_parallel(
992|    pre_data: &CircuitData<F, C, D>,
993|    pre_target: &BlockPreExecutionTarget,
994|    pre_exec: &BlockPreExec<F>,
995|) -> Proof {
996|    #[cfg(feature = "diagnostic_profile")]
997|    let _profile_context = plonky2::util::profile::enter_context(
998|        "pre_execution",
999|        0,
1000|        &[("proof_kind", 0)],
1001|    );
1002|    #[cfg(feature = "diagnostic_profile")]
1003|    let _profile_span = plonky2::util::profile::span("orchestration", "pre_execution_proof");
1004|    BlockPreExecutionCircuit::prove(pre_data, pre_exec, pre_target)
1005|        .expect("block pre-execution proof failed")
1006|}
1007|
1008|/// The fully serial entry point: pre-execution proof first, then the pipeline.
1009|///
1010|/// Test-only. The `prove` binary starts the pre-execution proof on a startup
1011|/// thread that overlaps the remaining circuit loads and then calls
1012|/// [`prove_block_after_pre`] directly, so nothing on the scored path routes
1013|/// through here. It is retained as the reference for what the serial ordering
1014|/// looked like — in particular that the exclusive-GPU switch below is legitimate
1015|/// only under that ordering, which no longer exists (see
1016|/// [`prove_pre_execution_parallel`]). `#[cfg(test)]` because the release build
1017|/// would otherwise warn it dead.
1018|#[cfg(test)]
1019|pub fn prove_block(block: Block<F>, circuits: Circuits) -> Proof {
1020|    // The pre-execution proof runs strictly before any other proving work, so
1021|    // the serialized GPU stream is otherwise idle: route its mid-size column
1022|    // trees to the GPU for just this phase.
1023|    plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
1024|    let pre_proof = prove_pre_execution_parallel(
1025|        &circuits.pre_data,
1026|        &circuits.pre_target,
1027|        &BlockPreExec::from_block(&block),
1028|    );
1029|    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
1030|    prove_block_after_pre(block, circuits, pre_proof)
1031|}
1032|
1033|/// The pipeline after the pre-execution proof. The startup-overlap path calls
1034|/// this once both the pre-execution proof and the remaining circuit loads have
1035|/// completed.
1036|pub(crate) fn prove_block_after_pre(
1037|    mut block: Block<F>,
1038|    mut circuits: Circuits,
1039|    pre_proof: Proof,
1040|) -> Proof {
1041|    #[cfg(feature = "diagnostic_profile")]
1042|    let _profile_context =
1043|        plonky2::util::profile::enter_context("block_pipeline", block.block_number, &[]);
1044|    #[cfg(feature = "diagnostic_profile")]
1045|    let _profile_span = plonky2::util::profile::span("orchestration", "prove_block_after_pre");
1046|    let pre_output = BlockPreExecWitness::from_public_inputs(&pre_proof.public_inputs);
1047|    let state_metadata_hash = pre_output.new_state_metadata.hash();
1048|
1049|    let mut tx_chunks = std::mem::take(&mut block.tx_chunks);
1050|    let mut heavy_chunks: Vec<(usize, Vec<Arc<Tx<F>>>)> = Vec::new();
1051|    let mut light_chunks: Vec<(usize, Vec<Arc<Tx<F>>>)> =
1052|        Vec::with_capacity(tx_chunks.len());
1053|    for (chunk_index, txs) in tx_chunks.drain(..).enumerate() {
1054|        if chunk_is_light(&txs) {
1055|            light_chunks.push((chunk_index, txs));
1056|        } else {
1057|            heavy_chunks.push((chunk_index, txs));
1058|        }
1059|    }
1060|    block.tx_chunks = tx_chunks;
1061|    block.tx_chunks.push(Vec::new());
1062|
1063|    // Both transaction paths prove concurrently and each ends in a strictly
1064|    // sequential chain tail, but the exclusive-GPU switch that tail wants is
1065|    // process-global. This counter lets a path tell "my own pipeline is done"
1066|    // apart from "no other proof is running": each path retires itself when its
1067|    // chain proof is finished, so only the last one standing claims the phase.
1068|    let active_paths = AtomicUsize::new(2);
1069|    let (light_chain_proof, heavy_chain_proof, block_target, block_data, block_pending) = {
1070|        // The pipeline only ever reads the circuits; the borrow ends with this
1071|        // block so the finished extensions can be released below.
1072|        let circuits = &circuits;
1073|        let active_paths = &active_paths;
1074|        std::thread::scope(|scope| {
1075|            // The final block circuit depends only on already-built circuit data
1076|            // and is not needed until the final proof, so it builds concurrently
1077|            // with the entire transaction/chain proving pipeline.
1078|            // Two-phase final-block witness (H13): this lane also runs the
1079|            // EARLY witness phase (block data + pre-proof generators) after the
1080|            // build, then joins the heavy path — which finishes ~30 s before
1081|            // the light path — and feeds its verify subtree here, mid-pipeline.
1082|            // Measured feed split: light 0.018 s vs heavy 0.575 s; the heavy
1083|            // verify subtree (ECDSA/keccak) owns the late witness cost, and
1084|            // moving it here deletes it from the serial tail. Both phases run
1085|            // WITHOUT `ParallelWitnessGuard` (thread-local; parallel rounds
1086|            // here would contend with the pipeline's pool). This is witness
1087|            // WORK MOVED OFF THE TAIL onto an otherwise-idle lane, not new
1088|            // parallelism: the lane sleeps in `join` until the heavy proof
1089|            // arrives, then does 0.6 s of serial work while ~the light spine
1090|            // alone is running. The circuit data is leaked to hand the pending
1091|            // witness a 'static borrow across the thread boundary — free, the
1092|            // worker exits via `process::exit`.
1093|            let heavy_handle_outer = std::thread::Builder::new()
1094|                .name("heavy-tx-chain".into())
1095|                .stack_size(PROVER_THREAD_STACK_BYTES)
1096|                .spawn_scoped(scope, || {
1097|                    prove_path(
1098|                        TxPath::Heavy,
1099|                        heavy_chunks,
1100|                        circuits,
1101|                        block.block_number,
1102|                        block.created_at,
1103|                        block.old_account_delta_tree_root,
1104|                        &pre_output,
1105|                        state_metadata_hash,
1106|                        active_paths,
1107|                    )
1108|                })
1109|                .expect("heavy transaction chain thread must start");
1110|            let block_ref = &block;
1111|            let pre_proof_ref = &pre_proof;
1112|            let block_circuit_handle = std::thread::Builder::new()
1113|                .name("block-circuit-build".into())
1114|                .stack_size(PROVER_THREAD_STACK_BYTES)
1115|                .spawn_scoped(scope, move || {
1116|                    #[cfg(feature = "diagnostic_profile")]
1117|                    let _profile_context = plonky2::util::profile::enter_context(
1118|                        "final_block_build",
1119|                        block_ref.block_number,
1120|                        &[],
1121|                    );
1122|                    #[cfg(feature = "diagnostic_profile")]
1123|                    let _profile_span =
1124|                        plonky2::util::profile::span("orchestration", "final_block_build_lane");
1125|                    let (block_target, block_data) = {
1126|                        #[cfg(feature = "diagnostic_profile")]
1127|                        let _span =
1128|                            plonky2::util::profile::span("orchestration", "build_block_circuit");
1129|                        circuits.build_block_circuit()
1130|                    };
1131|                    let block_data: &'static CircuitData<F, C, D> =
1132|                        Box::leak(Box::new(block_data));
1133|                    let early = BlockCircuit::witness_inputs_early(
1134|                        &block_target,
1135|                        block_ref,
1136|                        pre_proof_ref,
1137|                    )
1138|                    .expect("final block early witness inputs failed");
1139|                    let mut pending = PendingPartitionWitness::start(
1140|                        early,
1141|                        &block_data.prover_only,
1142|                        &block_data.common,
1143|                    )
1144|                    .expect("final block early witness phase failed");
1145|                    #[cfg(feature = "diagnostic_profile")]
1146|                    let _heavy_wait =
1147|                        plonky2::util::profile::span("wait", "heavy_path_join_for_final");
1148|                    let heavy_chain_proof = heavy_handle_outer
1149|                        .join()
1150|                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
1151|                    // The heavy path's thread has exited, so its shared guards
1152|                    // on the heavy transaction and chain circuits are gone, and
1153|                    // this lane dropped its own guard when `build_block_circuit`
1154|                    // returned above. Nothing reads those two circuits again:
1155|                    // the light pipeline uses the light pair, and the final
1156|                    // block proof uses only `block_data`, the three finished
1157|                    // proofs and the block. Retire their preprocessed
1158|                    // extensions here — 438 MiB of Metal shared buffers whose
1159|                    // release returns the pages to the OS immediately — instead
1160|                    // of holding them across the whole light phase.
1161|                    circuits.release_heavy_circuit_extensions();
1162|                    pending
1163|                        .feed(
1164|                            BlockCircuit::witness_inputs_heavy_chain(
1165|                                &block_target,
1166|                                &heavy_chain_proof,
1167|                            )
1168|                            .expect("final block heavy-chain witness inputs failed"),
1169|                        )
1170|                        .expect("final block heavy-chain witness feed failed");
1171|                    (block_target, block_data, pending, heavy_chain_proof)
1172|                })
1173|                .expect("block circuit build thread must start");
1174|            let light_chunks = std::mem::take(&mut light_chunks);
1175|            let light_handle = std::thread::Builder::new()
1176|                .name("light-tx-chain".into())
1177|                .stack_size(PROVER_THREAD_STACK_BYTES)
1178|                .spawn_scoped(scope, || {
1179|                    mark_spine_thread_latency_critical();
1180|                    prove_path(
1181|                        TxPath::Light,
1182|                        light_chunks,
1183|                        circuits,
1184|                        block.block_number,
1185|                        block.created_at,
1186|                        block.old_account_delta_tree_root,
1187|                        &pre_output,
1188|                        state_metadata_hash,
1189|                        active_paths,
1190|                    )
1191|                })
1192|                .expect("light transaction chain thread must start");
1193|            #[cfg(feature = "diagnostic_profile")]
1194|            let _block_lane_wait =
1195|                plonky2::util::profile::span("wait", "final_block_build_lane_join");
1196|            let (block_target, block_data, block_pending, heavy_chain_proof) =
1197|                block_circuit_handle
1198|                    .join()
1199|                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
1200|            #[cfg(feature = "diagnostic_profile")]
1201|            drop(_block_lane_wait);
1202|            #[cfg(feature = "diagnostic_profile")]
1203|            let _light_wait = plonky2::util::profile::span("wait", "light_path_join");
1204|            let light_chain_proof = light_handle
1205|                .join()
1206|                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
1207|            // The light path's thread has exited, so its shared guards on the
1208|            // light transaction and chain circuits are gone, and the block lane
1209|            // dropped its own light-chain guard when `build_block_circuit`
1210|            // returned long ago. Nothing reads the light pair again: the final
1211|            // block proof uses only `block_data`, the three finished proofs and
1212|            // the block. Retire their preprocessed extensions here — 438 MiB of
1213|            // Metal shared buffers whose release returns the pages to the OS
1214|            // immediately — instead of holding them through the final witness
1215|            // setup until the backstop below.
1216|            circuits.release_light_circuit_extensions();
1217|            (
1218|                light_chain_proof,
1219|                heavy_chain_proof,
1220|                block_target,
1221|                block_data,
1222|                block_pending,
1223|            )
1224|        })
1225|    };
1226|
1227|    // Every circuit but the block circuit has now produced its last proof, so
1228|    // their preprocessed low-degree extensions are unreachable. Release them
1229|    // before the final block proof — the process's peak-RSS moment — stacks its
1230|    // own extensions on top of them.
1231|    circuits.release_finished_circuit_extensions();
1232|
1233|    #[cfg(feature = "diagnostic_profile")]
1234|    let _profile_context =
1235|        plonky2::util::profile::enter_context("final_block", block.block_number, &[]);
1236|    #[cfg(feature = "diagnostic_profile")]
1237|    let _profile_span = plonky2::util::profile::span("orchestration", "final_block_tail");
1238|    let (light_chain_input, heavy_chain_input) =
1239|        final_chain_inputs(&light_chain_proof, &heavy_chain_proof);
1240|    // The final block witness runs on the serial tail with nothing else proving, so it alone
1241|    // opts into parallel worklist rounds; tx-proof and chain witness generation run concurrently
1242|    // with proving and stay sequential.
1243|    let _parallel_block_witness = ParallelWitnessGuard::new();
1244|    // For the same reason the serialized GPU stream is otherwise idle here:
1245|    // route the final block proof's mid-size column trees to the GPU for just
1246|    // this phase.
1247|    plonky2::hash::poseidon2::set_exclusive_gpu_phase(true);
1248|    let mut block_pending = block_pending;
1249|    {
1250|        #[cfg(feature = "diagnostic_profile")]
1251|        let _span = plonky2::util::profile::span("witness", "final_light_feed");
1252|        block_pending
1253|            .feed(
1254|                BlockCircuit::witness_inputs_light_chain(&block_target, light_chain_input)
1255|                    .expect("final block light-chain witness inputs failed"),
1256|            )
1257|            .expect("final block light-chain witness feed failed");
1258|    }
1259|    let _ = heavy_chain_input;
1260|    let final_proof = {
1261|        #[cfg(feature = "diagnostic_profile")]
1262|        let _span = plonky2::util::profile::span("orchestration", "final_block_proof");
1263|        BlockCircuit::prove_prepared(block_pending, block_data)
1264|            .expect("final block proof failed")
1265|    };
1266|    plonky2::hash::poseidon2::set_exclusive_gpu_phase(false);
1267|    final_proof
1268|}
1269|
1270|#[cfg(test)]
1271|mod tests {
1272|
1273|    use super::*;
1274|    use crate::api::{
1275|        HEAVY_TX_PER_PROOF, LIGHT_TX_PER_PROOF, PUBLIC_HEAVY_TX_COUNT, PUBLIC_LIGHT_TX_COUNT,
1276|    };
1277|
1278|    #[cfg(feature = "diagnostic_profile")]
1279|    #[test]
1280|    fn profile_path_context_names_are_stable() {
1281|        assert_eq!(profile_path_context(TxPath::Heavy, "witness"), "heavy_tx_witness");
1282|        assert_eq!(profile_path_context(TxPath::Light, "proof"), "light_tx_proof");
1283|        assert_eq!(profile_path_context(TxPath::Heavy, "chain"), "heavy_chain");
1284|        assert_eq!(profile_path_context(TxPath::Light, "chain"), "light_chain");
1285|    }
1286|
1287|    #[test]
1288|    fn prove_block_returns_one_final_block_proof() {
1289|        let prove: fn(Block<F>, Circuits) -> Proof = prove_block;
1290|        let _ = prove;
1291|    }
1292|
1293|    #[test]
1294|    fn parsed_mixed_chunks_have_expected_paths() {
1295|        std::thread::Builder::new()
1296|            .stack_size(32 * 1024 * 1024)
1297|            .spawn(|| {
1298|                let block = Block::<F>::from_json_with_empty_txs(
1299|                    include_bytes!("../bench_test.json"),
1300|                    HEAVY_TX_PER_PROOF,
1301|                    LIGHT_TX_PER_PROOF,
1302|                    PUBLIC_HEAVY_TX_COUNT,
1303|                    PUBLIC_LIGHT_TX_COUNT,
1304|                )
1305|                .expect("public fixture must parse");
1306|                let paths = block
1307|                    .tx_chunks
1308|                    .iter()
1309|                    .map(|txs| chunk_is_light(txs))
1310|                    .collect::<Vec<_>>();
1311|
1312|                assert_eq!(paths.len(), block.tx_chunks.len());
1313|                assert_eq!(paths.iter().filter(|&&is_light| !is_light).count(), 3);
1314|                assert_eq!(paths.iter().filter(|&&is_light| is_light).count(), 49);
1315|            })
1316|            .expect("orchestration test thread must start")
1317|            .join()
1318|            .expect("orchestration test thread must finish");
1319|    }
1320|
1321|    #[test]
1322|    fn empty_padding_transactions_share_storage_per_path() {
1323|        use std::sync::Arc;
1324|
1325|        std::thread::Builder::new()
1326|            .stack_size(PROVER_THREAD_STACK_BYTES)
1327|            .spawn(|| {
1328|                let block = Block::<F>::from_json_with_empty_txs(
1329|                    include_bytes!("../bench_test.json"),
1330|                    HEAVY_TX_PER_PROOF,
1331|                    LIGHT_TX_PER_PROOF,
1332|                    PUBLIC_HEAVY_TX_COUNT,
1333|                    PUBLIC_LIGHT_TX_COUNT,
1334|                )
1335|                .expect("public fixture must parse");
1336|                let heavy = block
1337|                    .tx_chunks
1338|                    .iter()
1339|                    .flatten()
1340|                    .find(|tx| tx.tx_circuit_type != TX_LIGHT)
1341|                    .expect("heavy padding must exist");
1342|                let light = block
1343|                    .tx_chunks
1344|                    .iter()
1345|                    .flatten()
1346|                    .find(|tx| tx.tx_circuit_type == TX_LIGHT)
1347|                    .expect("light padding must exist");
1348|                assert!(block
1349|                    .tx_chunks
1350|                    .iter()
1351|                    .flatten()
1352|                    .filter(|tx| tx.tx_circuit_type != TX_LIGHT)
1353|                    .all(|tx| Arc::ptr_eq(tx, heavy)));
1354|                assert!(block
1355|                    .tx_chunks
1356|                    .iter()
1357|                    .flatten()
1358|                    .filter(|tx| tx.tx_circuit_type == TX_LIGHT)
1359|                    .all(|tx| Arc::ptr_eq(tx, light)));
1360|                assert!(!Arc::ptr_eq(heavy, light));
1361|            })
1362|            .expect("padding sharing test thread must start")
1363|            .join()
1364|            .expect("padding sharing test thread must finish");
1365|    }
1366|
1367|    #[test]
1368|    fn exclusive_gpu_phase_is_claimed_only_by_the_last_running_path() {
1369|        // Two paths proving: the one that reaches its drain first (the three-chunk
1370|        // heavy path) must not claim the process-global exclusive phase while the
1371|        // forty-nine-chunk light pipeline is still running.
1372|        let active_paths = AtomicUsize::new(2);
1373|        assert!(!claims_exclusive_gpu_phase(&active_paths));
1374|
1375|        // The heavy path retires; the light path's drain is now genuinely alone.
1376|        active_paths.fetch_sub(1, Ordering::Release);
1377|        assert!(claims_exclusive_gpu_phase(&active_paths));
1378|
1379|        // Both retired: nothing is proving, so nothing claims the phase either.
1380|        active_paths.fetch_sub(1, Ordering::Release);
1381|        assert!(!claims_exclusive_gpu_phase(&active_paths));
1382|    }
1383|
1384|    #[test]
1385|    fn final_block_chain_inputs_are_light_then_heavy() {
1386|        let light = "light";
1387|        let heavy = "heavy";
1388|
1389|        assert_eq!(final_chain_inputs(&light, &heavy), (&light, &heavy));
1390|    }
1391|
1392|    /// Manual timing harness for the two-phase chain-step witness split. Run with:
1393|    /// `RAYON_NUM_THREADS=8 cargo test --release -p bench --bin prove -- --ignored chain_step`
1394|    #[test]
1395|    #[ignore = "manual timing harness; run explicitly with --release"]
1396|    fn chain_step_two_phase_timing() {
1397|        std::thread::Builder::new()
1398|            .stack_size(PROVER_THREAD_STACK_BYTES)
1399|            .spawn(chain_step_two_phase_timing_impl)
1400|            .expect("timing harness thread must start")
1401|            .join()
1402|            .expect("timing harness thread must finish");
1403|    }
1404|
1405|    fn chain_step_two_phase_timing_impl() {
1406|        use std::time::Instant;
1407|
1408|        const CHAIN_STEPS: u64 = 10;
1409|
1410|        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
1411|            .is_test(false)
1412|            .try_init();
1413|
1414|        use circuit::block_tx_chain_constraints::Circuit as _;
1415|        use circuit::types::constants::TX_TYPE_EMPTY;
1416|        use plonky2::field::types::{Field, PrimeField64};
1417|
1418|        use crate::api::{LIGHT_TX_MODE, PathCircuits};
1419|
1420|        let build_start = Instant::now();
1421|        let circuits = PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE);
1422|        println!("light path circuits built in {:?}", build_start.elapsed());
1423|
1424|        let block = Block::<F>::from_json_with_empty_txs(
1425|            include_bytes!("../bench_test.json"),
1426|            HEAVY_TX_PER_PROOF,
1427|            LIGHT_TX_PER_PROOF,
1428|            PUBLIC_HEAVY_TX_COUNT,
1429|            PUBLIC_LIGHT_TX_COUNT,
1430|        )
1431|        .expect("public fixture must parse");
1432|
1433|        // An all-empty (padding) chunk carries no state transition, so its embedded roots and
1434|        // metadata hash are the only values the tx and chain constraints must agree on.
1435|        // Chain-step cost is independent of tx contents: the chain circuit is fixed-size.
1436|        let mut empty_tx = (**block
1437|            .tx_chunks
1438|            .iter()
1439|            .flatten()
1440|            .find(|tx| tx.tx_type == TX_TYPE_EMPTY)
1441|            .expect("fixture must contain an empty padding tx"))
1442|            .clone();
1443|        empty_tx.tx_circuit_type = TX_LIGHT;
1444|        empty_tx.tx_index = F::NEG_ONE.to_canonical_u64();
1445|
1446|        let new_state_root = empty_tx.old_state_root;
1447|        let old_delta_root = empty_tx.old_account_delta_tree_root;
1448|        // The post-pre-execution metadata replayed natively: pre-execution only refreshes the
1449|        // timestamps of the enabled recalculations.
1450|        let mut new_state_metadata = block.state_metadata.clone();
1451|        if block.calculate_funding {
1452|            new_state_metadata.last_funding_round_timestamp = block.created_at;
1453|        }
1454|        if block.calculate_oracle_prices {
1455|            new_state_metadata.last_oracle_price_timestamp = block.created_at;
1456|        }
1457|        if block.calculate_premium {
1458|            new_state_metadata.last_premium_timestamp = block.created_at;
1459|        }
1460|        let state_metadata_hash = new_state_metadata.hash();
1461|        let jump = JumpState::initial(new_state_root, old_delta_root);
1462|
1463|        let light_chunk = vec![Arc::new(empty_tx); LIGHT_TX_PER_PROOF];
1464|        let (witness, _) = generate_tx_witness(
1465|            TxPath::Light,
1466|            0,
1467|            light_chunk,
1468|            &circuits.tx_data,
1469|            &circuits.tx_target,
1470|            block.created_at,
1471|            state_metadata_hash,
1472|            jump,
1473|        );
1474|        let tx_prove_start = Instant::now();
1475|        let mut tx_timing = TimingTree::new("tx-chunk-prove", log::Level::Debug);
1476|        let tx_proof = plonky2::plonk::prover::prove_with_partition_witness::<F, C, D>(
1477|            &circuits.tx_data.prover_only,
1478|            &circuits.tx_data.common,
1479|            witness,
1480|            &mut tx_timing,
1481|        )
1482|        .expect("tx proof failed");
1483|        println!("tx chunk prove total {:?}", tx_prove_start.elapsed());
1484|        tx_timing.print();
1485|
1486|        let base_proof = cyclic_base_witness(
1487|            &circuits.dummy_proof,
1488|            block.block_number,
1489|            block.created_at,
1490|            new_state_root,
1491|            new_state_root,
1492|            old_delta_root,
1493|        );
1494|
1495|        let mut previous: Option<Proof> = None;
1496|        for chain_step in 0..CHAIN_STEPS {
1497|            let cyclic_proof = previous.as_ref().unwrap_or(&base_proof);
1498|
1499|            let single_shot_start = Instant::now();
1500|            let inputs = BlockTxChainCircuit::generate_witness(
1501|                &circuits.chain_target,
1502|                &circuits.chain_data,
1503|                chain_step,
1504|                cyclic_proof,
1505|                &circuits.dummy_proof,
1506|                &tx_proof,
1507|            )
1508|            .expect("single-shot witness inputs failed");
1509|            let single_shot = generate_partial_witness::<F, C, D>(
1510|                inputs,
1511|                &circuits.chain_data.prover_only,
1512|                &circuits.chain_data.common,
1513|            )
1514|            .expect("single-shot witness generation failed");
1515|            let single_shot_elapsed = single_shot_start.elapsed();
1516|            drop(single_shot);
1517|
1518|            let phase1_start = Instant::now();
1519|            let early_inputs = BlockTxChainCircuit::witness_inputs_early(
1520|                &circuits.chain_target,
1521|                &circuits.chain_data,
1522|                chain_step,
1523|                &circuits.dummy_proof,
1524|                &tx_proof,
1525|            )
1526|            .expect("early witness inputs failed");
1527|            let mut pending = PendingPartitionWitness::start(
1528|                early_inputs,
1529|                &circuits.chain_data.prover_only,
1530|                &circuits.chain_data.common,
1531|            )
1532|            .expect("early witness generation failed");
1533|            let phase1_elapsed = phase1_start.elapsed();
1534|
1535|            let phase2_start = Instant::now();
1536|            pending
1537|                .feed(
1538|                    BlockTxChainCircuit::witness_inputs_cyclic(
1539|                        &circuits.chain_target,
1540|                        cyclic_proof,
1541|                    )
1542|                    .expect("cyclic witness inputs failed"),
1543|                )
1544|                .expect("cyclic witness generation failed");
1545|            let witness = pending
1546|                .finish()
1547|                .expect("chain step witness must be complete");
1548|            let phase2_elapsed = phase2_start.elapsed();
1549|
1550|            let prove_start = Instant::now();
1551|            let mut timing = TimingTree::new("chain-step-prove", log::Level::Debug);
1552|            let proof = prove_with_partition_witness::<F, C, D>(
1553|                &circuits.chain_data.prover_only,
1554|                &circuits.chain_data.common,
1555|                witness,
1556|                &mut timing,
1557|            )
1558|            .expect("chain step proof failed");
1559|            let prove_elapsed = prove_start.elapsed();
1560|            timing.print();
1561|
1562|            // Differential integration check for the production direct-seeding
1563|            // path. The reference above keeps the old PartialWitness map
1564|            // path solely for this manual timing harness.
1565|            let direct_start = Instant::now();
1566|            let direct_proof = chain_step_proof(
1567|                TxPath::Light,
1568|                &circuits.chain_target,
1569|                &circuits.chain_data,
1570|                chain_step,
1571|                previous.clone().map(ChainState::Ready),
1572|                &base_proof,
1573|                &circuits.dummy_proof,
1574|                &tx_proof,
1575|            );
1576|            let direct_elapsed = direct_start.elapsed();
1577|            assert_eq!(proof.public_inputs, direct_proof.public_inputs);
1578|
1579|            println!(
1580|                "chain step {chain_step}: single-shot witness {single_shot_elapsed:?}, \
1581|                 map phase1 {phase1_elapsed:?}, map phase2 {phase2_elapsed:?}, \
1582|                 map prove {prove_elapsed:?}, direct total {direct_elapsed:?}",
1583|            );
1584|            previous = Some(direct_proof);
1585|        }
1586|
1587|        circuits
1588|            .chain_data
1589|            .verify(previous.expect("chain must produce proofs"))
1590|            .expect("final chain step proof must verify");
1591|    }
1592|}
1593|
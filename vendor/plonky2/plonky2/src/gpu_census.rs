//! GPU occupancy, measured from the command buffers themselves.
//!
//! Every gate family that moves onto the Metal quotient union buys CPU time
//! with GPU time. That trade only pays while the GPU has slack, so the useful
//! number is not how much work it does but how much of the wall clock it is
//! busy. Metal reports each command buffer's actual execution window, so the
//! union of those windows is occupancy without any wall-clock guessing.
//!
//! Off unless `PLONKY2_GPU_CENSUS=1`.

use std::fmt::{self, Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::thread::ThreadId;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
use objc::{msg_send, sel, sel_impl};

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkLabel {
    path: &'static str,
    leaf_count: Option<usize>,
    leaf_width: Option<usize>,
    cap_height: Option<usize>,
    part: Option<(usize, usize)>,
}

#[cfg(any(
    test,
    all(feature = "std", target_arch = "aarch64", target_os = "macos")
))]
impl WorkLabel {
    const fn plain(path: &'static str) -> Self {
        Self {
            path,
            leaf_count: None,
            leaf_width: None,
            cap_height: None,
            part: None,
        }
    }

    const fn tree(
        path: &'static str,
        leaf_count: usize,
        leaf_width: usize,
        cap_height: usize,
    ) -> Self {
        Self {
            path,
            leaf_count: Some(leaf_count),
            leaf_width: Some(leaf_width),
            cap_height: Some(cap_height),
            part: None,
        }
    }

    const fn tree_part(
        path: &'static str,
        leaf_count: usize,
        leaf_width: usize,
        cap_height: usize,
        part: usize,
        parts: usize,
    ) -> Self {
        Self {
            path,
            leaf_count: Some(leaf_count),
            leaf_width: Some(leaf_width),
            cap_height: Some(cap_height),
            part: Some((part, parts)),
        }
    }
}

impl Display for WorkLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.path)?;
        let Some(leaf_count) = self.leaf_count else {
            return Ok(());
        };

        if leaf_count.is_power_of_two() {
            write!(f, "[lg={}", leaf_count.ilog2())?;
        } else {
            write!(f, "[n={leaf_count}")?;
        }
        if let Some(leaf_width) = self.leaf_width {
            write!(f, ",w={leaf_width}")?;
        }
        if let Some(cap_height) = self.cap_height {
            write!(f, ",cap={cap_height}")?;
        }
        if let Some((part, parts)) = self.part {
            write!(f, ",part={}/{parts}", part + 1)?;
        }
        f.write_str("]")
    }
}

#[derive(Clone, Copy)]
struct GpuSpan {
    start: f64,
    end: f64,
    work: WorkLabel,
    wait_thread: ThreadId,
}

#[derive(Clone, Copy)]
struct GpuGap {
    duration: f64,
    at: f64,
    before: WorkLabel,
    after: WorkLabel,
    before_sequence: usize,
    after_sequence: usize,
    before_wait_thread: ThreadId,
    after_wait_thread: ThreadId,
}

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("PLONKY2_GPU_CENSUS").as_deref() == Ok("1"));
static SPANS: LazyLock<Mutex<Vec<GpuSpan>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static REPORTED: AtomicBool = AtomicBool::new(false);

/// Records one completed command buffer. Safe to call on a buffer that never
/// ran: Metal reports a zero window, which is dropped.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub fn record(command_buffer: &metal::CommandBufferRef, path: &'static str) {
    record_work(command_buffer, WorkLabel::plain(path));
}

/// Records a completed Merkle/commitment command buffer with its exact static
/// path and numeric shape. The label itself never allocates on the hot path.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub fn record_tree(
    command_buffer: &metal::CommandBufferRef,
    path: &'static str,
    leaf_count: usize,
    leaf_width: usize,
    cap_height: usize,
) {
    record_work(
        command_buffer,
        WorkLabel::tree(path, leaf_count, leaf_width, cap_height),
    );
}

/// As [`record_tree`], plus this command buffer's zero-based position within
/// a streamed multi-buffer build.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub fn record_tree_part(
    command_buffer: &metal::CommandBufferRef,
    path: &'static str,
    leaf_count: usize,
    leaf_width: usize,
    cap_height: usize,
    part: usize,
    parts: usize,
) {
    record_work(
        command_buffer,
        WorkLabel::tree_part(path, leaf_count, leaf_width, cap_height, part, parts),
    );
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn record_work(command_buffer: &metal::CommandBufferRef, work: WorkLabel) {
    if !*ENABLED {
        return;
    }
    // `metal` 0.33 does not wrap these, but the selectors exist on
    // MTLCommandBuffer and return CFTimeInterval (a double).
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command_buffer, GPUStartTime],
            msg_send![command_buffer, GPUEndTime],
        )
    };
    if end > start && start > 0.0 {
        SPANS.lock().expect("gpu census poisoned").push(GpuSpan {
            start,
            end,
            work,
            // This is deliberately the wait/record thread, not an inferred
            // Metal submission or execution thread.
            wait_thread: std::thread::current().id(),
        });
    }
}

pub fn report() {
    if !*ENABLED || REPORTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let mut spans = SPANS.lock().expect("gpu census poisoned").clone();
    if spans.is_empty() {
        eprintln!("[gpu-census] no command buffers recorded");
        return;
    }
    spans.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .expect("gpu timestamps are finite")
    });

    let first = spans[0].start;
    let last = spans.iter().map(|span| span.end).fold(f64::MIN, f64::max);
    let submitted: f64 = spans.iter().map(|span| span.end - span.start).sum();

    // Union, so overlapping buffers are not double counted.
    let (mut busy, mut cur_start, mut cur_end) = (0.0, spans[0].start, spans[0].end);
    // Whoever closed the busy run is who the GPU was waiting on. Sequence
    // numbers are assigned only after sorting by the GPU's own start time.
    let mut closing = spans[0];
    let mut closing_sequence = 0;
    let mut gaps: Vec<GpuGap> = Vec::new();
    for (sequence, &span) in spans.iter().enumerate().skip(1) {
        if span.start > cur_end {
            busy += cur_end - cur_start;
            gaps.push(GpuGap {
                duration: span.start - cur_end,
                at: cur_end - first,
                before: closing.work,
                after: span.work,
                before_sequence: closing_sequence,
                after_sequence: sequence,
                before_wait_thread: closing.wait_thread,
                after_wait_thread: span.wait_thread,
            });
            (cur_start, cur_end) = (span.start, span.end);
            closing = span;
            closing_sequence = sequence;
        } else if span.end > cur_end {
            cur_end = span.end;
            closing = span;
            closing_sequence = sequence;
        }
    }
    busy += cur_end - cur_start;

    let wall_span = last - first;
    gaps.sort_by(|a, b| b.duration.partial_cmp(&a.duration).expect("gap is finite"));
    eprintln!(
        "[gpu-census] buffers {}  span {:.3}s  busy {:.3}s  occupancy {:.1}%",
        spans.len(),
        wall_span,
        busy,
        busy / wall_span * 100.0
    );
    eprintln!(
        "[gpu-census] submitted {:.3}s (overlap {:.3}s)  idle {:.3}s over {} gaps",
        submitted,
        submitted - busy,
        wall_span - busy,
        gaps.len()
    );

    // Attribute the idle time: a gap is bounded by the work that finished
    // before it and the work that ended it, which is what has to be overlapped.
    let mut by_pair: std::collections::HashMap<(WorkLabel, WorkLabel), (f64, usize)> =
        std::collections::HashMap::new();
    for gap in &gaps {
        let entry = by_pair.entry((gap.before, gap.after)).or_insert((0.0, 0));
        entry.0 += gap.duration;
        entry.1 += 1;
    }
    let mut pairs: Vec<_> = by_pair.into_iter().collect();
    pairs.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).expect("gap total is finite"));
    eprintln!("[gpu-census] idle by boundary (who finished -> who resumed):");
    for ((before, after), (total, count)) in pairs.into_iter().take(8) {
        eprintln!(
            "[gpu-census]   {:7.3}s  n={:<4} {} -> {}",
            total, count, before, after
        );
    }
    eprintln!(
        "[gpu-census] largest single gaps (gpu# is start-time order; wait_tid records the waiter):"
    );
    for gap in gaps.iter().take(8) {
        eprintln!(
            "[gpu-census]   {:7.1}ms  at t={:6.2}s  gpu#{:04}(wait_tid={:?}) {} -> gpu#{:04}(wait_tid={:?}) {}",
            gap.duration * 1e3,
            gap.at,
            gap.before_sequence,
            gap.before_wait_thread,
            gap.before,
            gap.after_sequence,
            gap.after_wait_thread,
            gap.after,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::WorkLabel;

    #[test]
    fn labels_render_without_preformatted_hot_path_strings() {
        assert_eq!(
            WorkLabel::plain("quotient.poseidon").to_string(),
            "quotient.poseidon"
        );
        assert_eq!(
            WorkLabel::tree("merkle.normal.shared", 1 << 20, 135, 4).to_string(),
            "merkle.normal.shared[lg=20,w=135,cap=4]"
        );
        assert_eq!(
            WorkLabel::tree_part("merkle.stream.absorb", 1 << 21, 17, 4, 1, 3).to_string(),
            "merkle.stream.absorb[lg=21,w=17,cap=4,part=2/3]"
        );
    }
}

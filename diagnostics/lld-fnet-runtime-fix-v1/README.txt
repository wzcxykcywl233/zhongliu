LLD F-branch runtime diagnosis / guarded low-memory execution v1
==============================================================

Problem and limits of current evidence
--------------------------------------
The old smoke stopped at Fprobe step0: 197.41s exceeded the 180s production
guard. Its F0 comparison used 46.7/72.4/72.1s and ~21.7GiB on TrackRAD steps,
but ~0.9-1.3s and 5.9GiB on LLD synthetic steps. The later nvidia-smi snapshot
was after exit, not evidence of the memory/competition state during training.
Memory pressure is a hypothesis, not a proven diagnosis of the 197s event.

Nothing in lld-mixed-domain-v1 or an earlier package is edited. No downloads,
installs, image builds/pulls, background jobs, or automatic deletion. All old
data/results mount read-only. No change to the research terms or dataset split.

Candidate runtime
-----------------
F0/Fprobe/F001/F010 keep the same partial fnet layers, sidecar, LR, optimizer,
precision, 128 queries, native TrackRAD frame sequence, 16 synthetic LLD frames,
four refinements and losses. No padding, frame dropping, point thinning or
mixed precision is introduced. M0/S arms do NOT get the new checkpoint runtime.

During F training only, non-reentrant activation checkpointing recomputes:
1. The existing complete fnet.forward (same batch, no frame chunk change).
2. Each existing local-correlation sampling + einsum + corr_mlp block.
The original offline forward is obtained from the imported source, and exactly
one AST-guarded correlation block is replaced in a per-instance method. If its
statements have changed, stop rather than guess. Disk/class-global sources are
untouched. Tensor names and inference checkpoint format remain unchanged.
The encoder recomputation calls its bound forward directly, so the domain hook
captures only the real forward, not an extra backward recomputation.
RNG preservation is enabled. Normalization stays eval. No backward detach is
added to the task path. Inference dispatch uses the original forward.
Reference: https://docs.pytorch.org/docs/stable/checkpoint.html
Checkpointing exchanges extra computation for less saved activation memory;
speedup is NOT assumed. The real GPU gate measures whether it helps here.

Independent bounded diagnosis (no validation or test mounts)
------------------------------------------------------------
Four F arms x original/lowmem x three consecutive paired steps (0,1,2), seed0.
Each job starts fresh, applies the same real AdamW updates and uses unchanged
cached TrackRAD teacher targets / exact synthetic LLD targets. It does not
continue the failed Fprobe optimizer. The original evidence is first, then the
candidate; this is not an interleaved statistical performance benchmark.
Each job is capped at 900 seconds INCLUDING setup and artifact I/O. Completed
jobs are hash-verified and skipped on repeat; incomplete jobs restart their
three-step probe. Preserve partial.json and logs if interrupted or timed out.

Per-step phases: data read/sampling, label binding, host-to-device, forward,
loss/features, task-gradient audit, adversarial-gradient audit, backward,
gradient validation/clipping, optimizer update. Each GPU phase is synchronized.
Snapshots include allocated/reserved peaks and device free/total memory.
PyTorch allocator usage is not total WDDM/other-process GPU usage; snapshots
are NOT continuous monitoring of every allocation made by other programs.
CPU evidence copying and evidence-file I/O are excluded from phase timing;
evidence I/O is reported separately. No change to the 180s production limit.

CPU evidence: outputs, clipped gradients, updated student and classifier state
for each step. Frozen tensors must be bit-identical to their initialization.
Candidate comparison is elementwise with fixed tolerances:
output atol2e-4/rtol1e-5; gradient atol2e-6/rtol1e-3;
updated state atol2e-6/rtol1e-5. Failure retains failed-evidence.pth and a JSON
error report; tolerances are never expanded automatically. Lowmem Fprobe/F0
students must be bit-identical after each of the three updates. These bounds
are implementation checks, not evidence of improved tracking accuracy.

Runtime promotion requires ALL four arms to pass numerical checks and:
- TrackRAD peak allocated memory reduced by at least15%, relative to original.
- All candidate peaks <=85% of physical device memory reported by CUDA.
- Every measured candidate step <=180s, including gradient-audit work.
- Warmed TrackRAD step2 <=90s and <=115% of the corresponding original time.
The warmed comparison uses only one step/arm; it is a safety gate, not a
speed-estimate confidence interval. Full smoke later rechecks all6 paired steps.
If any gate fails, no production training starts. Inspect summary.json and
phases.csv; do not raise the threshold or delete original files to bypass it.

Fresh production queue after successful diagnosis
------------------------------------------------
New root C:\zhongliu\lld-lowmem-v1; no checkpoints from lld-da-v1 are mixed in.
Old completed smoke and failed resume files remain intact in lld-da-v1.
This reruns the small smoke in the new execution protocol, then retains the
original 8x250 pilot, validation-based selection and up-to4x3x2000 full queue.
It is NOT a continuation of the old Fprobe optimizer. Subsequent interruptions
of the NEW queue resume its own model/optimizer/paired records every25steps;
smoke saves every step, and a completed over-limit step is also saved.
An over-limit saved record blocks resume/audit: restarting cannot silently skip
that failed runtime guard. Ordinary interruptions resume the last safe commit.
Production phase timings are in steps.json and current phase-profile.json.
All original frozen-tensor, paired-input, zero-control, reload-inference and
validation-selection audits remain in force. Test38 is explicit only.

The new plan binds the diagnostic receipt, runtime config, all repair source
hashes and all old source/provenance checks. Production revalidates small
diagnostic records/receipts, not gigabytes of archived CPU tensor evidence on
every inference job. Full raw evidence hashes are checked by diagnostic jobs
and their summary before promotion. GPU/software identity is checked again at
each production training/inference job; switching hardware requires diagnosis.

Run remotely (one line at a time)
---------------------------------
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

Only diagnosis, stop and inspect:
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-fnet-runtime-fix-v1\run_repair.ps1" -Stage diagnose

Diagnosis first; automatically start the queue ONLY if every gate passes:
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-fnet-runtime-fix-v1\run_repair.ps1" -Stage all

all stops after validation-10 (or earlier if a gate closes). test-38 still needs
an explicit -Stage test against this new root and frozen validation decision.
Do not launch another GPU training queue while the bounded comparison runs.
No script stops other GPU jobs or changes Windows/Docker/driver settings.

Defaults:
  Original LLD study (read-only): C:\zhongliu\lld-da-v1
  Diagnostic artifacts:           C:\zhongliu\lld-fdiag-v1
  New training results:           C:\zhongliu\lld-lowmem-v1
  Existing LLD data/audit:          C:\zhongliu\lld-mmri-v1 / lld-qc-v1
  Existing TrackRAD label cache:   C:\zhongliu\rs-v1
Reserve at least5GB for diagnostic tensor evidence, PLUS new training results.
An incomplete original baseline probe can still be slow; that is why diagnosis
has job-level time limits and partial phase records, not a full training run.

Verification
------------
test_runtime.py: real small CPU tracker output/gradient/optimizer equivalence,
unchanged inference/state names, checkpoint hook behavior, saved-activation
storage reduction (not a GPU peak claim), safe source guard, promotion gates,
and artifact integrity. test_runner.ps1 exercises mocked Docker queues in
PowerShell5/7, including failure stops, read-only mounts, explicit-only test,
bounded probe commands and path-overlap refusal. These are NOT remote GPU
measurements. Real GPU effectiveness is established only by the diagnostic.

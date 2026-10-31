LLD same-prestate numerical verification v1
==========================================

Purpose
-------
The previous low-memory F0 probe failed its existing AdamW-updated-weight
tolerance at step 1. Only one element failed, but the trajectories already had
different step-0 weights. That comparison alone cannot isolate same-state
calculation noise from accumulated differences. This package diagnoses that
question without modifying any previous implementation, evidence or gate.

Scope
-----
F0 / Fprobe / F001 / F010, seed 0. No task-accuracy evaluation, student selection,
validation-10/test-38 inference, full training, download, image build or pull.
Only the pinned local training image is used; containers have no network.
Original model operations, dimensions, frames, queries, precision, loss,
gradient checks, clipping, optimizers and learning rates are reused unchanged.
The candidate is precisely lld-fnet-runtime-fix-v1/runtime.py, not a new patch.

Design per arm: 22 bounded updates
--------------------------------
1. original: fresh canonical trajectory, steps 0..5. Saves complete prestates
   before steps 0, 1, 2: student and classifier weights/buffers, both AdamW states
   (including first/second moments and counters), parameter-group name order,
   train/eval modes, trainable names, Python/NumPy/Torch CPU/CUDA RNG states.
2. repeat: original implementation, restores canonical prestate 0, then performs
   six continuous updates. This measures repeatability plus its accumulated drift.
3. lowmem: checkpointed implementation, same prestate 0, six continuous updates.
4. same_repeat: original implementation independently restores canonical prestate
   1 and then prestate 2. Each update starts from the original, not its own last step.
5. same_lowmem: checkpointed implementation independently restores those same
   prestates. Step 0 from repeat/lowmem supplies the third same-prestate anchor.

Every restoration is CPU-hashed after loading: a near-match is NOT accepted.
Each step resets RNG exactly as the original engine does. Input video, queries,
cached TrackRAD teacher targets and known synthetic LLD targets are hash-bound
across all jobs. Teacher generation is not rerun. Frozen tensors must stay exact.
The domain-head hook and its gradients remain active in relevant F arms.
Every job is a new container/process; no GPU work runs in parallel.
Allocator cache is cleared before each measured step in ALL modes. Timing thus
describes controlled probes, not uninterrupted production throughput.

Outputs and interpretation
--------------------------
Default new root C:\zhongliu\lld-sstate-v1 (old lld-da-v1, lld-fdiag-v1 and
lld-numerics-v1 are never written).
summary.json: compact per-arm same-prestate and trajectory results.
reports/all/summary.json: per-step evidence, unchanged tolerance decisions,
timing/memory comparisons and receipt hashes.
reports/all/comparisons.csv: compact numeric comparisons by tensor group.
reports/all/tensor-differences.csv: every tensor, including exact-only AdamW
moment differences. AdamW moment differences are diagnostic, not a new gate.
F0-only reports are in reports/F0/. Root summary represents the last report scope.
Arm/job/pre-*.pth and step-*.pth retain reproducible evidence. Frozen tensor
payloads are omitted from POST-update packets only after full exact checks;
PREstate snapshots retain the full student.

All five original elementwise output/gradient/student/classifier thresholds are
unchanged. Out-of-tolerance results do not stop the bounded numeric experiment;
they remain failures in the report. Nonfinite values, changed data/code/state,
frozen tensor changes and missing evidence DO stop it. Do not equate a
successful process exit or complete=true with numerical acceptance.
Original-repeat differences do not automatically justify candidate differences.
The report NEVER creates the previous runtime verified.json receipt, never
modifies its passed flag and always says training_authorized=false.
Six steps and one seed cannot establish long-run training stability or accuracy.

Running
-------
Reserve about 20 GiB free space for the whole run. Each new job requires at
least 3 GiB remaining, otherwise stops without deletion. Do not run another GPU
queue simultaneously. All=88 diagnostic updates,
including intentionally slow original/repeated runs; no full/pilot queue.
Each GPU job has a 3600s wall-clock timeout; CPU report 1800s. Total time depends
on the original memory-pressure behavior and can be hours. The memory-saving
path's previous speed alone must not be used to estimate total diagnosis time.

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-same-state-verification-v1\run_verify.ps1" -Stage all

Optional first-pass F0-only (22 updates), then all reuses its five verified jobs:
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-same-state-verification-v1\run_verify.ps1" -Stage F0

Repeat the exact command after interruption. Completed jobs are deeply verified
and skipped. An incomplete job restarts FROM ITS ORIGINAL START, not from a
partial optimizer history. Logs append; previous upstream evidence is preserved.
For a completed all run, -Stage report repeats CPU analysis without GPU/data mounts.
Do not change tolerances, delete receipts, or use these diagnostic checkpoints
as production-training checkpoints.

Local checks
------------
python -m unittest discover -s diagnostics/lld-same-state-verification-v1 -p "test_*.py"
powershell.exe -NoProfile -File diagnostics/lld-same-state-verification-v1/test_runner.ps1
CPU tests do not substitute for the remote full-shape GPU numerical checks.

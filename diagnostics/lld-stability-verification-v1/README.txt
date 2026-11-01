LLD low-memory stability verification v1
=======================================

Purpose
-------
The preceding same-state diagnosis found small CUDA gradient/update differences
both in original-repeat and low-memory runs. Several old strict tolerances failed;
this package DOES NOT relax them, write their gate, or claim numerical equivalence.
It is a separate user-authorized bounded check of whether the low-memory execution
is stable over 100 updates. It does not authorize production training.

Inputs (read-only)
------------------
C:\zhongliu\lld-sstate-v1 : complete four-arm same-state evidence (20 jobs).
C:\zhongliu\lld-da-v1     : original study identity, not a starting checkpoint.
C:\zhongliu\rs-v1         : paired TrackRAD inputs/teacher-label cache and weights ID.
C:\zhongliu\lld-qc-v1     : unchanged LLD data-manifest.json.
C:\zhongliu\lld-mmri-v1   : original LLD volumes, used as synthetic-motion clips.
The local pinned Training image and original sources must match previous evidence.
No downloads, image builds/pulls, new teacher inference or evaluation-data access.
The old 88 diagnostic updates are NOT rerun; their saved files are hash-verified.

Arms and budget
---------------
F0     : train Mamba and fnet.layer4/conv2/conv3; no domain classifier.
Fprobe : same student, detached classifier; zero GRL, no student gradient path.
F001   : same student, partial-fnet adversarial gradient, maximum strength 0.001.
F010   : same student, partial-fnet adversarial gradient, maximum strength 0.01.
Other student tensors stay bit-exact. These are PARTIAL-BACKBONE adaptation arms,
not fully frozen-backbone arms. Classifiers are training-only.

100 updates per arm, seed 0, same inputs/targets, 50 TrackRAD + 50 synthetic LLD.
Fresh official scaled model and fresh Mamba, not old trained F checkpoints.
Original learning rates, optimizer groups/clips, 250-update GRL warmup, iterations,
128 queries and native-frame sampling remain unchanged. At update 99 the GRL has
not reached its specified maximum; this is not a test of the full training schedule.
LLD labels are known synthetic geometry; TrackRAD labels are frozen teacher output.
LLD is not a real motion-video dataset.

Verification sequence
---------------------
1. CPU plan: verify old source identities and every existing evidence receipt.
2. GPU graph-only check on the first real and synthetic training clips:
   zero-GRL loss must return None gradients for every trainable student parameter
   AND the feature embedding; classifier gradients must remain finite/nonzero.
   A positive-GRL diagnostic and the point task must both reach the encoder.
   No optimizer update, weight/buffer change or gradient-buffer write is allowed.
3. Four sequential low-memory jobs, each 100 updates. Finite objectives/gradients/
   updated weights, frozen-tensor hashes, memory and timings are checked.
   Task/positive adversarial gradient reachability is rechecked at fixed milestones.
4. At 0/25/50/75/100 updates run four fixed training-pool probes (indices100..103).
   These clips are not optimized in this 100-step run; they are still TRAINING-pool
   diagnostics, NOT independent validation. Probes preserve weights, modes and RNG.
   Report point loss, coordinate drift from initialization, V/C and out-of-bounds.
5. CPU paired report. Stop. No pilot/full queue, validation-10 or test-38 inference.

Safeguards and interpretation
----------------------------
Step >180s, nonfinite values, altered inputs/frozen tensors, or allocated-memory
peak above 85% of total device memory stops the job. Peak reserved memory and
device free memory are also recorded; PyTorch allocated memory is not all GPU use.
Per-GPU-job timeout: graph1200s, each100-step run7200s. CPU jobs3600s.
Fixed-probe loss >2x its initial value AND absolute rise >0.01 creates a review flag.
This exploratory threshold is not an accuracy gate and does not automatically stop
or promote an arm. Raw values are saved even below the threshold. Zero flags do not
prove improvement, generalization, clinical utility, or numerical equivalence.
F0/Fprobe are not required to have identical update tensors on nondeterministic
CUDA: the explicit graph check tests isolation, while outputs are reported separately.

Resume
------
Model, both AdamW states, RNG, parameter order/modes and completed data bindings
are saved atomically every10 updates and at probe milestones. An interrupted arm
replays at most9 unsaved updates. Completed arms are hash-checked and reused.
Do not claim bitwise CUDA reproducibility after resume. Explicit stability failures
write failure.json and cannot be skipped by rerunning. Keep failed evidence for review.
Use the same OutputRoot only with unchanged source/input/runtime identities.
OutputRoot must not overlap any input folder; nothing is deleted by this package.
Reserve at least a few GiB for four independent optimizer snapshots.

Run on remote Windows
---------------------
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-stability-verification-v1\run_verify.ps1" -Stage all

Default output: C:\zhongliu\lld-stab-v1
summary.json, fixed-probes.csv, training-steps.csv; graph/ and train/<arm>/ proofs.
Console is compact; detailed logs are retained. Stage plan/graph/train/report are
available for bounded reruns; train also verifies graph and produces the report.

Local tests use CPU and mock Docker only. Actual 4090D execution is performed by
the user; local tests are not presented as GPU stability results.

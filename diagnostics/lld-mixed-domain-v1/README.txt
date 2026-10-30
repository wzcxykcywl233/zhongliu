LLD-MMRI synthetic-motion mixed training / domain-adversarial study v1
==================================================================

Purpose and interpretation
--------------------------
Temporarily replace ACDC in the mixed-domain study WITHOUT editing any ACDC
package, prior checkpoint, dataset, or result. This package is an independent
experiment. It does not use the historical B1-adapted backbone.

LLD-MMRI-MedSAM2 is 3D multisequence liver MRI, NOT a native dynamic video.
498 filename patient groups x 8 sequences = 3984 image/mask pairs. Never treat
adjacent z slices, scan phases, or contrast sequences as motion frames.
Masks were produced with MedSAM2 human-in-the-loop; they are region annotations,
not biological point correspondences. No mask centroid / contour-index pseudo
correspondence is invented.

For each external step: sample patient, then scan, then lesion-bearing z slice;
normalize p1/p99 and resize as in the existing mixed study; sample 128 contour
queries from its mask. Generate 16 frames by a known smooth 2D affine path:
rotation <=8 degrees, translation <=4% of each image dimension, scale +/-4%.
The source frame is exactly identity. Tracks are the forward transformed query
coordinates. Images use inverse resampling. Visibility is in-bounds geometry,
NOT calibrated biological occlusion. No inter-slice or cross-sequence targets.
No target-frame selection or teacher confidence is used to accept LLD samples.
This trains synthetic appearance-motion equivariance, not actual liver motion.

The change from ACDC ALSO changes the motion distribution and target source.
Therefore an LLD-vs-ACDC difference is not an isolated dataset-quality effect.
All eight arms HERE have exactly paired synthetic motion, queries and targets.

License and acquisition
-----------------------
Official card: https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2
Original terms: https://github.com/LMMMEng/LLD-MMRI-Dataset
Original data paper: https://arxiv.org/abs/2402.17246
MedSAM2: https://github.com/bowang-lab/MedSAM2
Noncommercial research only; no external redistribution, except internal use
within the same organization/single site under the original terms. Cite both
the original LLD work and MedSAM2. Recheck terms before sharing derived data.
No images, masks, crops, or downloaded annotation JSON belong in Git.
The user has confirmed this scope; the download command also requires an
explicit -AcceptResearchTerms so accidental standalone invocation cannot fetch.

Pinned official revision: b7e8da56b267587689d8440e8298205f3fc4914e
Downloads approximately 18.7 GB (plan >=25 GB data space, PLUS checkpoint space).
The official API manifest is pinned, then saved locally; each LFS file is checked
against its official SHA256 and size. Small Git files use Git blob SHA1.
Downloads resume through .partial files. Corrupt final files are preserved and
cause a stop; there is no automatic destructive repair. Only download uses the
network. Audit/training/evaluation prohibit container networking and image pulls.
No Docker builds, package installs, or source downloads in the remote queue.

Data audit
----------
Check all 3984 image/mask pairs: hashes, 3D dimensions, size, spacing, origin,
direction, finite values, integral nonnegative masks and usable lesion slices.
A structural error stops training; inspect audit-errors.json. No silent case
drop. Structural success is NOT clinical annotation-quality certification.
Patient ID is MR[-]digits before the suffix; both spellings normalize together.
All sequences/suffixes of one patient stay in one split. Identical decoded 3D
images link patient groups transitively before deterministic 80/20 group split.
Without duplicate merging this gives 398 train / 100 dev patients. Actual counts
are in summary.json. This is not identity certification or near-duplicate QC.
External dev is reserved, not used by the sampler or the selection rule.
Training verifies selected raw scans against audited hashes on every use.

Training arms (same as the current domain-adversarial comparison)
---------------------------------------------------------------
M0      Frozen official backbone + fresh compact Mamba; no domain head.
S0      M0 + detached domain classifier (zero-reversal identity check).
S001    Mamba last-block representation; reversal max .001.
S010    Mamba last-block representation; reversal max .010.
F0      M0 + trainable fnet.layer4 / conv2 / conv3; no domain head.
Fprobe  F0 + detached classifier (zero-reversal identity check).
F001    Partial fnet adaptation + fnet-output reversal max .001.
F010    Partial fnet adaptation + fnet-output reversal max .010.

Transformer and other backbone tensors stay fixed in ALL arms. Only F arms
update the named late encoder layers. Classifier exists only during training.
Mamba LR 5e-5; encoder LR 5e-6; classifier LR 1e-4; AdamW decay1e-5;
separate student/classifier optimizers and clipping10; sigmoid warmup250steps.
One pooled clip embedding, equal domain update counts; no classifier BatchNorm.
TrackRAD and LLD alternate 1:1. No new region loss or segmentation head.
Use the SAME existing coordinate objective in both domains (teacher target on
TrackRAD, known synthetic target on LLD). Finite-loss/gradient checks fail closed.

TrackRAD: exact even-step inputs, augment draws, query hashes, teacher slot and
labels are reused from completed rs-v1/M0 for each seed. This preserves the
40/10/38 protocol and teacher choices. Previous results mount READ-ONLY; ACDC
raw data are never mounted. Old study sources are not edited or hash-bypassed.
New shared input fingerprints distinguish teacher targets from geometry targets.

Queue and gates
---------------
smoke: 8 arms x 6 steps, seed0; real task and adversarial gradients, frozen tensor
       proof, reload+GPU inference, 2 validation cases per arm; S0==M0 and
       Fprobe==F0 exact student tensors required.
pilot: all8 x250 steps, seed0, then official validation-10 metrics.
full:  M0,F0 and <=1 eligible adversarial strength per family, fresh training
       for 2000 steps at each seed0/1/2 (1000 updates per domain). No promotion
       means stop unless F0 itself qualifies vs M0. Full is not pilot continuation.
       Patient-balanced sampling; 2000 steps do NOT traverse all3984 scans.
test:  explicit stage only, frozen full roster, official 38-case report. No test
       threshold tuning or arm selection. Public test is already inspected,
       not a new blind clinical-generalization estimate.

Qualification: higher DSC, D98 drop <=.002, HD95 increase <=.02 vs matched
control; F candidates must pass vs both F0 and M0. Full confirmation also needs
>=2/3 positive DSC seeds and worst delta>=-.002. Compare all with untrained
pretrained reference too. Low domain accuracy alone cannot promote an arm.
Checkpoint+optimizer+input history commits every25steps; restart is deterministic
per step. Keep outputs after interruptions; repeat the unchanged command.
180sec/step runtime guard; serial GPU queue. Do not launch another GPU queue
against the same device while this runs. No automatic stopping of other jobs.

Remote commands (one line at a time; Docker Desktop Linux engine must be ready)
----------------------------------------------------------------------------
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-mixed-domain-v1\run_prepare.ps1" -Stage all -AcceptResearchTerms

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-mixed-domain-v1\run_experiment.ps1" -Stage all

After checking full validation and explicitly deciding to test:
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-mixed-domain-v1\run_experiment.ps1" -Stage test

Defaults:
  Download data    C:\zhongliu\lld-mmri-v1
  Structural audit C:\zhongliu\lld-qc-v1
  Training results C:\zhongliu\lld-da-v1
  Prior cache      C:\zhongliu\rs-v1 (unchanged, must have all3 full M0 seeds)
  Dataset paths    C:\zhongliu\ts-v1\frozen-study.json
  Local images     rs-v1\manifest.json images.Training / images.Evaluation
  Weights          repo\training-checkpoints

Use run_prepare -Stage download to resume only download; -Stage audit is offline.
Use run_experiment -Stage smoke for bounded GPU smoke only. all automatically
continues from smoke to pilot and eligible full, then stops at validation-10.
OutputRoot must be short and separate; never point to existing ACDC results.
Reports: lld-da-v1\reports\pilot or full\validation-10\summary.csv,
paired-deltas.csv, case-deltas.csv, training-diagnostics.csv, selection.json.
Times/feature diagnostics are not clinical quality metrics.

Local checks: test_data.py, test_domain.py, test_runner.ps1, test_prepare.ps1.
Local CPU/mock checks are not a substitute for remote GPU smoke or full audit.
Implementation verification: 31 Python checks, including actual CPU tracker
task/GRL gradients, known affine image/track alignment, pairing, zero controls,
resume, download checksums, patient grouping and selection gates. Both PowerShell
mock queues pass in Windows PowerShell5 and PowerShell7. One official image/mask
pair (MR-391135_1_C+A) passes real SimpleITK geometry/voxel checks and the full
16frame/128query CPU sampler, exact first-frame anchor and repeat fingerprints.
This is a one-scan integration check, NOT an all498-patient quality audit or GPU
training result. Downloaded sample and local test libraries are outside Git.

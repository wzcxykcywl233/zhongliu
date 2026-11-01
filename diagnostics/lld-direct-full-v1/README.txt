LLD direct full comparison v1
============================

Authorization and what this does NOT claim
-----------------------------------------
The user explicitly requested full training after the four 100-update stability
runs passed. This is a NEW full experiment, not a change to the old numerical gate.
Existing original/low-memory exact-tolerance failures remain untouched. The old
summary's full_training_authorized=false is preserved; permission is recorded
separately in this protocol. Stability alone is not accuracy or numerical equality.
The previous 100-step F001/F010 runs were slower than F0/Fprobe. The user elected
to proceed rather than do another timing diagnosis first; this code does not hide
the cost or change the objective to make the runtime appear faster.

Fixed full design (no 250-step pilot, no performance-based early roster pruning)
-------------------------------------------------------------------------------
M0      official scaled backbone frozen + train compact Mamba; no domain head.
F0      train Mamba + fnet.layer4/conv2/conv3; no domain head.
Fprobe  F0 + detached domain classifier; zero adversarial student gradient.
F001    F0 + domain classifier; fnet reversal maximum0.001.
F010    F0 + domain classifier; fnet reversal maximum0.010.

Five arms x three seeds (0,1,2) x2000 updates =15 fresh training jobs,30000 updates.
Each arm/seed:1000 TrackRAD cached-teacher updates +1000 LLD synthetic-motion updates.
All paired inputs/queries/targets are verified. 2000 updates are the full original
training budget, NOT an exhaustive pass over every LLD volume. No validation/dev
images are sampled during training. LLD data are static3D MRI plus known synthetic
affine motion, NOT real medical video; no new biological point labels are claimed.
The existing Train40/Validation10/Test38 separation remains unchanged.

Fresh initialization per seed; no100-step optimizer/checkpoint continuation.
Seed initialization matches rs-v1/M0 sidecar hashes. All five arms start with the
same student tensors per seed. F arms alone install the verified low-memory
encoder/correlation recomputation. No frame/query/precision reductions.
Transformer and other backbone tensors remain frozen. Mamba LR5e-5, encoder5e-6,
classifier1e-4, AdamW decay1e-5, separate clipping10 and optimizers, GRLwarmup250.
Point objective unchanged; no segmentation head or extra region loss.
The domain classifier is training-only and excluded from final.pth.

Verification and runtime
------------------------
Read-only bind to completed same-state evidence and no-alert stability results;
hash-check old files, sources, data audit, weights and pinned local images.
No repeated old GPU diagnosis. No source patch to any frozen package. No image
build/pull, download, package installation or container networking.
Actual zero-GRL graph isolation is supported by the completed stability graph
proof. Exact Fprobe/F0 final CUDA tensor equality is NOT assumed. Their official
metrics are reported as paired controls, including F001/F010 vs Fprobe.
Every completed job verifies frozen tensors and reloads its exported model for
finite GPU inference before the full checkpoint is accepted.

Every25 updates: atomic model+both optimizers+RNG snapshot, paired records and
frozen checks. Restart repeats at most24 unsaved updates; CUDA bitwise replay is
not promised. Complete jobs/outputs are hash-verified and reused. Preserve logs.
180seconds per update,85% allocated-memory budget, finite values and frozen-state
guards remain active. Explicit stability failures cannot be skipped on resume.
Each training container has a12-hour hard timeout and uses only GPU0, serially.
Do not run another GPU training queue concurrently. The100-step timing medians
suggest roughly40 hours for all12 F jobs alone, plus M0, audits and evaluation;
this is an estimate, not a guaranteed total. Allocate several GiB for checkpoints.

Evaluation
----------
all/full run all15 jobs then official validation-10 for15 students +pretrained.
Validation eligibility uses existing thresholds: mean DSC improvement, D98 drop
<=0.002, HD95 increase<=0.02, positive DSC in>=2/3 seeds, worst DSC>=-0.002.
F001/F010 must qualify against M0,F0 AND Fprobe; rank eligible DSC,HD95,name.
Regardless of eligibility, all five arms remain in the predeclared test roster.
No pilot filtering and no selecting on test results. all/full STOP after validation.
Only explicit -Stage test evaluates38 cases after checking the frozen validation
report. The public38-case set has been repeatedly inspected; do not call it blind.
This evaluates trained students, not standalone teacher accuracy.
Pretrained means official scaled model with the same hierarchical inference
profile but no learned sidecar, not the historical untouched submission baseline.

Remote commands
---------------
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-direct-full-v1\run_experiment.ps1" -Stage all

After inspecting validation, explicit test (no training):
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-direct-full-v1\run_experiment.ps1" -Stage test

Defaults: output C:\zhongliu\lld-full-v1 (NEW directory).
Read-only: rs-v1, lld-da-v1, lld-sstate-v1, lld-stab-v1, lld-qc-v1, raw lld-mmri-v1.
Paths to TrackRAD come from ts-v1/frozen-study.json. Pinned local Training and
Evaluation images come from rs-v1/manifest.json. No test dataset mount in all/full.
Stages plan/train/validation also exist. train stops before evaluation; validation
does not train. Logs are compact in the console, complete in each job directory.

Outputs
-------
full/train/seed_*/<arm>/final.pth, resume.pth, proof.json, steps.json, training.log
reports/full/validation-10/summary.csv, per-seed.csv, paired-deltas.csv,
case-deltas.csv, training-diagnostics.csv, selection.json and verified.json.
Explicit test uses reports/full/test-38 with the same report layout.

Local CPU/mock tests are not remote GPU training results. Medical images and
derived private clips must never be committed or redistributed through Git.

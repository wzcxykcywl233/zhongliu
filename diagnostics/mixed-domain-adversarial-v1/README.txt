Mixed ACDC/TrackRAD domain adversarial study v1
============================================

Question
--------
Does weak domain-adversarial supervision improve the mixed M0 student, and does
the answer differ between a frozen backbone and a partially trainable encoder?
This evaluates STUDENTS trained on teacher labels, not standalone teacher scores.
No clinical claim. The repeatedly inspected public test-38 is not a blind test.

Arms (all point-output; no region head)
-------------------------------------
M0     original frozen backbone + trainable compact Mamba; no domain classifier
S0     M0 + detached domain classifier; student must match M0 exactly
S001   M0 + sidecar GRL, maximum strength 0.001
S010   M0 + sidecar GRL, maximum strength 0.010
F0     train Mamba + fnet.layer4 / fnet.conv2 / fnet.conv3; no domain classifier
Fprobe F0 + detached domain classifier; student must match F0 exactly
F001   F0 + fnet-output GRL, maximum strength 0.001
F010   F0 + fnet-output GRL, maximum strength 0.010

Neither F branch changes model architecture or updates Transformer parameters.
They DO update a portion of the backbone, unlike the frozen mainline. The task
gradient is allowed to propagate through the frozen Transformer into the encoder.
Other encoder tensors and Transformer tensors must remain exactly unchanged.
Encoder LR=5e-6, Mamba LR=5e-5, classifier LR=1e-4, AdamW weight_decay=1e-5.
The classifier has a separate optimizer AND gradient clip, preventing its norm
from changing student clipping. GRL ramps from zero to the maximum over 250 steps
using a normalized sigmoid; classifier gradients are not reversed or scaled.

Sidecar tap: mean over actual non-query frames and points in the final Mamba
block, on the last refinement iteration. Existing detached backbone input is
retained. Fnet tap: mean final raw feature maps over spatial positions and actual
non-query frames, before tracker L2 normalization. All native fnet chunks are
included; no temporal padding or frame count/domain ID is fed to the classifier.
There is one classifier observation per clip. Updates alternate domains 1:1 as
in rs-v1; this is not a simultaneous balanced two-clip minibatch.
The classifier is LayerNorm(C)->Linear128->GELU->Linear1, without dropout/BN.

What stays fixed
----------------
Read-only PreviousRoot=rs-v1. Use its exact 2000-step input and cached teacher-label
sequences for seeds 0/1/2. Missing or changed labels stop the run; never regenerate
or patch the old cache. Short-window teacher repair provenance is verified.
78 ACDC external-train patients (exclude patient076 and patient082); external-dev20
unchanged. Only TrackRAD train40 enters training. Validation10 selects. No ACDC dev
images or labels enter this domain objective. Official scaled initialization,
fresh same-seed Mamba, NOT the previously adapted B1 encoder. Both domains use the
existing p1/p99 preprocessing, 128 queries, native ACDC intervals and TrackRAD
clips up to 60 frames. Teacher pool, teacher trajectories, losses, sampling,
learning rate for Mamba and point inference pipeline remain unchanged.
Only M0 trajectory loss is used; W1/D1/G1 objectives are not combined in this v1.

Queue
-----
plan  bind old receipts/source/weights references, no training
smoke 8 arms x 6 native paired updates (seed0), all 8 point deployment forwards
      on two validation cases; real task/GRL gradients and exact zero controls
pilot smoke gate, then 8 arms x 250 updates (seed0) + validation10 official metrics
full  restart from official weights, 2000 updates x 3 seeds for M0/F0 and at most
      one validation-selected strength per adversarial location. No pilot optimizer
      continuation. Evaluate validation10 and freeze final test roster.
all   smoke -> pilot -> automatic eligible full confirmation -> STOP validation10
test  explicit only; evaluate fixed full roster and official no-sidecar reference

Pilot promotion: positive DSC vs same-trainability control; D98 loss <=0.002;
HD95 increase <=0.02. Encoder adversarial candidates must ALSO qualify vs M0.
Choose best qualifying DSC per location, then HD95 and arm name. F0 itself may
trigger full M0/F0 confirmation when it qualifies vs M0 but no GRL arm qualifies.
No qualifying candidate/control => stop after pilot, no large training.
Full decision adds at least two DSC-positive seeds and worst delta >= -0.002.
Always report vs official pretrained, even if all mixed students remain worse.
The final explicit test reports all frozen full arms, not a test-picked winner.
Exact student state equality S0/M0 and Fprobe/F0 is required in smoke AND pilot.
If numerical/environment differences break equality, stop and diagnose; do not
silently loosen tolerance. Smoke is an interface check, not a benefit claim.

Safety and reproducibility
--------------------------
No old source modification; new separate output C:\zhongliu\da-v1 by default.
Pinned existing Training/Evaluation images, no build/pull/container network.
GPU jobs serial. Full command emits compact progress; full logs retained per arm.
Atomic model + both optimizers + records every 25 steps (every smoke step).
Crash resumes committed step, deterministic sampling and per-step RNG reset.
Completed jobs verify hashes before skipping. Exclusive output lock prevents
two queues writing simultaneously. Preserve .lock file; ownership uses open handle.
180s/step and process timeouts stop unexpected resource stalls. Partial encoder
autograd may use more VRAM/time; smoke must pass on the actual remote GPU.
Student-only final.pth reloads through existing setup_model, with GPU verification.
Domain classifiers exist only in resume.pth and add no inference parameters/cost.
Official evaluator is unmodified; metrics are transported byte-identically from
container tmpfs to avoid Windows host permissions / evaluator dependency drift.

Outputs
-------
reports/pilot/validation-10 and reports/full/validation-10:
  summary.csv, per-seed.csv, paired-deltas.csv, case-deltas.csv, selection.json
  training-diagnostics.csv: last100 online domain accuracy (and balanced accuracy),
  domain/task loss, within-domain clip feature spread, time, peak VRAM.
  Accuracy near 50% can mean a weak classifier or collapsed features: NOT proof
  of invariant useful features. These are training diagnostics, not held-out domain
  classification, and never selection criteria. Different organs/motion patterns
  can make global domain alignment harmful. Formal task metrics decide.
Training steps.json also records gradient-path checks, GRL strength and clip features.
reports/full/test-38 exists only after explicit test.

Remote commands (run one at a time)
----------------------------------
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\mixed-domain-adversarial-v1\run_experiment.ps1" -Stage all
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\mixed-domain-adversarial-v1\run_experiment.ps1" -Stage test

Local tests: python -m unittest discover -s diagnostics/mixed-domain-adversarial-v1 -p test_domain.py
PowerShell: diagnostics/mixed-domain-adversarial-v1/test_runner.ps1 (mock Docker)
No real GPU training or remote dataset is exercised by local tests.

Method reference: Ganin et al., Domain-Adversarial Training of Neural Networks,
JMLR 2016, https://jmlr.org/papers/v17/15-239.html
This is a supervised multi-source pseudo-label experiment inspired by GRL, not
an exact reproduction of the paper's unsupervised target-adaptation setting.

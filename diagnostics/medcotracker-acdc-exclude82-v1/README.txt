MedCoTracker patient082 whole-patient exclusion (v1)
==================================================

Purpose
-------
Test the effect of removing patient082, not changing the architecture/teachers/loss.
It has an image-intensity jump on z=1, native frames15->16, with corresponding
large teacher-coordinate jumps. This is evidence for an ablation, not proof that
all its slices are bad or that it caused prior downstream performance changes.

Only patient082 is excluded. The original raw data and native audit remain intact.
Original ACDC development20 is unchanged; eligible pretraining patients80->79.
Known duplicate patient074(dev)/076(train) is deliberately NOT corrected in this
single-factor experiment. Do not call external-dev independent or quality-certified.

Pairing and budget
------------------
Keep original 80-slot shuffled epochs and seeds. At an082 slot, substitute another
RV training patient, balanced in seeded cycles across the other15 RV patients.
Other slots keep original patient/index/epoch, native slice/clip, query RNG and
teacher slot. Their input video SHA must match the original label signature and
the original cached pseudo-label file is copied byte-identically. Replacement
slots get freshly generated labels using the original per-step RNG and teacher.
No original082 label is used. This preserves group exposure, not equal per-patient
weight: remaining RV patients receive a small compensating oversampling.

Each of seeds0/1/2 starts B1 from the SAME ORIGINAL OFFICIAL weights, not from old
adapted B1. Train fnet only for3000 updates at LR1e-6. Then freeze the adapted
backbone and attach an identically initialized fresh compact Mamba; train only
that branch on TrackRAD train40 for1000 updates at LR5e-5. Teachers, losses,
native min(16,T),128 external queries,60-frame/128-query transfer and inference
profile remain unchanged. No ACDC GT segmentation supervision or new filtering.
R1 (no external pretraining) is repeated as a paired training/runtime control.

Historical input: completed C:\zhongliu\mct-full-v1. Its plans, source hashes,
external initialization and all3000 steps must agree with the new plan.
New output: C:\zhongliu\mct-x82-v1 (short separate root). Never use the old root.
Training does not mount historical test results. Official historical metrics are
read only during the matching evaluation report stage, with hash verification.

Run on the remote GPU computer
------------------------------
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\medcotracker-acdc-exclude82-v1\run_experiment.ps1" -Stage all

all: three-seed plan +10-step TrackRAD smoke + actual replacement-label GPU smoke
(one unchanged cached clip and one replacement per teacher), then full3000/1000
training and validation10. Smoke failure prevents full training. Stops after
validation. No download/build/pull within the runner; existing pinned images only.
External optimizer checkpoints every25 updates; repeat unchanged command after
interruption. Completed stages are verified and reused, never silently mixed.

The new replacement-label smoke is NOT an accuracy assessment and updates no
student weights. Short smoke artifacts are never used as full-pretraining state.

Review validation:
Get-Content -LiteralPath "C:\zhongliu\mct-x82-v1\reports\validation-10\summary.csv"
Get-Content -LiteralPath "C:\zhongliu\mct-x82-v1\reports\validation-10\exclusion-vs-original-summary.csv"

Only if explicitly requested, run the already fixed three-seed38-case report:
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\medcotracker-acdc-exclude82-v1\run_experiment.ps1" -Stage test

Interpretation
--------------
These are downstream STUDENT metrics, not teachers' own tracking performance.
summary.csv B1 = excluded082; paired-deltas.csv = newB1 vs newR1/pretrained.
exclusion-vs-original-{paired,summary,cases}.csv = newB1 vs original80-patientB1,
plus repeatedR1 vs originalR1. That comparison isolates the exclusion policy.
If R1 drifts materially, investigate reproducibility before attribution.
DSC/D98 higher is better; HD95/MASD/CD lower. Check all three seeds and case-level
deltas, not only DSC. SD across3 seeds is not independent patient uncertainty.
Do not tune further using public test38; it has already been inspected repeatedly.
No guaranteed gain. Exclusion may help, be neutral or hurt through reduced diversity.

Implementation and tests
------------------------
Full-study common/external/study/train_transfer/reports/runner are isolated copies
of medcotracker-acdc-full-v1, with explicit exclusion/provenance/comparison changes.
No historical runtime files are edited (their source hashes remain valid).
CPU tests: python -m unittest discover -s diagnostics/medcotracker-acdc-exclude82-v1 -p "test_*.py"
PowerShell mock: diagnostics\medcotracker-acdc-exclude82-v1\test_runner.ps1
Mocks are not a remote GPU smoke. The remote -Stage all performs the actual smoke.

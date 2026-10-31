Read-only LLD numerical failure audit v1
======================================
The low-memory F0 probe passed step0, then failed a student weight tolerance
after step1. Earlier output/gradient groups passed, but the original comparator
stopped at its FIRST failing tensor. One reported element is not a claim that
only one model element differs. The updated states are from independently
evolving AdamW trajectories, not identical pre-step model/optimizer snapshots.

This package ONLY reads saved CPU evidence. It does not rerun model inference,
gradients, an optimizer, training or a GPU workload. No dataset mount, image
build/pull, network, new tolerance, gate override, checkpoint edit, or automatic
continuation. The original packages and results remain unchanged/read-only.

Checks:
- Original F-arm completed receipt and its artifact hashes; recorded source
  hashes; paired input/label fingerprints for the completed and failed steps.
- Every output, clipped gradient, student tensor and classifier tensor, using
  the SAME elementwise tolerances as the recorded v1 manifest, not a new bound.
- Full mismatch counts, affected tensor counts, absolute error, tolerance ratio
  and relative-L2. Frozen student tensors require exact equality separately.
- Matched step/domain runtime, never a speed ratio between step2 and step0.
- Worst failed tensor coordinates: saved prior/current gradients and weights,
  plus scalar float64 AdamW formula estimates using the original LR/decay and
  default betas(.9,.999)/epsilon1e-8. This is an explanatory calculation, NOT a
  bitwise replay of GPU AdamW, a causal proof of harmless rounding, or training.
  Missing gradient history is reported unavailable. Step0 has no update trace.

Reference equations: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
Numerical context: https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
CUDA recomputation/reduction effects and optimizer sensitivity must be checked
with controlled same-prestate original/original/lowmem comparisons if needed;
small drift alone does not validate a modified backward path. This report does
not replace that control and cannot authorize production training.

The incomplete candidate has no complete-job receipt. Its observed evidence
hashes are recorded now; the script does not invent a historical certificate.
Existing output inputs.json freezes these observed hashes; changed evidence
requires a new output root. Tensor files are loaded CPU-only with weights_only.

Remote:
git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-numerical-audit-v1\run_audit.ps1"

Input:  C:\zhongliu\lld-fdiag-v1 (read-only)
Output: C:\zhongliu\lld-numerics-v1
summary.json: compact global counts, original limits, matched timing, gate=false.
tensor-differences.csv: all tensors, including ones after the first failure.
update-traces.json: up to10 worst failed tensor-coordinate update explanations.
inputs.json: evidence and audit-code hashes.

No GPU flag. Container has --network none / --pull never, CPU4 and 600s timeout.
Local tests cover native isclose agreement, all-failure counting, scalar AdamW
equations, invalid/changed input rejection, read-only behavior and no gate file.

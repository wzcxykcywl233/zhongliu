LLD download transport repair v1
================================

Failure addressed: urllib ConnectionRefusedError while requesting the official
metadata API. This occurs BEFORE data audit/training. The traceback alone cannot
distinguish a refused target connection from an unusable inherited proxy. v1
retried payload downloads but did not retry the metadata request.

This package leaves diagnostics/lld-mixed-domain-v1 entirely UNCHANGED.
It does not change patient splitting, synthetic motion, supervision, losses,
weights, evaluation, or source fingerprints of an existing study.
Original data audit is invoked from its original file (same byte hash).
Existing research-terms.json, download-manifest.json, completed raw files and
.partial files retain the original format and are reused. No deletion/rebuild.

Changes:
- Metadata requests retry 3 times (30s each); clear final network error.
- Explicit HTTPS endpoint: official https://huggingface.co (default), or the
  THIRD-PARTY public mirror https://hf-mirror.com. Never automatic silent switch.
- Per-process proxy choice: inherit (default), direct, or explicit.
  direct uses an empty Python ProxyHandler, ignoring inherited proxy variables
  for this process only. It cannot bypass Docker Desktop's transparent proxy or
  network/firewall policies and does not change those settings.
  explicit takes a user-supplied HTTP(S) proxy URL reachable FROM THE CONTAINER.
  For a Windows-host proxy use host.docker.internal plus its ACTUAL port;
  localhost/127.0.0.1 inside this container does not refer to the Windows host.
  Embedded credentials are rejected. Authenticated proxies may be supplied via
  the existing container proxy configuration in inherit mode; never paste secrets
  into logs or send Hugging Face tokens to this public mirror downloader.
- Proxy values/credentials/signed redirect URLs are not printed. Diagnostic
  output lists only proxy environment KEY names, mode, endpoint and whether
  loopback proxies were detected. TLS verification stays enabled.
- Mirror catalog must match an independently obtained official catalog digest:
  revision b7e8da56b267587689d8440e8298205f3fc4914e, 7970 files.
  canonical catalog SHA256:
  5a16960e0d9e62108dd99d6362aeaa439fee943b10299844cb0e3ee710bb55e7
  Canonical: json.dumps(original_dataset.catalog(meta),sort_keys=True,
  separators=(',',':'),ensure_ascii=True).encode(), then SHA256.
  Every payload still passes original SHA256/size (LFS) or Git blob SHA1 checks.
  A wrong catalog/revision/checksum is never waived to solve a network error.

Remote Windows commands (run sequentially, do not continue after a failure):

git -C "C:\zhongliu\zhongliu-tuning" pull --ff-only origin main

powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-download-repair-v1\run_repair.ps1" -Stage all -AcceptResearchTerms -DownloadEndpoint "https://hf-mirror.com" -ProxyMode direct

This uses the SAME directories C:\zhongliu\lld-mmri-v1 and C:\zhongliu\lld-qc-v1.
No training is started. Once audit passes, the original training command applies:
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-mixed-domain-v1\run_experiment.ps1" -Stage all

Optional bounded network probe (metadata + README, no patient images, no GPU):
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "C:\zhongliu\zhongliu-tuning\diagnostics\lld-download-repair-v1\run_repair.ps1" -Stage network -DownloadEndpoint "https://hf-mirror.com" -ProxyMode direct

If direct mode times out and this computer requires a proxy, use inherit or
explicit with the correct working proxy address. Do not guess its port. Example
argument shape only (REPLACE actual-port before execution):
  -ProxyMode explicit -ProxyUrl "http://host.docker.internal:actual-port"
Official endpoint remains available by changing -DownloadEndpoint only.
Existing locked catalog allows payload resume without another metadata fetch.
Network probe success only proves metadata/small-file transport, not every CDN
route. Preparation still verifies every downloaded file before training.

Tests: test_download.py and test_repair.ps1 (PowerShell5/7 mocked Docker).
No remote network success, complete dataset download, or training accuracy is
implied by these tests. Remote networking must be verified on that computer.

References:
https://docs.docker.com/engine/cli/proxy/
https://docs.docker.com/desktop/features/networking/
https://hf-mirror.com/
Original data/license and training design: ../lld-mixed-domain-v1/README.txt

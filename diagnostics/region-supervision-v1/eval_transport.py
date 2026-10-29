"""Original evaluator/user/environment. Writable tmpfs, byte-identical metric transport."""
import base64
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    result=subprocess.run([sys.executable,'/opt/app/evaluate.py'],cwd='/opt/app',check=False)
    if result.returncode: raise SystemExit(result.returncode)
    data=Path('/output/metrics.json').read_bytes()
    print('REGION_METRICS_BASE64='+base64.b64encode(data).decode(),flush=True)

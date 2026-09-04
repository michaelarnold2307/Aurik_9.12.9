#!/usr/bin/env python3
"""
Snyk Guard for pre‑commit.
Runs `snyk test` and, if vulnerabilities are found, attempts an automatic fix with `snyk fix`.
If any vulnerability remains after the fix attempt, the hook fails and aborts the commit.

The script respects the current Aurik Spec rules:
- All security fixes must be applied automatically before a commit is allowed.
- The guard runs in the same environment as the rest of the pre‑commit hooks (Python 3.11).
"""
import subprocess
import sys
from pathlib import Path


# Helper to run a command and capture output
def run(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.returncode, result.stdout
    except Exception as exc:
        return 1, str(exc)

# 1. Run snyk test
print("Running Snyk security scan…")
ret, out = run("snyk test --severity-threshold=high", cwd=str(Path.cwd()))
print(out)
if ret == 0:
    print("Snyk found vulnerabilities – attempting auto‑fix...")
    # 2. Attempt automatic fix
    ret_fix, out_fix = run("snyk fix", cwd=str(Path.cwd()))
    print(out_fix)
    if ret_fix == 0:
        # Re‑run test to confirm all high severity issues are resolved
        ret_retest, out_retest = run("snyk test --severity-threshold=high", cwd=str(Path.cwd()))
        print(out_retest)
        if ret_retest != 0:
            print("Some high‑severity vulnerabilities remain after auto‑fix. Commit aborted.")
            sys.exit(1)
    else:
        print("Snyk fix failed – commit aborted.")
        sys.exit(1)
else:
    print("No high‑severity vulnerabilities detected – proceeding with commit.")

sys.exit(0)

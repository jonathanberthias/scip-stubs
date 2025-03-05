# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "scip-stubs",
#     "pyscipopt==5.3.0",
#     "mypy[faster-cache]",
# ]
#
# [tool.uv]
# reinstall-package = ["scip-stubs"]
#
# [tool.uv.sources]
# scip-stubs = {path = ".."}
# ///

"""Usage: `uv run tool/stubtest.py <OPTIONS>`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def main() -> int:
    cmd = [
        "stubtest",
        "--mypy-config-file",
        str(BASE_DIR / "pyproject.toml"),
        "--allowlist",
        str(BASE_DIR / ".allowlist"),
        "pyscipopt",
    ]
    print(*cmd)
    result = subprocess.run(cmd, check=False, env={"FORCE_COLOR": "1", **os.environ})
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

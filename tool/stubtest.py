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
        "--ignore-disjoint-bases",
    ]
    print(*cmd)
    result = subprocess.run(cmd, check=False, env={"FORCE_COLOR": "1", **os.environ})  # noqa: S603
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

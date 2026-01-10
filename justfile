# https://just.systems

default: stubtest typecheck

stubtest:
    uv run tool/stubtest.py

typecheck:
    uv run mypy .
    uv run basedpyright
    uv run pyright

typecheck-all: typecheck
    uv run ty check
    uv run pyrefly check

fmt:
    uvx ruff format
    dprint fmt .

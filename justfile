# https://just.systems

default: stubtest typecheck

stubtest:
    uv run tool/stubtest.py

[parallel]
typecheck: mypy basedpyright pyright
[parallel]
typecheck-all: mypy basedpyright pyright ty pyrefly

mypy:
    uv run mypy .
basedpyright:
    uv run basedpyright
pyright:
    uv run pyright
ty:
    uv run ty check
pyrefly:
    uv run pyrefly check

fmt:
    uvx ruff format
    dprint fmt .

"""
Variant of stubtest that just checks the methods in classes
and ensures the order is the same in the source and in the stubs.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import libcst as cst
from libcst.codemod import (
    CodemodContext,
    VisitorBasedCodemodCommand,
    exec_transform_with_prettyprint,
)
from typing_extensions import override

from pyscipopt import scip

COMPARISON_DUNDERS = {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"}


def get_runtime_order(classname: str) -> list[str]:
    klass = getattr(scip, classname)
    methods = []
    for attr in dir(klass):
        method = getattr(klass, attr)
        if method is getattr(object, attr, None):
            continue  # inherited from object
        methods.append(attr)
    init = ["__init__"] if "__init__" in methods else []
    dunders = [
        m
        for m in methods
        if m.startswith("__") and m.endswith("__") and m != "__init__"
    ]
    non_dunders = [m for m in methods if not (m.startswith("__") and m.endswith("__"))]
    return init + non_dunders + dunders


def get_methods_by_class(lines: list[str]) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {}
    current_class = None
    for i, line in enumerate(lines):
        m = re.match(r"^(?:cdef )?class (\w+)", line)
        if m:
            name = m.group(1)
            if i > 0 and lines[i - 1] != "@type_check_only":
                current_class = name
                classes[current_class] = []
        elif "cdef" in line:
            continue
        elif line and not line.startswith("  ") and not line.startswith("\t"):
            current_class = None
        elif current_class:
            # Some files have 2 spaces indentation
            m = re.match(r"\s{2,4}def (\w+)\(", line)
            if not m:
                continue
            fname = m.group(1)
            if fname in classes[current_class]:
                continue  # an overload, don't repeat
            classes[current_class].append(fname)
    return dict(classes)


def filter_methods(
    runtime_methods: dict[str, list[str]], source_methods: dict[str, list[str]]
) -> dict[str, list[str]]:
    def should_keep(klass: str, method: str) -> bool:
        if klass not in source_methods:
            return False
        if method in source_methods[klass]:
            return True
        if method in COMPARISON_DUNDERS and "__richcmp__" in source_methods[klass]:
            return True
        if method == "name":  # noqa: SIM103
            return True  # property in stubs, attribute in source
        return False

    filtered: dict[str, list[str]] = {}
    for klass, methods in runtime_methods.items():
        if klass not in source_methods:
            continue
        filtered[klass] = [m for m in methods if should_keep(klass, m)]
    return filtered


def load_scip_source() -> list[str]:
    venv_lib = Path(__file__).absolute().parent.parent / ".venv/lib"
    pyscipopt = venv_lib.glob("python*/site-packages/pyscipopt")
    scip_dir = next(pyscipopt)
    sources = scip_dir.glob("*.pxi")

    scip_lines = []
    for scip_file in sources:
        scip_lines.extend(scip_file.read_text().splitlines())
    return scip_lines


def collect_classes(source_lines: list[str]) -> list[str]:
    classes = []
    class_re = re.compile(r"^(?:cdef )?class (\w+)")
    for line in source_lines:
        m = class_re.match(line)
        if m:
            classes.append(m.group(1))
    return classes


def load_stub_source() -> list[str]:
    stub_file = Path(__file__).parent.parent / "src/pyscipopt-stubs/scip.pyi"
    return stub_file.read_text().splitlines()


def compare(classname: str | None, *, runtime: bool) -> int:
    stub_source = load_stub_source()
    stub_methods = get_methods_by_class(stub_source)
    scip_methods = determine_order(runtime=runtime)
    if classname:
        scip_methods = {classname: scip_methods[classname]}
        stub_methods = {classname: stub_methods[classname]}
    scip_keys = set(scip_methods)
    stub_keys = set(stub_methods)
    stub_extra = stub_keys - scip_keys
    assert not stub_extra, f"Extra keys in stub: {stub_extra}"
    scip_extra = scip_keys - stub_keys
    assert not scip_extra, f"Extra keys in scip: {scip_extra}"

    status = 0
    for klass in stub_methods:
        scip_meths = scip_methods[klass]
        stub_meths = stub_methods[klass]
        diff = list(
            difflib.unified_diff(
                stub_meths, scip_meths, "stub", "source", n=1, lineterm=""
            )
        )
        if diff:
            status += 1
            print(klass)
            print(end="\t")
            print("\n\t".join(diff))
    return status


class ReorderCommand(VisitorBasedCodemodCommand):
    """Codemod to reorder methods in classes based on provided order."""

    def __init__(
        self, context: CodemodContext, method_order: dict[str, list[str]]
    ) -> None:
        super().__init__(context)
        self.method_order = method_order
        self.current_class: str | None = None

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_class = node.name.value

    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        assert self.current_class is not None
        if self.current_class not in self.method_order:
            self.current_class = None
            return updated_node

        desired_order = self.method_order[self.current_class]
        method_map: dict[str, list[cst.FunctionDef]] = {}
        for func in updated_node.body.body:
            if isinstance(func, cst.FunctionDef):
                method_map.setdefault(func.name.value, []).append(func)

        reordered_methods = [
            method
            for method_name in desired_order
            for method in method_map.get(method_name, [])
        ]

        new_body = [
            item
            for item in updated_node.body.body
            if not (isinstance(item, cst.FunctionDef) and item.name.value in method_map)
        ] + reordered_methods

        self.current_class = None
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )


def determine_order(*, runtime: bool) -> dict[str, list[str]]:
    scip_source_lines = load_scip_source()
    scip_methods_from_source = get_methods_by_class(scip_source_lines)
    if runtime:
        all_classes = sorted(collect_classes(scip_source_lines))
        scip_methods = {klass: get_runtime_order(klass) for klass in all_classes}
        scip_methods = filter_methods(scip_methods, scip_methods_from_source)
    else:
        scip_methods = scip_methods_from_source
    return scip_methods


def reorder(classname: str | None, *, runtime: bool) -> None:
    scip_methods_by_class = determine_order(runtime=runtime)
    if classname is None:
        for klassname in scip_methods_by_class:
            print(f"# Class {klassname}")
            reorder(klassname, runtime=runtime)
        return
    stub_source_lines = load_stub_source()
    scip_methods = scip_methods_by_class[classname]
    classdef_re = re.compile(rf"(cdef )?class {classname}[\(:]")

    class_start = 0
    for i, line in enumerate(stub_source_lines):
        if classdef_re.match(line):
            class_start = i
            break
    else:
        raise ValueError(f"Class {classname} not found in stub")
    stub_source_lines = stub_source_lines[class_start + 1 :]

    def find_end() -> int:
        for i, line in enumerate(stub_source_lines):
            if line and not line.startswith(" "):
                return i
        raise ValueError("End of class not found")

    class_end = find_end()
    stub_source_lines = stub_source_lines[:class_end]

    stub_source = "\n".join(stub_source_lines)
    funcs = re.findall(
        r'(def (\w+)\(.*?\)(?: -> [, "\[\]\w]+)?:\s*(?:""".*?""")?(?:...)?(?:[# \w]*)(?=\n\s*def|$))',
        stub_source,
        re.DOTALL | re.MULTILINE,
    )
    source_by_name = {fname: func for func, fname in funcs}
    new_source_lines = []
    for meth in scip_methods:
        if meth not in source_by_name:
            print(f"Method {meth} not found in stubs")
            continue
        new_source_lines.append(source_by_name[meth])
    new_source = "    " + "\n    ".join(new_source_lines)
    print(new_source)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare or reorder method order in stubs vs source."
    )
    parser.add_argument("classname", nargs="?", help="Class to reorder.")
    parser.add_argument("--fix", action="store_true", help="Output fixed order.")
    parser.add_argument(
        "--runtime", action="store_true", help="Reorder methods in the runtime order."
    )
    parser.add_argument(
        "-i", "--inplace", action="store_true", help="Modify files in place."
    )
    args = parser.parse_args()
    if args.fix:
        if args.inplace:
            method_order = determine_order(runtime=args.runtime)
            context = CodemodContext()
            command = ReorderCommand(context, method_order)
            stub_file = Path(__file__).parent.parent / "src/pyscipopt-stubs/scip.pyi"
            new_code = exec_transform_with_prettyprint(command, stub_file.read_text())
            if new_code is None:
                print("No source returned.", file=sys.stderr)
                return
            stub_file.write_text(new_code)
        else:
            reorder(args.classname, runtime=args.runtime)
    else:
        status = compare(args.classname or None, runtime=args.runtime)
        sys.exit(status)


if __name__ == "__main__":
    main()

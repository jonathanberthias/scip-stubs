# /// script
# dependencies = [
#    "libcst",
#    "scip-stubs[pyscipopt]",
#    "typing-extensions",
# ]
# requires-python = ">=3.9"
#
# [tool.uv.sources]
# scip-stubs = {path = ".."}
# ///

from __future__ import annotations

import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import cast

import libcst as cst
from libcst.codemod import (
    CodemodContext,
    VisitorBasedCodemodCommand,
    exec_transform_with_prettyprint,
)
from typing_extensions import override


def remove_trailing_spaces(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


class DocstringImputer(VisitorBasedCodemodCommand):
    def __init__(
        self, context: CodemodContext, docstrings: dict[str, dict[str, str]], fix: bool
    ):
        super().__init__(context)
        self.docstrings = docstrings
        self.fix = fix
        self.current_class: str | None = None
        self.all_good = True

    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if not self.current_class:
            return updated_node
        qualname = f"{self.current_class}.{updated_node.name.value}"
        docstring = self.docstrings[self.current_class].get(updated_node.name.value)

        if not docstring:
            if current_docstring := updated_node.get_docstring():
                self.warn(f"Unexpected docstring at {qualname}")
                self.warn("- " + current_docstring)
                self.all_good = False
            return updated_node

        indent2 = " " * 8
        cleaned = remove_trailing_spaces(
            textwrap.indent(inspect.cleandoc(docstring), indent2)
        )
        new_body = cst.IndentedBlock(
            [
                cst.SimpleStatementLine(
                    body=[cst.Expr(cst.SimpleString(f'"""\n{cleaned}\n{indent2}"""'))]
                )
            ]
        )
        function_with_new_docstring = updated_node.with_changes(body=new_body)
        if not updated_node.deep_equals(function_with_new_docstring):
            self.all_good = False
            if self.fix:
                self.warn(f"Updating docstring for {qualname}")
                return function_with_new_docstring
            else:
                self.warn(f"Docstring mismatch at {qualname}")
        return updated_node

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_class = node.name.value

    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.current_class = None
        return updated_node


def load_scip_source() -> list[str]:
    venv_lib = Path(__file__).absolute().parent.parent / ".venv/lib"
    pyscipopt = venv_lib.glob("python*/site-packages/pyscipopt")
    scip_dir = next(pyscipopt)
    sources = scip_dir.glob("*.pxi")

    scip_lines = []
    for scip_file in sources:
        scip_lines.extend(scip_file.read_text().splitlines())
    return scip_lines


def parse_docstrings(source_pxi_lines: list[str]) -> dict[str, dict[str, str]]:
    """Collect all docstrings in the source lines.

    We can't use libcst since this applies to Cython code.
    """
    class_start_re = re.compile(r"(?:cdef )?class (\w+)[\(\)\[\]\"\*\w]*:")

    docstrings = {}
    current_class: str | None = None
    current_class_lines = []

    for line in source_pxi_lines:
        if (
            current_class
            and line
            and not line.startswith(" ")
            and not line.startswith("\t")
        ):
            # line with some non-indented content -> end of class definition
            docstrings[current_class] = parse_docstrings_in_class(current_class_lines)
            current_class = None
            current_class_lines = []
        if not current_class and (m := class_start_re.match(line)):
            # Start of class definition
            current_class = m.group(1)
            continue
        if current_class:
            current_class_lines.append(line)

    return docstrings


FUNCTION_DOCSTRING_RE = re.compile(
    r"""(def (\w+)\(.*?:\s*(["']{3}.*?["']{3})?(?:...)?(?:[# \w]*)(?=\n\s*def|$))""",
    re.DOTALL | re.MULTILINE,
)


def parse_docstrings_in_class(class_lines: list[str]) -> dict[str, str]:
    source = "\n".join(class_lines)
    funcs = cast(list[tuple[str, str, str]], re.findall(FUNCTION_DOCSTRING_RE, source))
    return {fname: docstring.strip(" '\"\n") for _func, fname, docstring in funcs}


def collect_classes(file: Path) -> list[str]:
    class_def_re = re.compile(r"class (\w+)[\(:)]")
    return class_def_re.findall(file.read_text())


def main(fix: bool):
    stub_file = Path(__file__).parent.parent / "pyscipopt" / "scip.pyi"
    scip_source = load_scip_source()

    classes = collect_classes(stub_file)
    docstrings = parse_docstrings(scip_source)
    assert set(docstrings) == set(classes)

    imputer = DocstringImputer(CodemodContext(), docstrings, fix=fix)
    new_source = exec_transform_with_prettyprint(imputer, stub_file.read_text())
    if fix and new_source == stub_file.read_text():
        print("No changes")
    elif fix:
        assert new_source is not None
        stub_file.write_text(new_source)
        exit(1)
    elif not imputer.all_good:
        exit(1)


if __name__ == "__main__":
    fix = len(sys.argv) > 1 and sys.argv[1] == "--fix"
    main(fix)

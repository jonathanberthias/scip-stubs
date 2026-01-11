from __future__ import annotations

import argparse
import ast
import inspect
import re
import site
import sys
import textwrap
from itertools import chain
from pathlib import Path
from typing import NoReturn, cast

import libcst as cst
from libcst.codemod import (
    CodemodContext,
    VisitorBasedCodemodCommand,
    exec_transform_with_prettyprint,
)
from typing_extensions import override

GLOBAL_NAMESPACE = "global_ns"
INDENT = " " * 4
LAMBDA_FUNCTIONS_IN_SOURCE = {"str_conversion"}


def remove_trailing_spaces(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


class DocstringImputer(VisitorBasedCodemodCommand):
    def __init__(
        self,
        context: CodemodContext,
        docstrings: dict[str, dict[str, str]],
        *,
        only_incomplete: bool,
    ) -> None:
        super().__init__(context)
        self.docstrings = docstrings
        self.current_context: str = GLOBAL_NAMESPACE
        self.only_incomplete = only_incomplete

    def contains_incomplete(self, node: cst.FunctionDef) -> bool:
        def is_incomplete(annotation: cst.Annotation | None) -> bool:
            if annotation is None:
                return False
            return (
                isinstance(annotation.annotation, cst.Name)
                and annotation.annotation.value == "Incomplete"
            )

        return_annotation = node.returns
        if is_incomplete(return_annotation):
            return True

        all_params = chain(
            node.params.params, node.params.kwonly_params, node.params.posonly_params
        )
        for param in all_params:
            if param.annotation is None and param.name.value != "self":
                return True
            if is_incomplete(param.annotation):
                return True
        return False

    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        docstring = self.docstrings.get(self.current_context, {}).get(
            updated_node.name.value, ""
        )

        if isinstance(updated_node.body, cst.SimpleStatementSuite):
            trailing_whitespace_or_header = updated_node.body.trailing_whitespace
        elif isinstance(updated_node.body, cst.IndentedBlock):
            trailing_whitespace_or_header = updated_node.body.header
        else:
            raise TypeError(f"Unexpected body type: {type(updated_node.body)}")

        has_incomplete = self.contains_incomplete(updated_node)
        empty_docstring = not docstring or (self.only_incomplete and not has_incomplete)

        if empty_docstring:
            return updated_node.with_changes(
                body=cst.SimpleStatementSuite(
                    body=[cst.Expr(cst.Ellipsis())],
                    trailing_whitespace=trailing_whitespace_or_header,
                )
            )

        indent = INDENT if self.current_context == GLOBAL_NAMESPACE else 2 * INDENT
        cleaned = remove_trailing_spaces(
            textwrap.indent(inspect.cleandoc(docstring), indent)
        )
        docstring_expr = cst.Expr(cst.SimpleString(cleaned.lstrip()))
        return updated_node.with_changes(
            body=cst.IndentedBlock(
                body=[cst.SimpleStatementLine(body=[docstring_expr])],
                header=trailing_whitespace_or_header,
            )
        )

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if self.current_context != GLOBAL_NAMESPACE:
            raise RuntimeError(
                f"Unexpected nested context: {self.current_context}->{node.name.value}"
            )
        self.current_context = node.name.value

    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.current_context = GLOBAL_NAMESPACE
        return updated_node


def load_scip_source() -> list[str]:
    site_packages = Path(site.getsitepackages()[0])
    scip_dir = site_packages / "pyscipopt"
    sources = chain(scip_dir.rglob("*.py"), scip_dir.rglob("*.pxi"))

    scip_lines = []
    for scip_file in sources:
        scip_lines.extend(scip_file.read_text().splitlines())
    return scip_lines


def parse_docstrings(source_pxi_lines: list[str]) -> dict[str, dict[str, str]]:
    """Collect all docstrings in the source lines.

    We can't use libcst since this applies to Cython code.
    """
    class_start_re = re.compile(r"(?:cdef )?class (\w+)[\(\)\[\]\"\*.\w]*:")

    current_context: str = GLOBAL_NAMESPACE
    current_context_lines: list[str] = []
    docstrings: dict[str, dict[str, str]] = {GLOBAL_NAMESPACE: {}}

    for line in source_pxi_lines:
        if line and not line.startswith(" ") and not line.startswith("\t"):
            # line with some non-indented content -> end of context definition
            docstrings[current_context].update(
                parse_docstrings_in_context(current_context_lines)
            )
            current_context = GLOBAL_NAMESPACE
            current_context_lines = []
        if current_context == GLOBAL_NAMESPACE and (m := class_start_re.match(line)):
            # Start of class definition
            current_context = m.group(1)
            docstrings[current_context] = {}
            continue
        if current_context:
            current_context_lines.append(line)

    return docstrings


FUNCTION_DOCSTRING_RE = re.compile(
    r"""(def ((?:__)?[a-zA-Z]\w+)\(.*?:\s*(["']{3}.*?["']{3})?(?:...)?(?:[# \w]*)(?=\n\s*def|$))""",
    re.DOTALL | re.MULTILINE,
)
NORMALIZE_QUOTES_RE = re.compile(r"'''(.*?)'''", re.DOTALL)


def parse_docstrings_in_context(lines: list[str]) -> dict[str, str]:
    source = "\n".join(lines)
    funcs = cast(
        "list[tuple[str, str, str]]", re.findall(FUNCTION_DOCSTRING_RE, source)
    )
    return {
        fname: NORMALIZE_QUOTES_RE.sub(r'"""\1"""', docstring.strip())
        for _func, fname, docstring in funcs
    }


class ClassCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.classes: list[str] = []

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for deco in node.decorator_list:
            if isinstance(deco, ast.Name) and deco.id == "type_check_only":
                break
        else:
            self.classes.append(node.name)
        self.generic_visit(node)


def collect_classes(file: Path) -> list[str]:
    visitor = ClassCollector()
    with file.open("r") as f:
        tree = ast.parse(f.read())
    visitor.visit(tree)
    return visitor.classes


def collect_global_functions(file: Path) -> set[str]:
    global_fn_def = re.compile(r"^def ([a-zA-Z]\w+)\(", re.MULTILINE)
    return set(global_fn_def.findall(file.read_text()))


def find_stub_files() -> list[Path]:
    folder = Path(__file__).parent.parent.joinpath("src/pyscipopt-stubs")
    stubs = list(folder.rglob("*.pyi"))
    if not stubs:
        raise FileNotFoundError(f"No stub files found in {folder}")
    return stubs


def sync_docstrings(*, only_incomplete: bool) -> NoReturn:
    exit_code = 0

    root = Path(__file__).parent.parent
    scip_source = load_scip_source()
    docstrings = parse_docstrings(scip_source)
    source_classes = set(docstrings) - {GLOBAL_NAMESPACE}
    source_functions = set(docstrings[GLOBAL_NAMESPACE])

    for stub_file in find_stub_files():
        classes = collect_classes(stub_file)
        missing_classes = set(classes) - source_classes
        assert not missing_classes, missing_classes

        functions = collect_global_functions(stub_file)
        missing_fns = functions - source_functions - LAMBDA_FUNCTIONS_IN_SOURCE
        assert not missing_fns, missing_fns

        imputer = DocstringImputer(
            CodemodContext(), docstrings, only_incomplete=only_incomplete
        )
        new_source = exec_transform_with_prettyprint(
            imputer,
            stub_file.read_text(),
            format_code=True,
            formatter_args=["ruff", "format", "--stdin-filename", stub_file.name, "-"],
        )
        path = stub_file.relative_to(root)
        if new_source != stub_file.read_text():
            assert new_source is not None
            print(f"{path}: Updating docstrings")
            stub_file.write_text(new_source)
            exit_code = 1
    sys.exit(exit_code)


def remove_docstrings() -> NoReturn:
    root = Path(__file__).parent.parent
    for stub_file in find_stub_files():
        imputer = DocstringImputer(CodemodContext(), {}, only_incomplete=False)
        new_source = exec_transform_with_prettyprint(
            imputer,
            stub_file.read_text(),
            format_code=True,
            formatter_args=["ruff", "format", "--stdin-filename", stub_file.name, "-"],
        )
        path = stub_file.relative_to(root)
        if new_source != stub_file.read_text():
            assert new_source is not None
            print(f"{path}: Removing docstrings")
            stub_file.write_text(new_source)
    sys.exit()


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        description="Sync docstrings from Cython source to stub files."
    )
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="Only add docstrings to incomplete methods.",
    )
    parser.add_argument(
        "--remove", action="store_true", help="Remove all docstrings from stub files."
    )
    args = parser.parse_args()

    if args.remove:
        remove_docstrings()
    else:
        sync_docstrings(only_incomplete=args.only_incomplete)


if __name__ == "__main__":
    main()

"""Usage: `uv run tool/stubtest.py`."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import libcst as cst
from libcst.codemod import (
    CodemodContext,
    VisitorBasedCodemodCommand,
    exec_transform_with_prettyprint,
)
from libcst.codemod.visitors import AddImportsVisitor
from typing_extensions import override

BASE_DIR = Path(__file__).parent.parent


def run_stubtest() -> tuple[list[str], int]:
    cmd = [
        "stubtest",
        "--concise",
        "--mypy-config-file",
        str(BASE_DIR / "pyproject.toml"),
        "--allowlist",
        str(BASE_DIR / ".allowlist"),
        "pyscipopt",
        "--ignore-disjoint-bases",
    ]
    print(*cmd)
    result = subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
    return (
        result.stdout.decode().splitlines() if result.stdout else [],
        result.returncode,
    )


@dataclass
class StubtestError:
    regex: ClassVar[str]
    fixer: ClassVar[type[StubtestFixer[Any]]]

    target: str
    message: str
    params: dict[str, str]

    def __post_init__(self) -> None:
        parts = self.target.rsplit(".")
        assert (parts[0], parts[1]) == ("pyscipopt", "scip"), parts
        assert len(parts) in [3, 4]

    def targets_method(self) -> bool:
        return len(self.target.rsplit(".")) == 4

    def _is_class_name(self, name: str) -> bool:
        name = name.lstrip("_")
        return bool(name) and name[0].isupper()

    def targets_class(self) -> bool:
        return len(self.target.rsplit(".")) == 3 and self._is_class_name(
            self.target.rsplit(".")[2]
        )

    def targets_function(self) -> bool:
        return (
            len(self.target.rsplit(".")) == 3
            and self.target.rsplit(".")[2][0].islower()
        )

    def classname(self) -> str | None:
        parts = self.target.rsplit(".")
        global_scoped = parts[2]
        if self._is_class_name(global_scoped):
            return global_scoped
        return None

    def method_name(self) -> str | None:
        parts = self.target.rsplit(".")
        if len(parts) == 4:
            return parts[3]
        return None

    def function_name(self) -> str | None:
        parts = self.target.rsplit(".")
        if len(parts) == 3:
            return parts[2]
        return None


T = TypeVar("T", bound=StubtestError)


class StubtestFixer(VisitorBasedCodemodCommand, Generic[T]):
    def __init__(self, context: CodemodContext, errors: list[T]) -> None:
        super().__init__(context)
        self.errors = errors
        self.current_context: str | None = None
        self.current_context_errors = [
            e for e in errors if e.targets_function() or e.targets_class()
        ]

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_context = node.name.value
        self.current_context_errors = [
            e
            for e in self.errors
            if e.targets_method() and e.classname() == self.current_context
        ]

    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.current_context = None
        self.current_context_errors = [
            e for e in self.errors if e.targets_function() or e.targets_class()
        ]
        return updated_node


class MissingStubFixer(StubtestFixer["MissingStubError"]):
    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        new_methods = [
            cst.FunctionDef(
                name=cst.Name(value=error.method_name() or "unknown"),
                params=cst.Parameters(params=[cst.Param(cst.Name("self"))]),
                body=cst.SimpleStatementSuite(body=[cst.Expr(value=cst.Ellipsis())]),
                returns=cst.Annotation(cst.Name("Incomplete")),
            )
            for error in self.current_context_errors
            if error.targets_method()
        ]
        if new_methods:
            AddImportsVisitor.add_needed_import(self.context, "_typeshed", "Incomplete")
        updated_node = updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=[*updated_node.body.body, *new_methods]
            )
        )
        return super().leave_ClassDef(original_node, updated_node)

    @override
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_functions = [
            cst.FunctionDef(
                name=cst.Name(value=error.function_name() or "unknown"),
                params=cst.Parameters(),
                body=cst.SimpleStatementSuite(body=[cst.Expr(value=cst.Ellipsis())]),
                returns=cst.Annotation(cst.Name("Incomplete")),
            )
            for error in self.current_context_errors
            if error.targets_function()
        ]
        if new_functions:
            AddImportsVisitor.add_needed_import(self.context, "_typeshed", "Incomplete")
        new_classes = [
            cst.ClassDef(
                name=cst.Name(value=error.classname() or "Unknown"),
                body=cst.IndentedBlock(body=[cst.SimpleStatementLine([cst.Pass()])]),
            )
            for error in self.current_context_errors
            if error.targets_class()
        ]
        return updated_node.with_changes(
            body=list(updated_node.body) + new_functions + new_classes
        )


@dataclass
class MissingStubError(StubtestError):
    regex: ClassVar[str] = r"is not present in stub$"
    fixer = MissingStubFixer


class InconsistentDefaultFixer(StubtestFixer["InconsistentDefaultError"]):
    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        for error in self.current_context_errors:
            if (
                not error.targets_method()
                or error.method_name() != updated_node.name.value
            ):
                continue
            param_name = error.params["param"]
            default_value = error.params["default"]
            new_params = []
            for param in updated_node.params.params:
                if param.name.value == param_name:
                    new_param = param.with_changes(
                        default=cst.Expr(cst.parse_expression(default_value))
                    )
                    new_params.append(new_param)
                else:
                    new_params.append(param)
            updated_node = updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params)
            )
        return updated_node


@dataclass
class InconsistentDefaultError(StubtestError):
    regex: ClassVar[str] = (
        r"is inconsistent, runtime parameter \"(?P<param>\w+)\" has a default value of (?P<default>.+), which is different from stub parameter default .+"
    )
    fixer = InconsistentDefaultFixer


class MissingParameterInStubFixer(StubtestFixer["MissingStubParameter"]):
    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        for error in self.current_context_errors:
            if error.method_name() != updated_node.name.value:
                continue
            param_name = error.params["param"]
            new_param = cst.Param(
                name=cst.Name(param_name),
                annotation=cst.Annotation(annotation=cst.Name("Incomplete")),
            )
            AddImportsVisitor.add_needed_import(self.context, "_typeshed", "Incomplete")
            # Add it right before the parameters with defaults
            first_defaulted_idx = next(
                (
                    i
                    for i, p in enumerate(updated_node.params.params)
                    if p.default is not None
                ),
                len(updated_node.params.params),
            )
            new_params = [
                *updated_node.params.params[:first_defaulted_idx],
                new_param,
                *updated_node.params.params[first_defaulted_idx:],
            ]
            updated_node = updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params)
            )
        return updated_node


@dataclass
class MissingStubParameter(StubtestError):
    regex: ClassVar[str] = (
        r'is inconsistent, stub does not have parameter "(?P<param>\w+)"'
    )
    fixer = MissingParameterInStubFixer


class MissingDefaultFixer(StubtestFixer["MissingDefaultInStub"]):
    def __init__(
        self, context: CodemodContext, errors: list[MissingDefaultInStub]
    ) -> None:
        super().__init__(context, errors)
        self.default = cst.SimpleString('"Wrong default"')

    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        for error in self.current_context_errors:
            if error.method_name() != updated_node.name.value:
                continue
            param_name = error.params["param"]
            new_params = []
            for param in updated_node.params.params:
                if param.name.value == param_name:
                    new_param = param.with_changes(default=cst.Expr(self.default))
                    new_params.append(new_param)
                else:
                    new_params.append(param)
            # ensure params with default are at the end
            new_params.sort(key=lambda p: p.default is not None)
            updated_node = updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params)
            )
        return updated_node


@dataclass
class MissingDefaultInStub(StubtestError):
    regex: ClassVar[str] = (
        r'is inconsistent, runtime parameter "(?P<param>\w+)" has a default value but stub parameter does not'
    )
    fixer = MissingDefaultFixer


class WrongNameFixer(StubtestFixer["WrongNameInStubError"]):
    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        this_function_errors = [
            e
            for e in self.current_context_errors
            if e.method_name() == updated_node.name.value
        ]
        if not this_function_errors:
            return updated_node

        # Reorder parameters according to expected order
        current_parameter_order = [
            param.name.value for param in updated_node.params.params
        ]
        desired_parameter_order = current_parameter_order.copy()
        for error in this_function_errors:
            stub_name = error.params["stub_name"]
            runtime_name = error.params["runtime_name"]
            if stub_name in current_parameter_order:
                index = current_parameter_order.index(stub_name)
                desired_parameter_order[index] = runtime_name
            else:
                desired_parameter_order.append(runtime_name)

        param_dict = {param.name.value: param for param in updated_node.params.params}

        new_params = []
        seen_default = False
        for param_name in desired_parameter_order:
            param = param_dict[param_name]
            if param.default is not None:
                seen_default = True
            elif seen_default:
                # Ensure parameters have default if there are parameters with defaults before them
                param = param.with_changes(
                    default=cst.Expr(cst.SimpleString('"Wrong default"'))
                )
            new_params.append(param)

        updated_node = updated_node.with_changes(
            params=updated_node.params.with_changes(params=new_params)
        )
        return updated_node


@dataclass
class WrongNameInStubError(StubtestError):
    regex: ClassVar[str] = (
        r'is inconsistent, stub parameter "(?P<stub_name>\w+)" differs from runtime parameter "(?P<runtime_name>\w+)"'
    )
    fixer = WrongNameFixer


class NotAFunctionFixer(StubtestFixer["NotAFunctionError"]):
    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if self.current_context_errors:
            new_statements = []
            for error in self.current_context_errors:
                new_statements.append(
                    cst.SimpleStatementLine(
                        body=[
                            cst.AnnAssign(
                                target=cst.Name(error.method_name() or "unknown"),
                                annotation=cst.Annotation(cst.Name("Incomplete")),
                                value=None,
                            )
                        ]
                    )
                )
                AddImportsVisitor.add_needed_import(
                    self.context, "_typeshed", "Incomplete"
                )
            updated_node = updated_node.with_changes(
                body=updated_node.body.with_changes(
                    body=[*updated_node.body.body, *new_statements]
                )
            )

        return super().leave_ClassDef(original_node, updated_node)

    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef | cst.RemovalSentinel:
        for error in self.current_context_errors:
            if error.method_name() != updated_node.name.value:
                continue
            return cst.RemoveFromParent()
        return updated_node


@dataclass
class NotAFunctionError(StubtestError):
    regex: ClassVar[str] = r"is not a function$"
    fixer = NotAFunctionFixer


class RemoveUnknownFixer(StubtestFixer["NotAtRuntimeError"]):
    @override
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef | cst.RemovalSentinel:
        for error in self.current_context_errors:
            if error.method_name() != updated_node.name.value:
                continue
            return cst.RemoveFromParent()
        return updated_node


@dataclass
class NotAtRuntimeError(StubtestError):
    regex: ClassVar[str] = r"is not present at runtime$"
    fixer = RemoveUnknownFixer


class MissingDisjointBaseFixer(StubtestFixer["MissingDisjointBaseError"]):
    @override
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        updated_node = super().leave_ClassDef(original_node, updated_node)
        for error in self.current_context_errors:
            if (
                not error.targets_class()
                or error.classname() != updated_node.name.value
            ):
                continue
            disjoint_decorator = cst.Decorator(decorator=cst.Name("disjoint_base"))
            updated_node = updated_node.with_changes(
                decorators=[*updated_node.decorators, disjoint_decorator]
            )
            AddImportsVisitor.add_needed_import(
                self.context, "typing_extensions", "disjoint_base"
            )
        return updated_node


@dataclass
class MissingDisjointBaseError(StubtestError):
    regex: ClassVar[str] = (
        r"is a disjoint base at runtime, but isn't marked with @disjoint_base in the stub$"
    )
    fixer = MissingDisjointBaseFixer


def parse_errors(lines: list[str]) -> list[StubtestError]:
    error_types = StubtestError.__subclasses__()
    errors: list[StubtestError] = []

    for line in lines:
        target, message = line.split(" ", maxsplit=1)
        if ".scip." not in target:
            continue
        for error_type in error_types:
            match = re.search(error_type.regex, message)
            if match:
                errors.append(
                    error_type(target=target, message=message, params=match.groupdict())
                )
                break
    return errors


def fix_stubtest_issues() -> None:  # noqa: C901
    stub_file = BASE_DIR / "src" / "pyscipopt-stubs" / "scip.pyi"
    for _ in range(6):
        errors, _ = run_stubtest()
        parsed_errors = parse_errors(errors)
        if not parsed_errors:
            if errors:
                print("Some stubtest issues remain but could not be parsed:")
                for line in errors:
                    print(line)
                return
            print("All stubtest issues fixed.")
            return
        errors_by_type: dict[type[StubtestError], list[StubtestError]] = {}
        for error in parsed_errors:
            errors_by_type.setdefault(type(error), []).append(error)
        for error_type, errors_of_type in errors_by_type.items():
            print(
                f"Applying fixer for {len(errors_of_type):3} errors of type {error_type.__name__}"
            )
            if len(errors_of_type) < 3:
                for e in errors_of_type:
                    print(f" - {e.target}: {e.message}")
            imputer = error_type.fixer(CodemodContext(), errors_of_type)
            new_source = exec_transform_with_prettyprint(imputer, stub_file.read_text())
            assert new_source is not None
            stub_file.write_text(new_source)
    remaining_errors, exit_code = run_stubtest()
    if remaining_errors:
        print("Some stubtest issues could not be fixed automatically:")
        for line in remaining_errors:
            print(line)
    sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Check type stubs conform to the pyscipopt package."
    )
    parser.add_argument(
        "--fix", action="store_true", help="Automatically try to fix some errors."
    )
    args = parser.parse_args()

    if args.fix:
        fix_stubtest_issues()
    else:
        errors, exit_code = run_stubtest()
        for line in errors:
            print(line)
        sys.exit(exit_code)

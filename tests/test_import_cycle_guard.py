from __future__ import annotations

import ast
from pathlib import Path


CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
DELETE_MARKER = "//DELETE THIS FILE"

ALLOWED_CYCLE_SUPERSETS = [
    frozenset(
        {
            "backend_fidelity",
            "camera_noise",
            "experiment_contracts",
            "high_fidelity_fluorescence",
        }
    ),
]


def _local_import_graph() -> dict[str, set[str]]:
    modules = {
        path.stem: path
        for path in CODEBASE.glob("*.py")
        if path.read_text(encoding="utf-8").strip() != DELETE_MARKER
    }
    graph = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name.split(".", 1)[0]
                    if imported in modules:
                        graph[name].add(imported)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module.split(".", 1)[0]
                if imported in modules:
                    graph[name].add(imported)
    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[frozenset[str]]:
    stack: list[str] = []
    on_stack: set[str] = set()
    index_by_node: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[frozenset[str]] = []

    def visit(node: str) -> None:
        index_by_node[node] = len(index_by_node)
        lowlink[node] = index_by_node[node]
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in index_by_node:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], index_by_node[neighbor])

        if lowlink[node] == index_by_node[node]:
            component: set[str] = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(frozenset(component))

    for node in graph:
        if node not in index_by_node:
            visit(node)
    return components


def test_import_cycles_do_not_expand_or_add_new_cycles() -> None:
    cycles = _strongly_connected_components(_local_import_graph())
    unexpected = []
    for cycle in cycles:
        if not any(cycle <= allowed for allowed in ALLOWED_CYCLE_SUPERSETS):
            unexpected.append(tuple(sorted(cycle)))

    assert unexpected == []

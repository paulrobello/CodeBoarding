"""Unified docs rendering from ``analysis.json``.

Collapses the four ``generate_markdown/html/mdx/rst`` functions in
``github_action.py`` and ``generate_markdown_docs`` in the former
``markdown.py`` into a single table-driven entry point.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents.agent_responses import AnalysisInsights, Relation
from diagram_analysis.analysis_json import build_id_to_name_map, parse_unified_analysis
from output_generators.html import generate_html_file
from output_generators.markdown import generate_markdown_file
from output_generators.mdx import generate_mdx_file
from output_generators.sphinx import generate_rst_file
from utils import sanitize

logger = logging.getLogger(__name__)


def _ancestor_in_level(component_id: str, level_ids: set[str]) -> str | None:
    """Return the closest ancestor (or the id itself) that lives in *level_ids*.

    Walks the dotted hierarchy from leaf toward root. Returns ``None`` if no
    ancestor is in the level (the component is outside this view).
    """
    if component_id in level_ids:
        return component_id
    parts = component_id.split(".")
    for i in range(len(parts) - 1, 0, -1):
        ancestor = ".".join(parts[:i])
        if ancestor in level_ids:
            return ancestor
    return None


def project_relations_to_level(
    global_relations: list[Relation],
    level_component_ids: set[str],
    id_to_name: dict[str, str],
) -> list[Relation]:
    """Project a global leaf-only relation set onto the components visible at a level.

    For each leaf relation ``src_id -> dst_id``, roll up both endpoints to the
    ancestor that lives in ``level_component_ids``. Drops edges where both
    endpoints collapse to the same level component (would render as self-loop),
    and aggregates duplicates by keeping the first label seen and summing
    ``edge_count``.

    Why: ``components_relations`` is stored once at the root as the deepest
    cross-boundary set. Each rendered level needs only edges visible at *that*
    level — mermaid would otherwise reference component names that aren't
    declared as nodes ("phantom nodes"), which is exactly what Devin flagged
    on PR #246.
    """
    aggregated: dict[tuple[str, str], Relation] = {}
    for rel in global_relations:
        src = _ancestor_in_level(rel.src_id, level_component_ids)
        dst = _ancestor_in_level(rel.dst_id, level_component_ids)
        if src is None or dst is None or src == dst:
            continue
        key = (src, dst)
        existing = aggregated.get(key)
        if existing is None:
            aggregated[key] = Relation(
                relation=rel.relation,
                src_name=id_to_name.get(src, src),
                dst_name=id_to_name.get(dst, dst),
                src_id=src,
                dst_id=dst,
                edge_count=rel.edge_count,
                is_static=rel.is_static,
            )
        else:
            existing.edge_count += rel.edge_count
            existing.is_static = existing.is_static or rel.is_static
    return list(aggregated.values())


# Writer-name lookup (resolved at call time so @patch on this module's names works).
# Only ``.md`` accepts ``demo``.
_FORMAT_WRITERS: dict[str, tuple[str, bool]] = {
    ".md": ("generate_markdown_file", True),
    ".html": ("generate_html_file", False),
    ".mdx": ("generate_mdx_file", False),
    ".rst": ("generate_rst_file", False),
}


def _load_entries(analysis_path: Path) -> list[tuple[str, AnalysisInsights, set[str]]]:
    """Return ``(filename, analysis, expanded_component_ids)`` for root + each sub-analysis.

    The root entry's filename is supplied by the caller (see :func:`render_docs`);
    here we emit it as ``"__root__"`` for the caller to rename.

    Each entry's ``analysis.components_relations`` is replaced with the global
    leaf set projected down to the components visible at that level (see
    ``project_relations_to_level``). This is what keeps the rendered mermaid
    diagrams free of phantom nodes once cross-boundary relations are stored as
    a single global leaf set at the root of ``analysis.json``.
    """
    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root_analysis, sub_analyses = parse_unified_analysis(data)
    id_to_name = build_id_to_name_map(root_analysis, sub_analyses)
    global_relations = list(root_analysis.components_relations)

    root_ids = {c.component_id for c in root_analysis.components}
    root_analysis.components_relations = project_relations_to_level(global_relations, root_ids, id_to_name)
    root_expanded = set(sub_analyses.keys())
    entries: list[tuple[str, AnalysisInsights, set[str]]] = [("__root__", root_analysis, root_expanded)]

    for comp_id, sub_analysis in sub_analyses.items():
        sub_ids = {c.component_id for c in sub_analysis.components}
        sub_analysis.components_relations = project_relations_to_level(global_relations, sub_ids, id_to_name)
        sub_expanded = {c.component_id for c in sub_analysis.components if c.component_id in sub_analyses}
        comp_name = id_to_name.get(comp_id, comp_id)
        entries.append((sanitize(comp_name), sub_analysis, sub_expanded))

    return entries


def render_docs(
    analysis_path: Path,
    *,
    repo_name: str,
    repo_ref: str,
    temp_dir: Path,
    format: str = ".md",
    root_name: str = "overview",
    demo_mode: bool = False,
) -> None:
    """Render an ``analysis.json`` into *format* docs under *temp_dir*.

    - ``repo_ref`` is the fully-formed link prefix (e.g.
      ``https://github.com/x/y/blob/main/.codeboarding``); this function
      does not construct it, because callers disagree on the tail segment.
    - ``root_name`` names the top-level file (``"overview"`` in the GitHub
      Action, ``"on_boarding"`` in the CLI workflow).
    - ``demo_mode`` is honored only by writers that accept it (currently
      markdown); it is silently ignored by others.
    """
    if format not in _FORMAT_WRITERS:
        raise ValueError(f"Unsupported extension: {format}")

    writer_name, accepts_demo = _FORMAT_WRITERS[format]
    writer: Callable[..., Any] = globals()[writer_name]
    for fname, analysis, expanded in _load_entries(analysis_path):
        out_name = root_name if fname == "__root__" else fname
        logger.info("Generating %s for: %s", format, out_name)
        kwargs: dict[str, Any] = {
            "repo_ref": repo_ref,
            "expanded_components": expanded,
            "temp_dir": temp_dir,
        }
        if accepts_demo:
            kwargs["demo"] = demo_mode
        writer(out_name, analysis, repo_name, **kwargs)

"""Result format serializers for LLM-native output.

Provides markdown rendering for call_endpoint results, optimized for LLM
comprehension per research (Markdown-KV: 60.7% accuracy, markdown-table:
51.9% at 25K tokens — improvingagents.com benchmark).

Rendering strategy:
- ≤20 items: Markdown-KV (per-entity ### headings, - key: value pairs)
- >20 items: Markdown table (denser, better accuracy/token ratio)
- Keyed batch results: per-source ## sections, each rendered via the above
- Empty/error/non-list results: fallback to compact JSON

All output is sanitized against markdown injection — hostile data in entity
names or values cannot forge section headers, fake footers, or inject
prompt instructions into the LLM context window.

Design doc: docs/design/output-formats.md
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "stable_key_union",
    "to_markdown_kv",
    "to_markdown_table",
    "render_text",
    "DEFAULT_FORMAT",
]

DEFAULT_FORMAT = "markdown"

# Threshold for switching markdown format from KV (per-entity headings)
# to table (denser). KV comprehension is higher but costs more tokens.
MARKDOWN_KV_THRESHOLD = 20


def stable_key_union(rows: list[dict[str, Any]]) -> list[str]:
    """Compute first-seen-order union of all keys across rows.

    Used to derive a stable column schema from a full cached result list,
    so columns don't drift across pagination offsets.
    """
    seen: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            for k in row:
                if k not in seen:
                    seen[k] = len(seen)
    return sorted(seen, key=seen.get)


# ---------------------------------------------------------------------------
# Cell formatting (shared across formats)
# ---------------------------------------------------------------------------


def _sanitize_markdown(s: str) -> str:
    """Neutralize markdown structural characters in untrusted cell values.

    Prevents injection attacks where hostile data in the knowledge graph
    could forge section headers, fake footers, fake footer fields, or
    inject prompt instructions into the LLM context window.

    Transformations:
    - Newlines / carriage returns / tabs → single space (prevents multi-line injection)
    - Leading '#' → escaped (prevents fake headings)
    - Leading '-' / '*' → escaped (prevents fake list items)
    - Leading '|' → escaped (prevents fake table cells)
    - Embedded '|' → escaped as '\\|' (prevents fake footer field separators
      since the footer joins parts with ' | ')
    - Literal '---' sequences → '- - -' (prevents fake horizontal rules/footers)
    - '```' backticks → escaped (prevents fake code blocks)
    """
    if not s:
        return s

    # Collapse whitespace (newlines/tabs become spaces)
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Neutralize footer separator '---' anywhere (used in our footer; forging it
    # could fake metadata). Split to avoid touching legitimate hyphens in names.
    s = s.replace("---", "- - -")

    # Escape pipes to prevent footer-field injection (footer joins parts with ' | ')
    # and table-cell injection (markdown table delimiter is '|'). Doing this
    # universally is simpler than tracking which path the value flows through.
    s = s.replace("|", "\\|")

    # Escape structural characters at the start of the (now single-line) value
    stripped = s.lstrip()
    if stripped:
        first = stripped[0]
        if first in "#-*":
            leading_ws = s[: len(s) - len(stripped)]
            s = leading_ws + "\\" + stripped

    # Neutralize fenced code blocks
    s = s.replace("```", "\\`\\`\\`")

    return s


def _format_value(val: Any) -> str:
    """Format a single value for human/LLM-readable output.

    Values are sanitized against markdown injection: hostile data cannot
    forge headings, footers, or code blocks in the rendered output.
    Sanitization applies recursively to:
    - scalar values (final str conversion)
    - nested dict KEYS (hostile keys can inject newlines otherwise)
    - nested dict values (recursive call re-enters _format_value)
    - nested list items (recursive call)

    - None → empty string
    - Nested dict → compact k=v pairs (no braces)
    - Nested list → comma-separated
    - Scalars → str(), sanitized
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, dict):
        # Flatten to "reach=15, bel=27". Keys must be sanitized too —
        # a hostile key like "x\n---\n### SYSTEM" would otherwise inject
        # raw markdown structure into the rendered output.
        return ", ".join(
            f"{_sanitize_markdown(str(k))}={_format_value(v)}"
            for k, v in val.items()
        )
    if isinstance(val, list):
        return ", ".join(_format_value(v) for v in val)
    return _sanitize_markdown(str(val))


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def to_markdown_kv(
    rows: list[dict[str, Any]],
    columns: list[str],
    heading_key: str | None = None,
) -> str:
    """Serialize rows as Markdown key-value blocks (one heading per entity).

    Best LLM comprehension format (60.7% accuracy in benchmarks).
    Each entity gets a ### heading (from heading_key or first column),
    followed by `- key: value` pairs.

    Parameters
    ----------
    rows : list[dict]
        Result dicts to serialize.
    columns : list[str]
        Column names to include.
    heading_key : str, optional
        Which key to use as the ### heading. Defaults to "name" if present,
        else first column.
    """
    if not rows or not columns:
        return ""

    # Determine heading key
    if heading_key is None:
        heading_key = "name" if "name" in columns else columns[0]

    # Remaining columns (exclude heading key)
    detail_cols = [c for c in columns if c != heading_key]

    blocks = []
    for row in rows:
        if not isinstance(row, dict):
            blocks.append(f"### {_sanitize_markdown(str(row))}")
            continue

        heading = _format_value(row.get(heading_key)) or "(unnamed)"
        lines = [f"### {heading}"]
        for col in detail_cols:
            val = row.get(col)
            if val is not None:
                lines.append(f"- {_sanitize_markdown(col)}: {_format_value(val)}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def to_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> str:
    """Serialize rows as a Markdown pipe-delimited table.

    Best accuracy/token ratio (51.9% accuracy at 25K tokens).

    Parameters
    ----------
    rows : list[dict]
        Result dicts to serialize.
    columns : list[str]
        Column names (determines column order).
    """
    if not rows or not columns:
        return ""

    # Header — sanitize column names (escapes pipes, newlines, etc.)
    safe_cols = [_sanitize_markdown(c) for c in columns]
    header = "| " + " | ".join(safe_cols) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    # Data rows. _format_value sanitizes scalars; non-dict rows go through
    # _sanitize_markdown directly. Both paths escape pipes, so the table
    # delimiter can't be forged from cell content.
    data_lines = []
    for row in rows:
        if not isinstance(row, dict):
            cells = [_sanitize_markdown(str(row))] + [""] * (len(columns) - 1)
        else:
            cells = [_format_value(row.get(col)) for col in columns]
        data_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + data_lines)


# ---------------------------------------------------------------------------
# Unified text renderer
# ---------------------------------------------------------------------------


def _render_footer(response: dict[str, Any]) -> str:
    """Build a compact metadata footer from the response envelope.

    All interpolated values are sanitized for defense in depth — even
    though most fields come from server-controlled mappings, hostile
    upstream data (e.g. from grounding failures) should never corrupt
    the footer structure.
    """
    parts = []

    pagination = response.get("pagination")
    if pagination:
        parts.append(
            f"{_sanitize_markdown(str(pagination.get('returned', '?')))} of "
            f"{_sanitize_markdown(str(pagination.get('total', '?')))}"
        )
        if pagination.get("has_more"):
            parts.append(
                f"next: offset={_sanitize_markdown(str(pagination.get('next_offset', '?')))}"
            )
    elif "total" in response:
        parts.append(f"{_sanitize_markdown(str(response['total']))} results")

    # Batch call: report partial failures so agents can retry
    if "failed" in response:
        failed = response["failed"]
        if isinstance(failed, dict) and failed:
            fail_keys = ", ".join(_sanitize_markdown(str(k)) for k in list(failed)[:5])
            parts.append(f"failed: {len(failed)} ({fail_keys})")

    if "suggested_next" in response:
        nav = response["suggested_next"]
        nav_compact = ", ".join(
            f"{_sanitize_markdown(str(n.get('from', '?')))}->"
            f"{_sanitize_markdown(str(n.get('to', '?')))}"
            for n in nav[:3]
        )
        parts.append(f"nav: {nav_compact}")

    type_meta = response.get("_type_metadata")
    if type_meta:
        parts.append(f"type: {_sanitize_markdown(str(type_meta.get('type', '?')))}")
        next_steps = type_meta.get("next_steps")
        if next_steps:
            safe_steps = [_sanitize_markdown(str(s)) for s in next_steps[:2]]
            parts.append(f"explore: {'; '.join(safe_steps)}")

    warning = response.get("_format_warning")
    if warning:
        parts.append(f"warning: {_sanitize_markdown(str(warning))}")

    return " | ".join(parts)


def render_text(response: dict[str, Any], fmt: str = DEFAULT_FORMAT) -> str:
    """Render a call_endpoint response as LLM-readable markdown.

    Dispatches based on response shape:
    - dict results (keyed batch) → per-source ## sections
    - list results → markdown-KV (≤20 items) or markdown-table (>20)
    - empty/error/non-list → compact JSON fallback
    - fmt="json" → compact JSON (internal _-prefixed fields stripped)

    Appends a compact metadata footer with pagination, navigation hints,
    and type metadata when present.

    Parameters
    ----------
    response : dict
        Response dict from call_endpoint. "results" is expected to be
        list[dict] (single query) or dict[str, list] (keyed batch).
    fmt : str
        Only "json" or "markdown" (default) are meaningful. Any other
        value is treated as markdown.

    Returns
    -------
    str
        Markdown for LLM consumption, or compact JSON.
    """
    if fmt == "json":
        # Strip internal fields before JSON serialization
        clean = {k: v for k, v in response.items() if not k.startswith("_")}
        return json.dumps(clean, separators=(",", ":"), default=str)

    results = response.get("results")

    # Keyed batch results: dict of {source: [rows]} — render as per-source sections
    if isinstance(results, dict) and results:
        body = _render_keyed_batch(results)
        footer = _render_footer(response)
        if footer:
            return f"{body}\n---\n{footer}"
        return body

    if not isinstance(results, list) or not results:
        # Non-list/non-dict or empty — always JSON fallback
        return json.dumps(response, separators=(",", ":"), default=str)

    columns = response.get("_columns") or stable_key_union(results)
    if not columns:
        return json.dumps(response, separators=(",", ":"), default=str)

    body = _render_rows(results, columns)

    footer = _render_footer(response)
    if footer:
        return f"{body}\n---\n{footer}"
    return body


def _render_rows(
    results: list[dict[str, Any]],
    columns: list[str],
) -> str:
    """Render a list of rows as markdown-KV (small) or markdown-table (large)."""
    if len(results) <= MARKDOWN_KV_THRESHOLD:
        return to_markdown_kv(results, columns)
    return to_markdown_table(results, columns)


def _render_keyed_batch(results: dict[str, Any]) -> str:
    """Render a keyed batch result (dict of source -> rows) as sectioned markdown.

    Each source becomes a ## heading; its row list is rendered below.
    Empty sources are marked.
    """
    sections = []
    for source, rows in results.items():
        safe_source = _sanitize_markdown(str(source))
        if not isinstance(rows, list) or not rows:
            sections.append(f"## {safe_source}\n_(no results)_")
            continue

        # Derive columns per-section (each source may have different shape)
        cols = stable_key_union(rows)
        if not cols:
            sections.append(f"## {safe_source}\n{_sanitize_markdown(str(rows))}")
            continue

        body = _render_rows(rows, cols)
        sections.append(f"## {safe_source}\n{body}")

    return "\n\n".join(sections)

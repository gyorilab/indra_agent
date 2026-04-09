# LLM-Native Output Formats

**Status**: Shipped
**Branch**: `feat/llm-native-output-formats`

## Problem

The MCP server originally returned `call_endpoint` results as compact JSON strings via `json.dumps(obj, separators=(',', ':'))`. Every response was a list of dicts where each item repeated all key names, wrapped in braces and escaped quotes. A 50-item × 6-key response produced hundreds of tokens of pure syntactic overhead — tokens the LLM has to process but that carry no semantic content.

2025 research (Chroma's context-rot study, Anthropic's context engineering guide, improvingagents.com format benchmarks) consistently shows that compact JSON is a poor format for LLM comprehension — not just for token efficiency but for *accuracy*. The question isn't "how do we fit more in the context" but "how does the model best understand what's there."

## Research Basis

[improvingagents.com benchmark (Sep 2025)](https://www.improvingagents.com/blog/best-input-data-format-for-llms/) tested 11 formats on 1000-record retrieval tasks with GPT-4.1-nano:

| Format | Accuracy | Tokens |
|---|---|---|
| **Markdown-KV** | **60.7%** | 52K |
| XML | 56.0% | 76K |
| INI | 55.7% | 48K |
| YAML | 54.7% | 55K |
| HTML | 53.6% | 75K |
| JSON | 52.3% | 66K |
| **Markdown table** | **51.9%** | **25K** |
| Natural language | 49.6% | 43K |
| JSONL | 45.0% | 54K |
| CSV | 44.3% | 20K |
| Pipe-delimited | 41.1% | 43K |

- **Markdown-KV** (per-entity headings with `key: value` pairs) has the **highest comprehension** — 8 points above JSON, 16 above CSV.
- **Markdown tables** have the best accuracy/token ratio — 51.9% at only 25K tokens.
- Compact JSON ranks middle-of-pack for comprehension and poorly for tokens.

[Anthropic's context engineering guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) reinforces: "the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome." Every unnecessary token in a tool response depletes the attention budget.

## Architecture

### No `format` parameter

Agents shouldn't have to think about output format. The tool always returns markdown, and the renderer auto-picks the variant based on result count. Zero cognitive load.

### Rendering strategy (`formats.py:render_text`)

```
response dict
  ↓
  ├─ dict results (keyed batch) → per-source ## sections → each rendered as below
  ├─ list results + ≤20 items → markdown-KV (per-entity ### headings)
  ├─ list results + >20 items → markdown-table (pipe-delimited)
  ├─ empty / non-list / error → compact JSON fallback
  └─ fmt="json" (internal/test) → compact JSON (strips _-prefixed fields)
```

All output ends with a compact metadata footer:
```
---
10 of 47 | next: offset=10 | nav: Gene->Disease, Gene->Pathway | type: gene | explore: Find diseases; Find pathways
```

### Column schema stability

Columns are derived **once over the full cached list** (not per-page), so the schema is stable across pagination offsets. The stable column set is attached to the response dict as `_columns` (internal field, stripped from JSON mode output). Fix: when `fields` projection is applied, column set is the projected fields directly.

### Markdown injection hardening

`_format_value` and `_sanitize_markdown` neutralize hostile data in entity names/descriptions:
- Newlines / carriage returns → single space (prevents multi-line injection)
- Leading `#`, `-`, `*`, `|` → escaped (prevents fake headings/lists/cells)
- Literal `---` sequences → `- - -` (prevents fake footer separators)
- Triple backticks → escaped (prevents fake code blocks)

Sanitization is applied at the **value layer** (`_format_value`) and at the **heading/column layer** (explicit `_sanitize_markdown` calls in `to_markdown_kv` and `to_markdown_table`). Tests cover prompt-injection scenarios, fake-heading forging, fake-footer forging, and code-block injection.

## FastMCP `structured_output=False`

[Claude Code v2.0.21+ prioritizes `structuredContent` over `TextContent`](https://github.com/anthropics/claude-code/issues/9962) — when FastMCP auto-generates both (which it does for dict returns), Claude Code displays the structured JSON as `{"result":"..."}` with escaped `\n` sequences instead of rendering the cleanly-formatted text.

**Fix**: every `@mcp.tool` decorator now sets `structured_output=False`, disabling FastMCP's auto-generation. Only `TextContent` is sent. Claude Code then renders the markdown with real newlines.

## What shipped

**New modules:**
- `src/indra_agent/mcp_server/formats.py` — sanitizer, `to_markdown_kv`, `to_markdown_table`, `render_text`, keyed batch rendering
- `tests/mcp_server/test_formats.py` — unit and integration tests

**Modified:**
- `autoclient_tools.py` — `call_endpoint` returns format-agnostic response with `_columns`; tool wrappers call `render_text(result)`; all decorators have `structured_output=False`; defensive copy before presentation transforms; enrichment fix (uses public `build_type_metadata`)
- `server.py` — all decorators have `structured_output=False`
- `enrichment.py` — added public `build_type_metadata()` wrapper
- `pagination.py` — unchanged (token estimation remains JSON-based; markdown output may be slightly larger than the estimate but stays under MCP 25k limit — documented in tool docstring)

**Preexisting bugs fixed as part of this work:**
- **Enrichment double-pagination** (previously `call_endpoint` → `enrich_results` → internal paginate → outer paginate, dropping `_type_metadata`). Now `call_endpoint` calls `build_type_metadata` directly, no double-pagination, `_type_metadata` preserved in envelope.
- **`sort_by` mutation race** on shared in-flight/cached results. Defensive shallow copy before any presentation transform.

## Example Output

What the model sees for `call_endpoint("get_targets_for_drug", {"drug": "imatinib"})`:

```
### PDGFRB
- db_ns: HGNC
- db_id: 8804

### PDGFRA
- db_ns: HGNC
- db_id: 8803

### ABL1
- db_ns: HGNC
- db_id: 76
---
5 of 22 | next: offset=5 | nav: Gene->Disease, Gene->Pathway | type: gene | explore: Find diseases: Query gene-disease associations; Find pathways: Query gene-pathway memberships
```

No braces. No `{":":,}` syntax tax. No escaped delimiters. Real markdown that Claude Code renders with real newlines.

"""Single gateway for autoclient endpoints with graph navigation.

Progressive disclosure pattern for agent interaction:
1. ground_entity: Natural language → CURIEs (GILDA)
2. suggest_endpoints: "You have Gene entities. You can explore: Disease, Pathway..."
3. call_endpoint: Execute function with auto-grounding

Graph navigation approach:
- Functions like get_diseases_for_gene define edges: Gene → Disease
- Agents navigate the knowledge graph by following these edges
- suggest_endpoints shows WHERE you can go, not just WHAT you can call
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from indra_agent.mcp_server.cache import cache as _cache, make_key, DEFAULT_TTL


def compact_json(obj: Any) -> str:
    """Serialize to compact JSON for token efficiency."""
    return json.dumps(obj, separators=(',', ':'), default=str)


from indra_agent.mcp_server.mappings import (
    CURIE_PREFIX_TO_ENTITY,
    PARAM_NAMESPACE_FILTERS,
    MIN_CONFIDENCE_THRESHOLD,
    AMBIGUITY_SCORE_THRESHOLD,
    ORGANISM_TO_TAXONOMY_ID,
)
from indra_agent.mcp_server.registry import _get_registry, clear_registry_cache
from indra_agent.mcp_server.serialization import process_result, resolve_entity_names
from indra_agent.mcp_server.pagination import paginate_response, estimate_tokens

logger = logging.getLogger(__name__)

# Request coalescing: prevent duplicate Neo4j queries for identical concurrent requests.
# Maps cache key → asyncio.Future so concurrent callers await a single execution.
_inflight: Dict[str, asyncio.Future] = {}

# Map single-entity functions → native batch variants.
# These use WHERE IN (single Neo4j query) instead of fan-out.
# Values: (batch_function_name, batch_param_name)
# Explicit param names avoid naive pluralization ("target"→"targets" is safe,
# but e.g. "analysis"→"analysiss" would break).
BATCH_FUNCTION_MAP: Dict[str, Tuple[str, str]] = {
    "get_drugs_for_target": ("get_drugs_for_targets", "targets"),
    "get_targets_for_drug": ("get_targets_for_drugs", "drugs"),
    # get_shared_pathways_for_genes has different semantics (intersection, not union)
    # so we do NOT map get_pathways_for_gene → get_shared_pathways_for_genes
}


def _detect_entity_types(entity_ids: List[str]) -> Set[str]:
    """Detect entity types from CURIE prefixes."""
    detected = set()
    for eid in entity_ids:
        eid_lower = eid.lower()
        for prefix, entity_type in CURIE_PREFIX_TO_ENTITY.items():
            if eid_lower.startswith(prefix + ':') or eid_lower.startswith(prefix + '_'):
                detected.add(entity_type)
                break
    return detected


def _normalize_organism(organism: Optional[str]) -> Optional[str]:
    """Normalize organism name to NCBI taxonomy ID for gilda.

    Resolution layers:
    1. Passthrough if already a numeric taxonomy ID (e.g., "9606")
    2. Local lookup from gilda's organism_labels + common English names
    3. NCBI Entrez Taxonomy API fallback for arbitrary names

    Gilda expects taxonomy IDs like '9606', not names like 'human'.
    Passing unrecognized names silently drops all organism-specific results.
    """
    if organism is None:
        return None
    stripped = organism.strip()
    # Already a taxonomy ID (e.g., "9606")
    if stripped.isdigit():
        return stripped
    # Layer 1+2: Local lookup (gilda canonical + common English names)
    taxonomy_id = ORGANISM_TO_TAXONOMY_ID.get(stripped.lower())
    if taxonomy_id:
        return taxonomy_id
    # Layer 3: NCBI Entrez Taxonomy API fallback
    try:
        from indra.databases.taxonomy_client import get_taxonomy_id
        taxonomy_id = get_taxonomy_id(stripped)
        if taxonomy_id:
            logger.info("Resolved organism '%s' → %s via NCBI Entrez", stripped, taxonomy_id)
            return taxonomy_id
    except Exception as e:
        logger.debug("NCBI taxonomy lookup failed for '%s': %s", stripped, e)
    logger.warning(
        "Unknown organism '%s' — not a recognized name or taxonomy ID. "
        "Passing None to gilda (defaults to human/9606). "
        "Valid examples: 'human', '9606', 'mouse', '10090'",
        organism,
    )
    return None


def _lookup_xrefs(
    namespace: str,
    identifier: str,
    client,
    param_name: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Look up equivalent identifiers via xref relationships in the graph.

    Parameters
    ----------
    namespace : str
        Source namespace (e.g., "mesh", "doid")
    identifier : str
        Source identifier
    client
        Neo4j client
    param_name : str, optional
        Parameter name for filtering (e.g., "disease" filters to disease namespaces)

    Returns
    -------
    :
        List of (namespace, id) tuples including the original plus any xrefs found
    """
    from indra_cogex.representation import norm_id

    # Start with the original identifier
    equivalents = [(namespace.lower(), identifier)]

    # Get allowed namespaces for this parameter type
    allowed_namespaces = None
    if param_name:
        allowed_namespaces = PARAM_NAMESPACE_FILTERS.get(param_name.lower())

    try:
        # Query for xref relationships (bidirectional)
        source_id = norm_id(namespace, identifier)
        query = """
            MATCH (source:BioEntity {id: $source_id})-[:xref]-(target:BioEntity)
            RETURN target.id AS target_id
            LIMIT 20
        """

        results = client.query_tx(query, source_id=source_id, squeeze=True)

        for target_id in results:
            if ":" in target_id:
                target_ns, target_identifier = target_id.split(":", 1)
                target_ns_lower = target_ns.lower()

                # Filter by allowed namespaces if specified
                if allowed_namespaces and target_ns_lower not in allowed_namespaces:
                    continue

                equivalents.append((target_ns_lower, target_identifier))

        logger.debug(f"Found {len(equivalents)-1} xrefs for {namespace}:{identifier}")

    except Exception as e:
        logger.warning(f"xref lookup failed for {namespace}:{identifier}: {e}")

    return equivalents


def suggest_endpoints(
    entity_ids: List[str],
    intent: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Suggest navigation options from current entities.

    This is GRAPH NAVIGATION, not search. Given entities you have,
    shows where you can go next in the knowledge graph.

    Parameters
    ----------
    entity_ids : List[str]
        CURIEs from previous results (e.g., ["HGNC:6407", "MESH:D010300"])
    intent : str, optional
        Exploration goal (e.g., "find drug targets", "understand mechanisms")
    top_k : int
        Max suggestions per source entity type

    Returns
    -------
    :
        Dict with source_entities, navigation_options, total_sources,
        intent, and hint fields
    """
    registry, func_mapping, edge_map = _get_registry()

    if not registry:
        return {"error": "Function registry not available", "navigation_options": []}

    # Detect what entity types we have
    source_types = _detect_entity_types(entity_ids)

    if not source_types:
        return {
            "error": "Could not detect entity types from IDs. Use CURIE format (e.g., HGNC:6407)",
            "hint": "Ground natural language terms first using ground_entity",
            "entity_ids_received": entity_ids[:5],
        }

    # Build navigation options grouped by source → target
    navigation = []
    for source_type in sorted(source_types):
        if source_type not in edge_map:
            continue

        source_nav = {
            "from": source_type,
            "can_reach": [],
        }

        for target_type, functions in sorted(edge_map[source_type].items()):
            # Apply intent filtering if provided
            relevant_funcs = functions
            if intent:
                intent_lower = intent.lower()
                # Boost functions matching intent keywords
                scored = []
                for f in functions:
                    score = 1
                    if any(kw in f.lower() for kw in intent_lower.split()):
                        score = 2
                    scored.append((score, f))
                scored.sort(reverse=True)
                relevant_funcs = [f for _, f in scored]

            # Return just function names — params are inferrable from name pattern
            source_nav["can_reach"].append({
                "target": target_type,
                "functions": list(relevant_funcs[:3]),
            })

        if source_nav["can_reach"]:
            navigation.append(source_nav)

    return {
        "source_entities": list(source_types),
        "navigation_options": navigation,
        "total_sources": len(source_types),
        "intent": intent,
        "hint": "Use call_endpoint with one of the suggested functions. "
                "Pass entity as [namespace, id] tuple, e.g., gene=[\"HGNC\", \"6407\"]",
    }


async def call_endpoint(
    endpoint: str,
    kwargs: str,
    get_client_func: Callable,
    auto_ground: bool = True,
    disclosure_level: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    include_navigation: bool = False,
    fields: Optional[List[str]] = None,
    estimate: bool = False,
    sort_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Call an autoclient endpoint with optional auto-grounding and enrichment.

    If auto_ground=True and a string is passed where a CURIE tuple is expected,
    automatically ground via GILDA.

    Parameters
    ----------
    endpoint : str
        Function name (e.g., "get_diseases_for_gene")
    kwargs : str
        JSON string of arguments. Entity args can be:
        - CURIE tuple: ["HGNC", "6407"]
        - Natural language: "LRRK2" (auto-grounded if auto_ground=True)
    get_client_func : Callable
        Function to get Neo4j client
    auto_ground : bool
        Whether to auto-ground string params to CURIEs (default: True)
    disclosure_level : str, optional
        Enrich results with metadata. Options:
        - None (default): Raw results only (most token-efficient)
        - "minimal": Pass-through (same as None)
        - "standard": Add descriptions + next steps (~250 tokens/item)
        - "detailed": Add provenance (~400 tokens/item)
        - "exploratory": Add workflows + research context (~750 tokens/item)
    offset : int
        Starting offset for pagination (default: 0)
    limit : int, optional
        Maximum items to return per page. If response exceeds ~20k tokens,
        it will be automatically truncated with has_more=True.
    sort_by : str, optional
        Sort results before pagination. Options:
        - "evidence": Descending by source_counts sum (most-validated first)
        - "name": Alphabetical by entity name
        Default: None (preserve query order).

    Returns
    -------
    :
        Dict with endpoint, parameters, results, result_count, pagination,
        and optionally grounding_applied, enrichment, or error fields.
        If has_more=True in pagination, call again with next_offset to continue.
    """
    registry, func_mapping, _ = _get_registry()

    if endpoint not in func_mapping:
        return {
            "error": f"Unknown endpoint: {endpoint}",
            "hint": "Use suggest_endpoints to find available functions",
            "available": list(registry.keys())[:20],
        }

    # Parse kwargs
    try:
        parsed_kwargs = json.loads(kwargs) if isinstance(kwargs, str) else kwargs
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "kwargs_received": kwargs}

    func = func_mapping[endpoint]
    grounding_info = {}

    # Process parameters - normalize CURIEs and auto-ground strings
    from inspect import signature
    from typing import get_args, get_origin

    try:
        func_sig = signature(func)
        for param_name, param_value in list(parsed_kwargs.items()):
            if param_name not in func_sig.parameters:
                continue

            param_type = func_sig.parameters[param_name].annotation

            # Check if parameter expects Tuple[str, str] (CURIE)
            if get_origin(param_type) is tuple and get_args(param_type) == (str, str):
                # Case 1: Already a list/tuple CURIE — pass through as-is
                # Upstream norm_id() handles namespace normalization internally
                if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
                    parsed_kwargs[param_name] = list(param_value)
                    continue

                # Case 2: String - try auto-grounding if enabled
                if auto_ground and isinstance(param_value, str):

                    # Ground with parameter semantics filtering
                    # param_name (disease, gene, drug) filters to appropriate namespaces
                    grounding = await ground_entity(
                        term=param_value,
                        limit=10,
                        param_name=param_name,  # Semantic filtering!
                    )

                    if "error" in grounding:
                        return {
                            "error": f"Could not ground '{param_value}': {grounding['error']}",
                            "parameter": param_name,
                            "hint": "Provide explicit CURIE: [namespace, id]",
                        }

                    if not grounding.get("groundings"):
                        return {
                            "error": f"No grounding found for '{param_value}' as {param_name}",
                            "parameter": param_name,
                            "namespaces_searched": grounding.get("namespaces_allowed"),
                            "hint": "Provide explicit CURIE: [namespace, id]",
                        }

                    top = grounding["top_match"]
                    top_score = top["score"]

                    # DUAL-CHECK AMBIGUITY DETECTION
                    # 1. Absolute threshold: top score must be >= MIN_CONFIDENCE_THRESHOLD
                    if top_score < MIN_CONFIDENCE_THRESHOLD:
                        return {
                            "error": f"Low confidence grounding for '{param_value}'",
                            "parameter": param_name,
                            "top_score": top_score,
                            "threshold": MIN_CONFIDENCE_THRESHOLD,
                            "grounding_options": grounding["groundings"][:5],
                            "hint": "Choose one and provide as [namespace, id]",
                        }

                    # 2. Relative clustering: no result in top 5 within AMBIGUITY_SCORE_THRESHOLD
                    if len(grounding["groundings"]) > 1:
                        top_5 = grounding["groundings"][:5]
                        for other in top_5[1:]:
                            if top_score - other["score"] < AMBIGUITY_SCORE_THRESHOLD:
                                return {
                                    "error": f"Ambiguous term '{param_value}' for {param_name}",
                                    "parameter": param_name,
                                    "reason": f"Multiple results within {AMBIGUITY_SCORE_THRESHOLD} of top score",
                                    "grounding_options": top_5,
                                    "hint": "Choose one and provide as [namespace, id]",
                                }

                    # Apply grounding - unambiguous match!
                    # Uppercase namespace: upstream functions validate against
                    # uppercase prefixes (e.g., mesh_term[0] != "MESH") before
                    # calling norm_id(). Gilda returns lowercase, graph expects uppercase.
                    parsed_kwargs[param_name] = [top["namespace"].upper(), top["identifier"]]
                    grounding_info[param_name] = {
                        "input": param_value,
                        "grounded_to": top,
                        "method": "gilda",
                        "param_filter": param_name,
                    }

            # Check if parameter expects List[Tuple[str, str]] (list of CURIEs)
            # e.g., get_shared_pathways_for_genes(genes: List[Tuple[str, str]])
            elif (get_origin(param_type) is list
                  and get_args(param_type)
                  and get_origin(get_args(param_type)[0]) is tuple
                  and get_args(get_args(param_type)[0]) == (str, str)):

                if not isinstance(param_value, list):
                    continue

                # Separate pre-formed CURIEs from strings needing grounding
                grounded_list = []
                strings_to_ground = []
                string_indices = []  # Track positions for reinsertion

                for i, item in enumerate(param_value):
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        # Already a CURIE — uppercase namespace for upstream
                        grounded_list.append((i, [str(item[0]).upper(), str(item[1])]))
                    elif auto_ground and isinstance(item, str):
                        strings_to_ground.append(item)
                        string_indices.append(i)
                    else:
                        return {
                            "error": f"Invalid item in {param_name}[{i}]: expected [namespace, id] or string",
                            "parameter": param_name,
                            "item": item,
                            "hint": "Each item should be a CURIE [namespace, id] or a string to auto-ground",
                        }

                # Batch-ground all strings in one call
                if strings_to_ground:
                    # Infer entity type from param_name (e.g., "genes" → "gene")
                    entity_filter = param_name.rstrip("s") if param_name.endswith("s") else param_name
                    batch_result = await ground_entity(
                        terms=strings_to_ground,
                        param_name=entity_filter,
                    )

                    if "error" in batch_result:
                        return {
                            "error": f"Batch grounding failed for {param_name}: {batch_result['error']}",
                            "parameter": param_name,
                        }

                    mappings = batch_result.get("mappings", {})
                    failed_terms = batch_result.get("failed", [])

                    if failed_terms:
                        return {
                            "error": f"Could not ground {len(failed_terms)} of {len(strings_to_ground)} terms for {param_name}",
                            "parameter": param_name,
                            "failed_terms": failed_terms,
                            "hint": "Provide explicit CURIEs: [[namespace, id], ...]",
                        }

                    for idx, term in zip(string_indices, strings_to_ground):
                        mapping = mappings.get(term)
                        if not mapping:
                            return {
                                "error": f"No grounding found for '{term}' in {param_name}",
                                "parameter": param_name,
                                "hint": f"Provide explicit CURIE for '{term}': [namespace, id]",
                            }
                        ns, id_ = mapping["curie"].split(":", 1)
                        grounded_list.append((idx, [ns.upper(), id_]))

                    grounding_info[param_name] = {
                        "input": strings_to_ground,
                        "grounded_count": len(mappings),
                        "method": "gilda_batch",
                        "param_filter": entity_filter,
                    }

                # Sort by original index and extract just the CURIEs
                grounded_list.sort(key=lambda x: x[0])
                parsed_kwargs[param_name] = [curie for _, curie in grounded_list]

    except Exception as e:
        logger.warning(f"Auto-grounding failed: {e}")
        # Continue without auto-grounding

    # Execute function with xref fallback (or serve from cache)
    try:
        client = get_client_func()

        # Check cache first (same endpoint + kwargs = cache hit)
        result_id = make_key("endpoint", endpoint, parsed_kwargs)
        try:
            cached = await asyncio.to_thread(_cache.get, result_id)
        except Exception:
            logger.debug("Cache read failed for %s, treating as miss", result_id)
            cached = None

        if cached is not None:
            logger.debug(f"Cache hit for {endpoint} (result_id={result_id})")
            processed = cached
        elif result_id in _inflight:
            # Another request is already executing this exact query — coalesce
            logger.debug(f"Coalescing request for {endpoint} (result_id={result_id})")
            processed = await _inflight[result_id]
        else:
            # Cache miss, no in-flight request — we execute
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            _inflight[result_id] = future
            try:
                # Cache miss — execute query
                result = await asyncio.to_thread(func, client=client, **parsed_kwargs)
                processed = process_result(result)

                # Resolve entity names from graph (batch query for all results)
                if isinstance(processed, list) and processed:
                    processed = await asyncio.to_thread(resolve_entity_names, processed, client)

                # CROSS-REFERENCE FALLBACK: If no results and we auto-grounded, try xrefs
                if (isinstance(processed, list) and len(processed) == 0 and grounding_info):
                    logger.info(f"No results with original grounding, trying xrefs...")

                    for param_name, ground_info in grounding_info.items():
                        if param_name not in parsed_kwargs:
                            continue

                        original_curie = parsed_kwargs[param_name]
                        namespace, identifier = original_curie[0], original_curie[1]

                        equivalents = _lookup_xrefs(namespace, identifier, client, param_name)

                        if len(equivalents) > 1:
                            logger.info(f"Trying {len(equivalents)-1} xrefs for {param_name}")
                            ground_info["xrefs_tried"] = []

                            for equiv_ns, equiv_id in equivalents[1:]:
                                test_kwargs = parsed_kwargs.copy()
                                test_kwargs[param_name] = [equiv_ns, equiv_id]

                                try:
                                    xref_result = await asyncio.to_thread(func, client=client, **test_kwargs)
                                    xref_processed = process_result(xref_result)

                                    ground_info["xrefs_tried"].append({
                                        "namespace": equiv_ns,
                                        "identifier": equiv_id,
                                        "result_count": len(xref_processed) if isinstance(xref_processed, list) else 1
                                    })

                                    if isinstance(xref_processed, list) and len(xref_processed) > 0:
                                        logger.info(f"Found {len(xref_processed)} results using xref {equiv_ns}:{equiv_id}")
                                        xref_processed = await asyncio.to_thread(resolve_entity_names, xref_processed, client)
                                        processed = xref_processed
                                        parsed_kwargs[param_name] = [equiv_ns, equiv_id]
                                        ground_info["xref_used"] = {
                                            "namespace": equiv_ns,
                                            "identifier": equiv_id,
                                            "original_namespace": namespace,
                                            "original_identifier": identifier,
                                        }
                                        break
                                except Exception as e:
                                    logger.debug(f"xref {equiv_ns}:{equiv_id} failed: {e}")
                                    ground_info["xrefs_tried"].append({
                                        "namespace": equiv_ns,
                                        "identifier": equiv_id,
                                        "error": str(e)
                                    })

                # Cache results for subsequent paginated/projected access
                if isinstance(processed, list):
                    try:
                        await asyncio.to_thread(lambda: _cache.set(result_id, processed, expire=DEFAULT_TTL, tag="endpoint"))
                    except Exception:
                        logger.debug("Cache write failed for %s", result_id)

                # Resolve the future so coalesced waiters get the result
                future.set_result(processed)
            except Exception as exc:
                # Propagate exception to coalesced waiters
                future.set_exception(exc)
                raise
            finally:
                # Always clean up to prevent memory leaks
                _inflight.pop(result_id, None)

        # Estimate mode: return metadata only, results stay server-side
        if estimate and isinstance(processed, list):
            fields_available = set()
            for item in processed[:10]:
                if isinstance(item, dict):
                    fields_available.update(item.keys())

            return {
                "result_id": result_id,
                "result_count": len(processed),
                "token_estimate_full": estimate_tokens(processed),
                "fields_available": sorted(fields_available),
                "sample": processed[:3],
            }

        # Sort results if requested (before field projection and pagination)
        if sort_by and isinstance(processed, list):
            if sort_by == "evidence":
                def _evidence_key(item):
                    if isinstance(item, dict):
                        sc = item.get("source_counts", {})
                        if isinstance(sc, dict):
                            return sum(sc.values())
                        return item.get("evidence_count", 0)
                    return 0
                processed.sort(key=_evidence_key, reverse=True)
            elif sort_by == "name":
                processed.sort(key=lambda x: (x.get("name", "") or "").lower() if isinstance(x, dict) else "")
            else:
                logger.warning("Unknown sort_by value '%s', ignoring (valid: 'evidence', 'name')", sort_by)

        # Apply field projection if requested (before enrichment, after name resolution)
        if fields and isinstance(processed, list):
            field_set = set(fields)
            processed = [
                {k: v for k, v in item.items() if k in field_set}
                for item in processed
                if isinstance(item, dict)
            ]

        # Optionally apply enrichment if disclosure_level specified
        enrichment_info = None
        if disclosure_level and disclosure_level != "minimal":
            try:
                from indra_agent.mcp_server.enrichment import enrich_results, DisclosureLevel
                if isinstance(processed, list) and processed:
                    enrichment_result = await asyncio.to_thread(
                        enrich_results,
                        results=processed,
                        disclosure_level=DisclosureLevel(disclosure_level),
                        result_type=None,  # Auto-detect from results
                        client=client,
                    )
                    processed = enrichment_result["results"]
                    enrichment_info = {
                        "disclosure_level": disclosure_level,
                        "token_estimate": enrichment_result["token_estimate"],
                    }
            except ValueError as e:
                logger.warning(f"Invalid disclosure_level '{disclosure_level}': {e}")
            except Exception as e:
                logger.warning(f"Enrichment failed: {e}")

        # Generate suggested_next from result entity types (opt-in only)
        suggested_next = None
        if include_navigation and isinstance(processed, list) and processed:
            # Extract entity IDs from results for navigation suggestions
            result_ids = []
            for item in processed[:10]:  # Sample first 10 for efficiency
                if isinstance(item, dict):
                    if "db_ns" in item and "db_id" in item:
                        result_ids.append(f"{item['db_ns']}:{item['db_id']}")
                    elif "id" in item:
                        result_ids.append(item["id"] if isinstance(item["id"], str) else f"{item['id'][0]}:{item['id'][1]}")

            if result_ids:
                nav_suggestions = suggest_endpoints(result_ids, intent=None, top_k=3)
                if "navigation_options" in nav_suggestions and nav_suggestions["navigation_options"]:
                    # Flatten to compact format: [{target, functions}]
                    suggested_next = []
                    for nav in nav_suggestions["navigation_options"]:
                        for reach in nav.get("can_reach", [])[:3]:
                            suggested_next.append({
                                "from": nav["from"],
                                "to": reach["target"],
                                "functions": reach["functions"][:2],
                            })

        # Apply pagination to results
        if isinstance(processed, list):
            paginated = paginate_response(
                processed,
                offset=offset,
                limit=limit,
                include_token_estimate=True,
            )
            final_results = paginated["results"]
            pagination_info = paginated["pagination"]
        else:
            final_results = processed
            pagination_info = {
                "total": 1,
                "offset": 0,
                "returned": 1,
                "has_more": False,
                "token_estimate": estimate_tokens(processed),
            }

        # Build slim response — don't echo endpoint/parameters (agent already knows)
        response = {
            "results": final_results,
        }

        # Collapse pagination when trivial (single page)
        if pagination_info.get("has_more"):
            response["pagination"] = pagination_info
        else:
            response["total"] = pagination_info.get("total", len(final_results) if isinstance(final_results, list) else 1)

        if suggested_next:
            response["suggested_next"] = suggested_next

        # Only report grounding when xref fallback was used (unexpected resolution)
        if grounding_info:
            xref_fallbacks = {k: v for k, v in grounding_info.items() if v.get("xref_used")}
            if xref_fallbacks:
                response["xref_fallback"] = xref_fallbacks

        if enrichment_info:
            response["enrichment"] = enrichment_info

        return response

    except Exception as e:
        logger.error(f"Error executing {endpoint}: {e}", exc_info=True)
        return {
            "endpoint": endpoint,
            "error": str(e),
            "parameters": parsed_kwargs,
        }


async def batch_call(
    endpoint: str,
    entity_param: str,
    entity_values: List[str],
    common_kwargs: str = "{}",
    get_client_func: Callable = None,
    auto_ground: bool = True,
    fields: Optional[List[str]] = None,
    max_concurrent: int = 10,
    merge_strategy: str = "keyed",
) -> Dict[str, Any]:
    """Execute an endpoint for multiple entities in parallel.

    Replaces N serial MCP round-trips with 1. Internally fans out
    via asyncio.gather with bounded concurrency, or routes to native
    batch variants where available (single Neo4j WHERE IN query).

    Parameters
    ----------
    endpoint : str
        Function name (e.g., "get_diseases_for_gene")
    entity_param : str
        Which parameter to batch over (e.g., "gene")
    entity_values : list of str
        Entity values to query (e.g., ["SIRT3", "PRKN", "MAPT"])
    common_kwargs : str
        JSON string of additional shared parameters (default: "{}")
    get_client_func : Callable
        Function to get Neo4j client
    auto_ground : bool
        Auto-ground entity strings to CURIEs (default: True)
    fields : list of str, optional
        Project results to only these keys per item
    max_concurrent : int
        Max parallel queries (default: 10, prevents Neo4j overload)
    merge_strategy : str
        "keyed" (default): {entity: results} dict.
        "flat": All results concatenated into one list.

    Returns
    -------
    :
        Dict with results, total_entities, successful, total_results,
        and optionally failed dict for partial failures.
    """
    if not entity_values:
        return {"error": "entity_values must be a non-empty list"}

    # Parse common kwargs once
    parsed_common = json.loads(common_kwargs) if isinstance(common_kwargs, str) else common_kwargs

    # Check for native batch route
    registry, func_mapping, _ = _get_registry()
    batch_info = BATCH_FUNCTION_MAP.get(endpoint)

    if batch_info:
        native_batch, batch_param_name = batch_info
        if native_batch in func_mapping:
            # NATIVE BATCH PATH: single Neo4j WHERE IN query
            return await _batch_call_native(
                native_batch, batch_param_name, entity_param, entity_values,
                parsed_common, get_client_func, auto_ground, fields,
            )

    # FAN-OUT PATH: N parallel call_endpoint invocations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_one(entity_value: str) -> Tuple[str, Dict]:
        async with semaphore:
            kwargs = {entity_param: entity_value, **parsed_common}
            result = await call_endpoint(
                endpoint=endpoint,
                kwargs=json.dumps(kwargs),
                get_client_func=get_client_func,
                auto_ground=auto_ground,
                fields=fields,
            )
            return entity_value, result

    pairs = await asyncio.gather(
        *[_call_one(v) for v in entity_values],
        return_exceptions=True,
    )

    # Merge results
    results: Dict[str, Any] = {}
    failed: Dict[str, Any] = {}
    total_results = 0

    for i, pair in enumerate(pairs):
        if isinstance(pair, Exception):
            entity_val = entity_values[i] if i < len(entity_values) else f"_exception_{i}"
            failed[entity_val] = {"error": str(pair)}
            continue
        entity_value, result = pair
        if "error" in result:
            failed[entity_value] = {"error": result["error"]}
        else:
            items = result.get("results", result)
            if merge_strategy == "keyed":
                results[entity_value] = items
                total_results += len(items) if isinstance(items, list) else 1
            else:
                # Flat: extend list
                if isinstance(items, list):
                    results.setdefault("_flat", []).extend(items)
                    total_results += len(items)
                else:
                    results.setdefault("_flat", []).append(items)
                    total_results += 1

    response: Dict[str, Any] = {
        "results": results.get("_flat", []) if merge_strategy == "flat" else results,
        "total_entities": len(entity_values),
        "successful": len(entity_values) - len(failed),
        "total_results": total_results,
    }
    if failed:
        response["failed"] = failed

    return response


async def _batch_call_native(
    batch_endpoint: str,
    batch_param_name: str,
    entity_param: str,
    entity_values: List[str],
    common_kwargs: Dict,
    get_client_func: Callable,
    auto_ground: bool,
    fields: Optional[List[str]],
) -> Dict[str, Any]:
    """Route to native batch variant (single WHERE IN query).

    Native batch functions like get_drugs_for_targets accept
    List[Tuple[str,str]] instead of Tuple[str,str]. We ground all
    entities, then call the batch function once.

    Parameters
    ----------
    batch_endpoint : str
        Native batch function name (e.g., "get_drugs_for_targets")
    batch_param_name : str
        Parameter name for the batch function (e.g., "targets")
    entity_param : str
        Original single-entity parameter name (e.g., "target")
    entity_values : list of str
        Entity values to ground and batch
    common_kwargs : dict
        Additional shared parameters
    get_client_func : Callable
        Function to get Neo4j client
    auto_ground : bool
        Whether to auto-ground entity strings to CURIEs
    fields : list of str, optional
        Project results to only these keys per item

    Returns
    -------
    :
        Dict with results, total_entities, successful, total_results,
        batch_mode="native", and optionally failed dict.
    """
    grounded_entities: List[Tuple[str, List[str]]] = []
    failed: Dict[str, Any] = {}

    if auto_ground:
        grounding_result = await ground_entity(
            terms=entity_values,
            param_name=entity_param,
        )
        for entity_value in entity_values:
            mapping = grounding_result.get("mappings", {}).get(entity_value)
            if mapping:
                curie = mapping["curie"]
                ns, id_ = curie.split(":", 1)
                # Uppercase namespace: upstream batch functions may validate
                # against uppercase prefixes before calling norm_id()
                grounded_entities.append((entity_value, [ns.upper(), id_]))
            else:
                # Propagate specific error from grounding result
                fail_info = grounding_result.get("failed", {}).get(entity_value)
                if isinstance(fail_info, dict):
                    failed[entity_value] = fail_info
                else:
                    failed[entity_value] = {"error": f"Grounding failed for '{entity_value}'"}
    else:
        # Assume already CURIEs in "NS:ID" format
        for entity_value in entity_values:
            if ":" in entity_value:
                ns, id_ = entity_value.split(":", 1)
                grounded_entities.append((entity_value, [ns, id_]))
            else:
                failed[entity_value] = {"error": "Expected CURIE format NS:ID"}

    if not grounded_entities:
        return {"error": "All entities failed grounding", "failed": failed}

    # Build batch parameter: List[Tuple[str,str]]
    batch_kwargs = {
        batch_param_name: [tuple(curie) for _, curie in grounded_entities],
        **common_kwargs,
    }

    # Execute native batch function directly
    registry, func_mapping, _ = _get_registry()
    batch_func = func_mapping[batch_endpoint]
    client = get_client_func()

    try:
        result = await asyncio.to_thread(batch_func, client=client, **batch_kwargs)
        processed = process_result(result)

        # Native batch functions return Mapping[str, Iterable[Agent]] — a dict
        # keyed by entity string. Flatten to list for consistent downstream handling,
        # or resolve names on each value list if dict.
        if isinstance(processed, dict):
            # Each value is a list of items for that entity
            all_items = []
            for entity_key, items in processed.items():
                if isinstance(items, list):
                    all_items.extend(items)
                else:
                    all_items.append(items)
            processed = all_items

        # Resolve entity names from graph
        if isinstance(processed, list) and processed:
            processed = await asyncio.to_thread(resolve_entity_names, processed, client)

        # Apply field projection if requested
        if fields and isinstance(processed, list):
            field_set = set(fields)
            processed = [
                {k: v for k, v in item.items() if k in field_set}
                for item in processed if isinstance(item, dict)
            ]

        response: Dict[str, Any] = {
            "results": processed,
            "total_entities": len(entity_values),
            "successful": len(grounded_entities),
            "total_results": len(processed) if isinstance(processed, list) else 1,
            "batch_mode": "native",
        }
        if failed:
            response["failed"] = failed
        return response

    except Exception as e:
        logger.error(f"Native batch call failed for {batch_endpoint}: {e}", exc_info=True)
        return {
            "error": f"Native batch call failed: {e}",
            "batch_endpoint": batch_endpoint,
            "failed": failed,
        }


async def ground_entity(
    term: Optional[str] = None,
    terms: Optional[List[str]] = None,
    organism: Optional[str] = None,
    limit: int = 10,
    param_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Ground natural language term(s) to CURIEs using GILDA.

    Supports both single-term and batch modes. Batch mode iterates internally,
    replacing N MCP round-trips with 1.

    Parameters
    ----------
    term : str, optional
        Single natural language term (e.g., "LRRK2", "Parkinson's disease")
    terms : list of str, optional
        Batch mode: list of terms to ground. Returns compact mappings dict.
    organism : str, optional
        Organism context. Accepts common names ("human", "mouse") or NCBI
        taxonomy IDs ("9606", "10090"). Names are resolved internally.
        Default when omitted: human (9606).
    limit : int
        Max results before filtering (default: 10)
    param_name : str, optional
        Parameter name for semantic filtering (e.g., "disease", "gene", "drug")

    Returns
    -------
    :
        Single mode: Dict with query, groundings, top_match, param_filter,
        namespaces_allowed. Batch mode: Dict with mappings, failed, param_filter.
    """
    if not term and not terms:
        return {"error": "Provide either 'term' (single) or 'terms' (batch)"}

    # Batch mode
    if terms:
        return await _ground_entity_batch(terms, organism, limit, param_name)

    # Single-term mode (original behavior)
    return await _ground_entity_single(term, organism, limit, param_name)


async def _ground_entity_single(
    term: str,
    organism: Optional[str] = None,
    limit: int = 10,
    param_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Ground a single term to CURIEs."""
    try:
        from indra_cogex.apps.search.search import gilda_ground

        # F1 fix: normalize organism name → taxonomy ID before passing to gilda
        resolved_organism = _normalize_organism(organism)

        # F5/F6 fix: run blocking gilda_ground in a thread to avoid blocking the event loop
        raw_results = await asyncio.to_thread(
            gilda_ground,
            term,
            organisms=[resolved_organism] if resolved_organism else None,
            limit=limit,
            normalize_curies=True,
        )

        if isinstance(raw_results, dict) and "error" in raw_results:
            return {"error": raw_results["error"], "query": term}

        allowed_namespaces = None
        if param_name:
            allowed_namespaces = PARAM_NAMESPACE_FILTERS.get(param_name.lower())

        raw_count = len(raw_results) if isinstance(raw_results, list) else 0
        groundings = []
        for result in raw_results:
            term_info = result.get("term", {})
            namespace = term_info.get("db", "").lower()

            if allowed_namespaces and namespace not in allowed_namespaces:
                continue

            groundings.append({
                "curie": f"{term_info.get('db', '')}:{term_info.get('id', '')}",
                "namespace": term_info.get("db", ""),
                "identifier": term_info.get("id", ""),
                "name": term_info.get("entry_name", ""),
                "score": round(result.get("score", 0), 3),
                "source": term_info.get("source", ""),
            })

        response = {
            "query": term,
            "groundings": groundings,
            "top_match": groundings[0] if groundings else None,
            "param_filter": param_name,
            "namespaces_allowed": list(allowed_namespaces) if allowed_namespaces else None,
        }

        # F3 fix: add diagnostics when grounding returns no results
        if not groundings and raw_count > 0:
            response["diagnostics"] = {
                "raw_matches": raw_count,
                "filtered_out": raw_count,
                "reason": f"All {raw_count} gilda matches excluded by namespace filter for '{param_name}'",
            }
        elif not groundings and raw_count == 0:
            diag = {"raw_matches": 0, "reason": "No gilda matches for this term"}
            if resolved_organism:
                diag["organism_resolved"] = resolved_organism
            if organism and organism != resolved_organism:
                diag["organism_input"] = organism
            response["diagnostics"] = diag

        return response

    except ImportError as e:
        return {
            "error": f"Grounding dependencies not available: {e}",
            "hint": "Install gilda: pip install gilda",
        }
    except Exception as e:
        return {"error": f"Grounding failed: {e}", "query": term}


async def _ground_entity_batch(
    terms: List[str],
    organism: Optional[str] = None,
    limit: int = 10,
    param_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Ground multiple terms, returning compact mappings.

    Uses concurrent asyncio.gather for throughput — 1 MCP call instead of N,
    with bounded concurrency via semaphore.
    """
    try:
        from indra_cogex.apps.search.search import gilda_ground

        # F1 fix: normalize organism name → taxonomy ID
        resolved_organism = _normalize_organism(organism)

        allowed_namespaces = None
        if param_name:
            allowed_namespaces = PARAM_NAMESPACE_FILTERS.get(param_name.lower())

        # Gilda lookups are fast (~5ms each, read-only trie), higher concurrency OK
        semaphore = asyncio.Semaphore(20)

        async def _ground_one(term: str) -> Tuple[str, Optional[Dict], Optional[str]]:
            """Ground a single term, returning (term, mapping_or_None, error_or_None)."""
            async with semaphore:
                try:
                    # F5/F6 fix: run blocking gilda_ground in a thread
                    raw_results = await asyncio.to_thread(
                        gilda_ground,
                        term,
                        organisms=[resolved_organism] if resolved_organism else None,
                        limit=limit,
                        normalize_curies=True,
                    )

                    if isinstance(raw_results, dict) and "error" in raw_results:
                        return (term, None, raw_results["error"])

                    # Find best match after namespace filtering
                    best = None
                    for result in raw_results:
                        term_info = result.get("term", {})
                        namespace = term_info.get("db", "").lower()
                        if allowed_namespaces and namespace not in allowed_namespaces:
                            continue
                        best = {
                            "curie": f"{term_info.get('db', '')}:{term_info.get('id', '')}",
                            "name": term_info.get("entry_name", ""),
                            "score": round(result.get("score", 0), 3),
                        }
                        break  # Take first match after filtering

                    return (term, best, None)

                except Exception as e:
                    logger.warning(f"Batch grounding failed for '{term}': {e}")
                    return (term, None, str(e))

        results = await asyncio.gather(*[_ground_one(t) for t in terms])

        # Merge results
        mappings = {}
        failed = {}
        for term, mapping, error in results:
            if error:
                failed[term] = {"error": error}
            elif mapping:
                mappings[term] = mapping
            else:
                failed[term] = {"error": "No matching grounding after namespace filtering"}

        result = {
            "mappings": mappings,
            "mapped_count": len(mappings),
            "failed_count": len(failed),
            "param_filter": param_name,
        }
        if failed:
            result["failed"] = failed
        return result

    except ImportError as e:
        return {
            "error": f"Grounding dependencies not available: {e}",
            "hint": "Install gilda: pip install gilda",
        }
    except Exception as e:
        return {"error": f"Batch grounding failed: {e}"}


def get_navigation_schema(
    entity_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Get the navigation schema (edge map) for discovery.

    Parameters
    ----------
    entity_type : str, optional
        Filter to edges from/to this entity type (e.g., "Gene", "Disease").
        When omitted, returns the full schema.

    Returns all possible navigation paths in the knowledge graph
    as extracted from function signatures.
    """
    registry, _, edge_map = _get_registry()

    schema = {
        "entity_types": sorted(set(
            list(edge_map.keys()) +
            [t for targets in edge_map.values() for t in targets.keys()]
        )),
        "edges": [],
    }

    for source, targets in sorted(edge_map.items()):
        for target, functions in sorted(targets.items()):
            # Filter by entity_type if specified
            if entity_type and source != entity_type and target != entity_type:
                continue

            # Return just function names — params are inferrable from name pattern
            schema["edges"].append({
                "from": source,
                "to": target,
                "functions": list(functions),
            })

    return schema


def register_gateway_tools(mcp, get_client_func: Callable) -> int:
    """Register gateway tools on MCP server.

    Only 5 tools instead of 100+:
    1. ground_entity - Natural language → CURIEs
    2. suggest_endpoints - Graph navigation suggestions
    3. call_endpoint - Execute any autoclient function
    4. get_navigation_schema - Full edge map for discovery
    5. batch_call - Execute endpoint for multiple entities in parallel

    Parameters
    ----------
    mcp : FastMCP
        MCP server instance
    get_client_func : Callable
        Function to get Neo4j client

    Returns
    -------
    :
        Number of tools registered (always 5)
    """
    # Pre-build registry
    _get_registry()

    @mcp.tool(
        name="ground_entity",
        annotations={"title": "Ground Entity (GILDA)", "readOnlyHint": True}
    )
    async def ground_entity_tool(
        term: Optional[str] = None,
        terms: Optional[List[str]] = None,
        organism: Optional[str] = None,
        limit: int = 10,
        param_name: Optional[str] = None,
    ) -> str:
        """Ground natural language to CURIEs using GILDA with semantic filtering.

        Supports single term or batch mode. Batch mode grounds multiple terms
        in one call, returning compact mappings.

        Parameter semantics eliminate cross-type ambiguity:
        - disease="ALS" → filters to disease namespaces → MESH:D000690 (not SOD1 gene)
        - gene="ALS" → filters to gene namespaces → HGNC:396 (SOD1)
        - drug="aspirin" → filters to drug namespaces → CHEBI:15365

        Parameters
        ----------
        term : str, optional
            Single natural language term (e.g., "LRRK2", "Parkinson's disease")
        terms : list of str, optional
            Batch mode: multiple terms to ground in one call.
            Returns compact {mappings: {term: {curie, name, score}}, failed: [...]}
        organism : str, optional
            Organism context (e.g., "human")
        limit : int
            Max results (default: 10)
        param_name : str, optional
            Parameter type for semantic filtering: disease, gene, drug, pathway, etc.
            Filters results to appropriate namespaces for that entity type.
        """
        result = await ground_entity(term, terms, organism, limit, param_name)
        return compact_json(result)

    @mcp.tool(
        name="suggest_endpoints",
        annotations={"title": "Suggest Navigation", "readOnlyHint": True}
    )
    async def suggest_endpoints_tool(
        entity_ids: List[str],
        intent: Optional[str] = None,
        top_k: int = 10,
    ) -> str:
        """Suggest where to navigate next in the knowledge graph.

        Given entity CURIEs you have, shows what entity types you can reach
        and which functions traverse those edges.

        Parameters
        ----------
        entity_ids : List[str]
            CURIEs from previous results (e.g., ["HGNC:6407"])
        intent : str, optional
            Exploration goal (e.g., "find drug targets")
        top_k : int
            Max suggestions per entity type
        """
        result = suggest_endpoints(entity_ids, intent, top_k)
        return compact_json(result)

    @mcp.tool(
        name="call_endpoint",
        annotations={"title": "Call Endpoint", "readOnlyHint": True}
    )
    async def call_endpoint_tool(
        endpoint: str,
        kwargs: str,
        auto_ground: bool = True,
        disclosure_level: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        include_navigation: bool = False,
        fields: Optional[List[str]] = None,
        estimate: bool = False,
        sort_by: Optional[str] = None,
    ) -> str:
        """Call any autoclient endpoint with optional auto-grounding.

        Parameters
        ----------
        endpoint : str
            Function name (e.g., "get_diseases_for_gene")
        kwargs : str
            JSON arguments. Entities can be:
            - CURIE tuple: {"gene": ["HGNC", "6407"]}
            - Natural language: {"gene": "LRRK2"} (auto-grounded)
        auto_ground : bool
            Auto-ground strings to CURIEs (default: True)
        disclosure_level : str, optional
            Enrich results with metadata. Options:
            - None (default): Raw results with names (most efficient)
            - "standard": Add descriptions + suggested next steps
            - "detailed": Add provenance metadata
            - "exploratory": Add workflows + research context
        offset : int
            Starting offset for pagination (default: 0). Use next_offset
            from previous response to continue fetching.
        limit : int, optional
            Max items per page. Responses are auto-truncated to ~20k tokens.
        include_navigation : bool
            Include suggested next navigation steps (default: False).
            Set to True when exploring unfamiliar entity types.
        fields : list of str, optional
            Project results to only these keys (e.g., ["db_ns", "db_id", "name"]).
            Dramatically reduces token usage for large result sets.
        estimate : bool
            Estimate mode (default: False). When True, executes the query and
            caches results server-side, but returns only metadata (count, token
            cost, sample, available fields). Use this to probe query cost before
            fetching full results. Subsequent calls with same endpoint+kwargs
            serve from cache.

            For large result sets, use estimate=True first to probe query cost,
            then fetch with fields projection and pagination:
              1. estimate=True -> see count, token cost, available fields, sample
              2. fields=["db_ns","db_id","name"], limit=50 -> fetch compact page
        sort_by : str, optional
            Sort results before pagination. Options:
            - "evidence": Descending by source_counts sum (most-validated first)
            - "name": Alphabetical by entity name
            Default: None (preserve query order).
        """
        result = await call_endpoint(
            endpoint, kwargs, get_client_func, auto_ground, disclosure_level,
            offset=offset, limit=limit,
            include_navigation=include_navigation, fields=fields,
            estimate=estimate, sort_by=sort_by,
        )
        return compact_json(result)

    @mcp.tool(
        name="get_navigation_schema",
        annotations={"title": "Get Navigation Schema", "readOnlyHint": True}
    )
    async def get_navigation_schema_tool(
        entity_type: Optional[str] = None,
    ) -> str:
        """Get full navigation schema showing all entity types and edges.

        Returns the complete map of how entity types connect in the
        knowledge graph, extracted from function signatures.

        Parameters
        ----------
        entity_type : str, optional
            Filter to edges from/to this entity type (e.g., "Gene", "Disease").
            Dramatically reduces response size when you only need edges for one type.
        """
        result = get_navigation_schema(entity_type=entity_type)
        return compact_json(result)

    @mcp.tool(
        name="batch_call",
        annotations={"title": "Batch Call Endpoint", "readOnlyHint": True}
    )
    async def batch_call_tool(
        endpoint: str,
        entity_param: str,
        entity_values: List[str],
        common_kwargs: str = "{}",
        auto_ground: bool = True,
        fields: Optional[List[str]] = None,
        max_concurrent: int = 10,
        merge_strategy: str = "keyed",
    ) -> str:
        """Execute an endpoint for multiple entities in parallel.

        Replaces N serial call_endpoint calls with 1 batch call.
        Internally fans out via asyncio.gather, or routes to native
        batch functions where available (single Neo4j query).

        Parameters
        ----------
        endpoint : str
            Function name (e.g., "get_diseases_for_gene")
        entity_param : str
            Which parameter to batch over (e.g., "gene")
        entity_values : list of str
            Entity values to query (e.g., ["SIRT3", "PRKN", "MAPT", "NLRP3"])
            Auto-grounded if auto_ground=True.
        common_kwargs : str
            JSON string of additional shared parameters (default: "{}")
        auto_ground : bool
            Auto-ground entity strings to CURIEs (default: True)
        fields : list of str, optional
            Project results to only these keys per item.
        max_concurrent : int
            Max parallel queries (default: 10, prevents Neo4j overload)
        merge_strategy : str
            "keyed" (default): {entity: results} dict.
            "flat": All results concatenated into one list.
        """
        result = await batch_call(
            endpoint, entity_param, entity_values, common_kwargs,
            get_client_func, auto_ground, fields, max_concurrent,
            merge_strategy,
        )
        return compact_json(result)

    logger.info("Registered 5 gateway tools: ground_entity, suggest_endpoints, "
                "call_endpoint, get_navigation_schema, batch_call")
    return 5


__all__ = [
    "register_gateway_tools",
    "suggest_endpoints",
    "call_endpoint",
    "batch_call",
    "ground_entity",
    "get_navigation_schema",
    "clear_registry_cache",
]

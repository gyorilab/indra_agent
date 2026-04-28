"""Function registry and edge map cache for MCP gateway."""

import logging
import random
import re
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from indra_agent.mcp_server.mappings import ENTITY_TYPE_MAPPINGS, _normalize_param_to_entity_type

logger = logging.getLogger(__name__)

# Module-level registry and edge map cache
#
# Cache Management Strategy:
# - Lazy initialization on first use (_get_registry)
# - Validation before returning cached values (_validate_cached_registry)
# - Automatic rebuild on corruption detection (e.g., 'dict' object not callable)
# - Thread-safe via RLock for concurrent MCP requests
# - Version tracking for diagnostics (clear_registry_cache increments)
# - Hot-reload support: call clear_registry_cache() when queries_web code changes
#
_FUNCTION_REGISTRY: Optional[Dict[str, Any]] = None
_FUNC_MAPPING: Optional[Dict[str, Callable]] = None
_EDGE_MAP: Optional[Dict[str, Dict[str, List[str]]]] = None  # source → target → [functions]
_CAPABILITY_INDEX: Optional[Dict[str, Any]] = None  # entity_type → category → [entries]

# Cache metadata for diagnostics and staleness detection
_CACHE_VERSION: int = 0
_CACHE_INITIALIZED_AT: Optional[datetime] = None

# Thread lock for cache operations (reentrant to allow same-thread recursive calls)
_cache_lock = threading.RLock()


def _validate_cached_registry() -> bool:
    """Validate that cached functions are actually callable.

    Samples random functions from the cache to detect corruption like
    non-callable objects that can occur during hot-reload scenarios.

    Returns
    -------
    :
        True if cache is valid, False if corrupted
    """
    global _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP

    try:
        # Check that basic structures exist and are correct types
        if not isinstance(_FUNCTION_REGISTRY, dict):
            logger.error(f"Registry corruption: _FUNCTION_REGISTRY is {type(_FUNCTION_REGISTRY)}, expected dict")
            return False

        if not isinstance(_FUNC_MAPPING, dict):
            logger.error(f"Registry corruption: _FUNC_MAPPING is {type(_FUNC_MAPPING)}, expected dict")
            return False

        if not isinstance(_EDGE_MAP, dict):
            logger.error(f"Registry corruption: _EDGE_MAP is {type(_EDGE_MAP)}, expected dict")
            return False

        # Sample validation: check that a few random functions from func_mapping are callable
        if _FUNC_MAPPING:
            sample_size = min(5, len(_FUNC_MAPPING))
            sample_funcs = random.sample(list(_FUNC_MAPPING.items()), sample_size)

            for func_name, func in sample_funcs:
                if not callable(func):
                    logger.error(f"Registry corruption: {func_name} is {type(func)}, expected callable")
                    return False

        logger.debug("Cache validation passed")
        return True

    except Exception as e:
        logger.error(f"Cache validation error: {e}")
        return False


def _clear_registry_internal():
    """Internal function to clear registry cache (assumes lock is held)."""
    global _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP, _CAPABILITY_INDEX

    _FUNCTION_REGISTRY = None
    _FUNC_MAPPING = None
    _EDGE_MAP = None
    _CAPABILITY_INDEX = None
    logger.debug("Registry cache cleared (internal)")


def invalidate_cache():
    """Invalidate and clear the function registry cache.

    Thread-safe cache invalidation. Next call to _get_registry() will rebuild.
    Alias for clear_registry_cache() for backward compatibility.
    """
    clear_registry_cache()


def clear_registry_cache() -> Dict[str, Any]:
    """Clear the function registry cache.

    Thread-safe cache invalidation. Next call to _get_registry() will rebuild.
    Use this when queries_web code has been modified while server is running.

    Returns
    -------
    :
        Dict with status, was_cached, previous_version, and timestamp
    """
    global _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP, _CACHE_VERSION

    with _cache_lock:
        was_cached = _FUNCTION_REGISTRY is not None
        old_version = _CACHE_VERSION

        _clear_registry_internal()

        logger.info(f"Registry cache cleared (was v{old_version}, cached={was_cached})")

        return {
            "status": "cleared",
            "was_cached": was_cached,
            "previous_version": old_version,
            "timestamp": datetime.now().isoformat(),
        }


def get_registry_status() -> Dict[str, Any]:
    """Get detailed status of registry cache for diagnostics.

    Thread-safe read of cache metadata without triggering initialization.
    Useful for debugging cache corruption or staleness issues.

    Returns
    -------
    :
        Dict with cached, version, initialized_at, age_seconds, metrics,
        validation, and status fields
    """
    global _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP, _CAPABILITY_INDEX, _CACHE_VERSION, _CACHE_INITIALIZED_AT

    with _cache_lock:
        is_cached = _FUNCTION_REGISTRY is not None

        if not is_cached:
            return {
                "cached": False,
                "version": _CACHE_VERSION,
                "status": "not initialized",
            }

        # Get cache metrics
        num_functions = len(_FUNCTION_REGISTRY) if _FUNCTION_REGISTRY else 0
        num_mappings = len(_FUNC_MAPPING) if _FUNC_MAPPING else 0
        num_edges = sum(len(targets) for targets in _EDGE_MAP.values()) if _EDGE_MAP else 0
        num_capabilities = 0
        if _CAPABILITY_INDEX and "_all" in _CAPABILITY_INDEX:
            num_capabilities = sum(len(entries) for entries in _CAPABILITY_INDEX["_all"].values())

        # Sample function validation
        sample_valid = True
        validation_error = None
        try:
            if _FUNC_MAPPING:
                sample = random.choice(list(_FUNC_MAPPING.items()))
                func_name, func = sample
                if not callable(func):
                    sample_valid = False
                    validation_error = f"{func_name} is {type(func)}, not callable"
        except Exception as e:
            sample_valid = False
            validation_error = str(e)

        # Time since initialization
        age_seconds = None
        if _CACHE_INITIALIZED_AT:
            age_seconds = (datetime.now() - _CACHE_INITIALIZED_AT).total_seconds()

        return {
            "cached": True,
            "version": _CACHE_VERSION,
            "initialized_at": _CACHE_INITIALIZED_AT.isoformat() if _CACHE_INITIALIZED_AT else None,
            "age_seconds": age_seconds,
            "metrics": {
                "registry_functions": num_functions,
                "func_mapping_entries": num_mappings,
                "navigation_edges": num_edges,
                "capability_entries": num_capabilities,
            },
            "validation": {
                "sample_check_passed": sample_valid,
                "error": validation_error,
            },
            "status": "healthy" if sample_valid else "corrupted",
        }


def _get_registry():
    """Lazily build and cache the function registry and edge map.

    Thread-safe lazy initialization with cache validation.
    """
    global _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP, _CAPABILITY_INDEX, _CACHE_VERSION, _CACHE_INITIALIZED_AT

    with _cache_lock:
        # Validate cache before returning
        if _FUNCTION_REGISTRY is not None:
            # Validate that cached functions are actually callable
            if not _validate_cached_registry():
                logger.warning("Cache validation failed - rebuilding registry")
                _clear_registry_internal()

        if _FUNCTION_REGISTRY is not None:
            return _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP

        logger.info("Building function registry and edge map...")

        try:
            from indra_cogex.apps.queries_web import (
                FUNCTION_CATEGORIES,
                CATEGORY_DESCRIPTIONS,
                func_mapping,
                module_functions,
                examples_dict,
                SKIP_GLOBAL,
                SKIP_ARGUMENTS,
            )
            from indra_cogex.apps.queries_web.helpers import get_docstring
            from indra_cogex.apps.queries_web.introspection import build_function_registry
        except ImportError as e:
            logger.warning(f"Could not import queries_web: {e}")
            _FUNCTION_REGISTRY = {}
            _FUNC_MAPPING = {}
            _EDGE_MAP = {}
            _CACHE_INITIALIZED_AT = datetime.now()
            return _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP

        # Build function registry
        _FUNCTION_REGISTRY = build_function_registry(
            module_functions=module_functions,
            func_mapping=func_mapping,
            function_categories=FUNCTION_CATEGORIES,
            category_descriptions=CATEGORY_DESCRIPTIONS,
            examples_dict=examples_dict,
            skip_global=SKIP_GLOBAL,
            skip_arguments=SKIP_ARGUMENTS,
            get_docstring_func=get_docstring,
        )
        _FUNC_MAPPING = func_mapping

        # Build edge map from function naming patterns
        _EDGE_MAP = _build_edge_map(_FUNCTION_REGISTRY)

        # Build capability index for functions not in edge map
        _CAPABILITY_INDEX = _build_capability_index(_FUNCTION_REGISTRY, _EDGE_MAP)

        # Update cache metadata
        _CACHE_VERSION += 1
        _CACHE_INITIALIZED_AT = datetime.now()

        num_cap = sum(len(e) for e in _CAPABILITY_INDEX.get("_all", {}).values())
        logger.info(f"Built function registry v{_CACHE_VERSION} with {len(_FUNCTION_REGISTRY)} functions, "
                    f"{sum(len(targets) for targets in _EDGE_MAP.values())} navigation edges, "
                    f"{num_cap} capability entries")
        return _FUNCTION_REGISTRY, _FUNC_MAPPING, _EDGE_MAP


def _build_edge_map(registry: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Build edge map from function naming patterns.

    Pattern analysis (from schema_builder approach):
    - get_X_for_Y → edge: Y → X (navigate from Y to X)
    - get_Xs_for_Y → edge: Y → X (plural)
    - is_X_in_Y → edge: X ∈ Y (membership check)
    - has_X_Y_association → edge: X ↔ Y (bidirectional)

    Returns
    -------
    :
        Nested dict: source_entity → target_entity → [function_names]
    """
    edge_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # Regex patterns for function name analysis
    get_for_pattern = re.compile(r'get_(\w+?)s?_for_(\w+?)s?$')
    is_in_pattern = re.compile(r'is_(\w+)_in_(\w+)')
    has_assoc_pattern = re.compile(r'has_(\w+)_(\w+)_association')

    for func_name, metadata in registry.items():
        # Try get_X_for_Y pattern (most common)
        match = get_for_pattern.match(func_name)
        if match:
            target_param = match.group(1)  # X (what we get)
            source_param = match.group(2)  # Y (what we have)

            target_type = ENTITY_TYPE_MAPPINGS.get(target_param.lower())
            source_type = ENTITY_TYPE_MAPPINGS.get(source_param.lower())

            if target_type and source_type:
                edge_map[source_type][target_type].append(func_name)
            continue

        # Try is_X_in_Y pattern
        match = is_in_pattern.match(func_name)
        if match:
            x_param = match.group(1)
            y_param = match.group(2)

            x_type = ENTITY_TYPE_MAPPINGS.get(x_param.lower())
            y_type = ENTITY_TYPE_MAPPINGS.get(y_param.lower())

            if x_type and y_type:
                # Both directions for membership checks
                edge_map[x_type][y_type].append(func_name)
            continue

        # Try has_X_Y_association pattern
        match = has_assoc_pattern.match(func_name)
        if match:
            x_param = match.group(1)
            y_param = match.group(2)

            x_type = ENTITY_TYPE_MAPPINGS.get(x_param.lower())
            y_type = ENTITY_TYPE_MAPPINGS.get(y_param.lower())

            if x_type and y_type:
                # Bidirectional for associations
                edge_map[x_type][y_type].append(func_name)
                edge_map[y_type][x_type].append(func_name)

    # Convert defaultdicts to regular dicts
    return {source: dict(targets) for source, targets in edge_map.items()}


def _build_capability_index(
    registry: Dict[str, Any],
    edge_map: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Build capability index for functions not covered by the edge map.

    Groups functions by the entity types their parameters accept,
    making them discoverable through suggest_endpoints and
    get_navigation_schema even though they don't follow the
    ``get_X_for_Y`` naming convention.

    Returns
    -------
    :
        ``{entity_type: {category: [entries]}, "_all": {category: [entries]}}``
        where each entry is ``{"name", "description", "entity_types"}``.
    """
    # Collect all functions already in the edge map
    edge_funcs: set = set()
    for targets in edge_map.values():
        for funcs in targets.values():
            edge_funcs.update(funcs)

    # Build index
    by_entity: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    all_by_cat: Dict[str, List[Dict]] = defaultdict(list)

    for func_name, metadata in registry.items():
        if func_name in edge_funcs:
            continue

        # metadata: [name, short_desc, full_doc, category, cat_desc, module, params, ...]
        short_desc = metadata[1] if len(metadata) > 1 else ""
        category = metadata[3] if len(metadata) > 3 else "uncategorized"
        params = metadata[6] if len(metadata) > 6 else {}

        # Resolve parameter names → entity types
        entity_types: set = set()
        for param_name in params:
            etype = _normalize_param_to_entity_type(param_name)
            if etype is not None:
                entity_types.add(etype)

        entry = {
            "name": func_name,
            "description": short_desc,
            "entity_types": sorted(entity_types),
        }

        for etype in entity_types:
            by_entity[etype][category].append(entry)

        all_by_cat[category].append(entry)

    result = {etype: dict(cats) for etype, cats in by_entity.items()}
    result["_all"] = dict(all_by_cat)
    return result


def _get_capability_index() -> Optional[Dict[str, Any]]:
    """Return the capability index, building the registry if needed."""
    global _CAPABILITY_INDEX

    with _cache_lock:
        if _CAPABILITY_INDEX is None:
            _get_registry()  # triggers build
        return _CAPABILITY_INDEX


__all__ = [
    "invalidate_cache",
    "clear_registry_cache",
    "get_registry_status",
    "_get_capability_index",
]

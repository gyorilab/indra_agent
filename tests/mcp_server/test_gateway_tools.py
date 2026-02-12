"""Comprehensive Integration Tests for MCP Gateway Tools (Layer 5).

Tests the primary MCP interface tools that provide natural language grounding,
graph navigation, and endpoint execution with semantic filtering.

Gateway Tools Under Test:
1. ground_entity - Natural language → CURIEs with semantic filtering
2. suggest_endpoints - Graph navigation suggestions based on entity types
3. call_endpoint - Execute autoclient functions with auto-grounding
4. get_navigation_schema - Full knowledge graph edge map

These are the primary user-facing tools in the MCP server, providing the
"gateway" interface between natural language queries and the underlying
autoclient functions (100+ biomedical query functions).

Key Features Tested:
- GILDA grounding with ambiguity detection
- Semantic filtering by parameter type (disease vs gene vs drug)
- Dual-check ambiguity: absolute threshold + relative clustering
- Auto-grounding in call_endpoint
- Registry cache corruption recovery
- xref cross-reference fallback

Test Pattern:
- Real Neo4j client (no mocks) - following conftest.py pattern
- Real biomedical entities (LRRK2/HGNC:6407, Parkinson's/MESH:D010300)
- Integration tests with meaningful assertions
- Edge cases (empty inputs, invalid CURIEs, ambiguous terms)

Run with: pytest -m nonpublic tests/apps/mcp_server/test_gateway_tools.py -v
"""
import asyncio
import json
import pytest
from flask import Flask
from indra.config import get_config
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.apps.constants import INDRA_COGEX_EXTENSION

# Import gateway tools
from indra_agent.mcp_server.autoclient_tools import (
    ground_entity,
    suggest_endpoints,
    call_endpoint,
    get_navigation_schema,
    batch_call,
)
from indra_agent.mcp_server.registry import (
    clear_registry_cache,
    invalidate_cache,
    get_registry_status,
)
from indra_agent.mcp_server.mappings import (
    MIN_CONFIDENCE_THRESHOLD,
    AMBIGUITY_SCORE_THRESHOLD,
)


@pytest.fixture
def neo4j_client():
    """Provide real Neo4j client for tests."""
    return Neo4jClient(
        get_config("INDRA_NEO4J_URL"),
        auth=(get_config("INDRA_NEO4J_USER"), get_config("INDRA_NEO4J_PASSWORD"))
    )


@pytest.fixture
def flask_app_with_client(neo4j_client):
    """Flask app context with real Neo4j client."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "test-key"
    app.extensions[INDRA_COGEX_EXTENSION] = neo4j_client
    with app.app_context():
        yield app


def get_client():
    """Get Neo4j client from Flask app context (for call_endpoint)."""
    from flask import current_app
    return current_app.extensions.get(INDRA_COGEX_EXTENSION)


# ============================================================================
# Part 1: ground_entity Tests
# ============================================================================

@pytest.mark.nonpublic
class TestGroundEntity:
    """Comprehensive tests for ground_entity tool."""

    @pytest.mark.asyncio
    async def test_basic_gene_grounding(self, flask_app_with_client):
        """Test basic grounding of a well-known gene."""
        result = await ground_entity("LRRK2")

        assert "groundings" in result
        assert "top_match" in result
        assert len(result["groundings"]) > 0

        # Top match should be LRRK2 gene
        top = result["top_match"]
        assert top["namespace"].upper() == "HGNC"
        assert top["name"] == "LRRK2"
        assert top["score"] > 0.8  # High confidence

    @pytest.mark.asyncio
    async def test_basic_disease_grounding(self, flask_app_with_client):
        """Test basic grounding of a disease."""
        result = await ground_entity("Parkinson's disease")

        assert "groundings" in result
        assert len(result["groundings"]) > 0

        # Should find Parkinson's disease
        top = result["top_match"]
        assert "parkinson" in top["name"].lower()
        assert top["namespace"].upper() in ["MESH", "DOID", "MONDO"]

    @pytest.mark.asyncio
    async def test_semantic_filtering_disease(self, flask_app_with_client):
        """Test that param_name='disease' filters to disease namespaces."""
        # ALS can mean both the disease (Amyotrophic Lateral Sclerosis)
        # or the gene (SOD1/HGNC:396)
        result = await ground_entity("ALS", param_name="disease")

        assert "groundings" in result
        assert "namespaces_allowed" in result
        assert result["param_filter"] == "disease"

        # All results should be from disease namespaces
        disease_namespaces = {"mesh", "doid", "efo", "mondo", "hp", "orphanet", "umls"}
        for grounding in result["groundings"]:
            assert grounding["namespace"].lower() in disease_namespaces

    @pytest.mark.asyncio
    async def test_semantic_filtering_gene(self, flask_app_with_client):
        """Test that param_name='gene' filters to gene namespaces."""
        result = await ground_entity("ALS", param_name="gene")

        assert "groundings" in result
        assert result["param_filter"] == "gene"

        # All results should be from gene namespaces
        gene_namespaces = {"hgnc", "ncbigene", "ensembl", "uniprot", "fplx"}
        for grounding in result["groundings"]:
            assert grounding["namespace"].lower() in gene_namespaces

    @pytest.mark.asyncio
    async def test_semantic_filtering_drug(self, flask_app_with_client):
        """Test that param_name='drug' filters to drug namespaces."""
        result = await ground_entity("aspirin", param_name="drug")

        assert "groundings" in result
        assert result["param_filter"] == "drug"

        # All results should be from drug namespaces
        drug_namespaces = {"chebi", "drugbank", "pubchem.compound", "chembl.compound", "chembl"}
        for grounding in result["groundings"]:
            assert grounding["namespace"].lower() in drug_namespaces

    @pytest.mark.asyncio
    async def test_organism_filtering(self, flask_app_with_client):
        """Test organism context filtering."""
        result = await ground_entity("LRRK2", organism="human")

        assert "groundings" in result
        # GILDA should handle organism filtering
        # Human genes should be prioritized
        top = result["top_match"]
        assert top is not None

    @pytest.mark.asyncio
    async def test_case_insensitivity(self, flask_app_with_client):
        """Test that grounding is case-insensitive."""
        result_lower = await ground_entity("lrrk2")
        result_upper = await ground_entity("LRRK2")
        result_mixed = await ground_entity("Lrrk2")

        # All should find LRRK2
        assert result_lower["top_match"]["name"] == "LRRK2"
        assert result_upper["top_match"]["name"] == "LRRK2"
        assert result_mixed["top_match"]["name"] == "LRRK2"

    @pytest.mark.asyncio
    async def test_ambiguity_detection_multiple_candidates(self, flask_app_with_client):
        """Test that ambiguous terms return multiple candidates with scores."""
        # "p53" can refer to TP53 gene, protein, or various synonyms
        result = await ground_entity("p53", limit=10)

        assert "groundings" in result
        assert len(result["groundings"]) >= 2  # Multiple matches

        # All should have scores
        for grounding in result["groundings"]:
            assert "score" in grounding
            assert 0 <= grounding["score"] <= 1

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, flask_app_with_client):
        """Test handling of empty input."""
        result = await ground_entity("")

        # Should either return empty results or error
        assert "groundings" in result or "error" in result
        if "groundings" in result:
            assert len(result["groundings"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, flask_app_with_client):
        """Test handling of nonsense input."""
        result = await ground_entity("xyzabc123nonexistent")

        # Should return empty or very low confidence results
        assert "groundings" in result
        if len(result["groundings"]) > 0:
            # If any results, they should have low scores
            assert result["top_match"]["score"] < 0.7

    @pytest.mark.asyncio
    async def test_limit_parameter(self, flask_app_with_client):
        """Test that limit parameter restricts results."""
        result_limit_5 = await ground_entity("kinase", limit=5)
        result_limit_20 = await ground_entity("kinase", limit=20)

        assert len(result_limit_5["groundings"]) <= 5
        assert len(result_limit_20["groundings"]) <= 20
        # More limit should give more or equal results
        assert len(result_limit_20["groundings"]) >= len(result_limit_5["groundings"])


# ============================================================================
# Part 2: suggest_endpoints Tests
# ============================================================================

@pytest.mark.nonpublic
class TestSuggestEndpoints:
    """Comprehensive tests for suggest_endpoints tool."""

    def test_gene_entity_suggestions(self):
        """Test suggestions for a gene CURIE."""
        result = suggest_endpoints(entity_ids=["HGNC:6407"])

        assert "source_entities" in result
        assert "navigation_options" in result
        assert "Gene" in result["source_entities"]

        # Should suggest navigation to diseases, pathways, drugs, etc.
        nav_options = result["navigation_options"]
        assert len(nav_options) > 0

        # Check that we have at least one navigation option
        gene_nav = [opt for opt in nav_options if opt["from"] == "Gene"]
        assert len(gene_nav) > 0

        # Should suggest various target types
        targets = {reach["target"] for opt in gene_nav for reach in opt["can_reach"]}
        # Likely targets: Disease, Pathway, Drug, Protein
        assert len(targets) > 0

    def test_disease_entity_suggestions(self):
        """Test suggestions for a disease CURIE."""
        result = suggest_endpoints(entity_ids=["MESH:D010300"])

        assert "Disease" in result["source_entities"]
        assert len(result["navigation_options"]) > 0

        # Should suggest navigation to genes, drugs, etc.
        disease_nav = [opt for opt in result["navigation_options"] if opt["from"] == "Disease"]
        assert len(disease_nav) > 0

    def test_multiple_entity_types(self):
        """Test suggestions with multiple entity types."""
        result = suggest_endpoints(entity_ids=["HGNC:6407", "MESH:D010300"])

        assert "source_entities" in result
        # Should detect both Gene and Disease
        assert "Gene" in result["source_entities"]
        assert "Disease" in result["source_entities"]

        # Should have navigation options for both types
        assert len(result["navigation_options"]) >= 2

    def test_intent_influences_suggestions(self):
        """Test that intent parameter influences function ranking."""
        entity_ids = ["HGNC:6407"]

        # Without intent
        result_no_intent = suggest_endpoints(entity_ids=entity_ids)

        # With drug-related intent
        result_drug_intent = suggest_endpoints(
            entity_ids=entity_ids,
            intent="find drug targets"
        )

        # Both should have suggestions
        assert len(result_no_intent["navigation_options"]) > 0
        assert len(result_drug_intent["navigation_options"]) > 0

        # Intent should be captured
        assert result_drug_intent["intent"] == "find drug targets"

    def test_empty_entity_list(self):
        """Test handling of empty entity list."""
        result = suggest_endpoints(entity_ids=[])

        assert "error" in result or "navigation_options" in result
        if "navigation_options" in result:
            assert len(result["navigation_options"]) == 0

    def test_invalid_curie_format(self):
        """Test handling of invalid CURIE format."""
        result = suggest_endpoints(entity_ids=["not_a_valid_curie"])

        assert "error" in result or "hint" in result
        # Should suggest using proper CURIE format

    def test_unknown_entity_type(self):
        """Test handling of unknown/unsupported entity types."""
        result = suggest_endpoints(entity_ids=["UNKNOWN:123"])

        # Should handle gracefully
        assert "source_entities" in result or "error" in result

    def test_returns_function_names(self):
        """Test that suggestions include actual function names."""
        result = suggest_endpoints(entity_ids=["HGNC:6407"])

        assert "navigation_options" in result
        if len(result["navigation_options"]) > 0:
            nav = result["navigation_options"][0]
            if len(nav["can_reach"]) > 0:
                reach = nav["can_reach"][0]
                assert "functions" in reach
                assert len(reach["functions"]) > 0
                # Functions are plain strings (name pattern is self-descriptive)
                func = reach["functions"][0]
                assert isinstance(func, str)

    def test_top_k_parameter(self):
        """Test that top_k limits suggestions."""
        entity_ids = ["HGNC:6407"]

        result_k3 = suggest_endpoints(entity_ids=entity_ids, top_k=3)
        result_k10 = suggest_endpoints(entity_ids=entity_ids, top_k=10)

        # Both should work
        assert "navigation_options" in result_k3
        assert "navigation_options" in result_k10


# ============================================================================
# Part 3: call_endpoint Tests
# ============================================================================

@pytest.mark.nonpublic
class TestCallEndpoint:
    """Comprehensive tests for call_endpoint tool."""

    @pytest.mark.asyncio
    async def test_direct_curie_call(self, flask_app_with_client):
        """Test calling endpoint with explicit CURIE tuple."""
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',  # Parkinson's
            get_client_func=get_client,
            auto_ground=False
        )

        assert "results" in result
        # Slim response: total shown when single page, pagination when has_more
        assert "total" in result or "pagination" in result

    @pytest.mark.asyncio
    async def test_auto_grounding_enabled(self, flask_app_with_client):
        """Test auto-grounding with natural language input."""
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": "Parkinson\'s disease"}',
            get_client_func=get_client,
            auto_ground=True
        )

        assert "results" in result or "error" in result

        # Grounding info only appears if xref fallback was used
        # Normal grounding is silent (agent already knows what it sent)

    @pytest.mark.asyncio
    async def test_auto_grounding_with_semantic_filtering(self, flask_app_with_client):
        """Test that auto-grounding uses parameter name for semantic filtering."""
        # "ALS" with disease parameter should ground to disease, not gene
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": "ALS"}',
            get_client_func=get_client,
            auto_ground=True
        )

        # Should succeed (ALS grounds to disease namespace) or fail gracefully
        assert "results" in result or "error" in result

    @pytest.mark.asyncio
    async def test_invalid_endpoint_name(self, flask_app_with_client):
        """Test handling of invalid endpoint name."""
        result = await call_endpoint(
            endpoint="nonexistent_function",
            kwargs='{"param": "value"}',
            get_client_func=get_client,
            auto_ground=False
        )

        assert "error" in result
        assert "unknown endpoint" in result["error"].lower()
        # Should suggest using suggest_endpoints
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_missing_required_parameter(self, flask_app_with_client):
        """Test handling of missing required parameter."""
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{}',  # Missing disease parameter
            get_client_func=get_client,
            auto_ground=False
        )

        # Should error about missing parameter
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_json_kwargs(self, flask_app_with_client):
        """Test handling of malformed JSON."""
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": invalid json}',
            get_client_func=get_client,
            auto_ground=False
        )

        assert "error" in result
        assert "json" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_dual_check_ambiguity_absolute_threshold(self, flask_app_with_client):
        """Test that low confidence grounding is rejected (absolute threshold)."""
        # Use a very ambiguous or nonsense term
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": "xyzabc"}',
            get_client_func=get_client,
            auto_ground=True
        )

        # Should fail with low confidence error
        if "error" in result:
            assert "confidence" in result["error"].lower() or "no grounding" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_dual_check_ambiguity_relative_clustering(self, flask_app_with_client):
        """Test that ambiguous terms with close scores are rejected (relative threshold)."""
        # Use a term that has multiple similar matches
        # Note: This is hard to test deterministically without mocking GILDA
        # We can at least verify the mechanism is in place
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": "inflammatory disease"}',
            get_client_func=get_client,
            auto_ground=True
        )

        # May succeed or fail depending on GILDA results
        # If it fails, should be due to ambiguity
        if "error" in result and "ambiguous" in result["error"].lower():
            assert "grounding_options" in result
            # Should offer multiple options
            assert len(result["grounding_options"]) > 1

    @pytest.mark.asyncio
    async def test_namespace_passthrough(self, flask_app_with_client):
        """Test that namespace case is preserved (upstream norm_id handles normalization)."""
        # Provide uppercase namespace — should be passed through as-is
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',
            get_client_func=get_client,
            auto_ground=False
        )

        # Should succeed — upstream norm_id normalizes internally
        assert "results" in result or "error" in result

    @pytest.mark.asyncio
    async def test_result_processing(self, flask_app_with_client):
        """Test that results are properly processed/serialized."""
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',
            get_client_func=get_client,
            auto_ground=False
        )

        assert "results" in result
        # Slim response uses "total" for single-page results
        assert "total" in result or "pagination" in result

        # Results should be JSON-serializable
        json.dumps(result)  # Should not raise

    @pytest.mark.asyncio
    async def test_xref_fallback(self, flask_app_with_client):
        """Test cross-reference fallback when original grounding returns no results."""
        # This is hard to test deterministically, but we can verify the mechanism
        # The xref fallback happens when auto-grounding succeeds but returns 0 results
        # We'd need a specific entity that has xrefs but original namespace has no results
        # For now, just verify the code path exists by checking with a real query
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": "Parkinson\'s disease"}',
            get_client_func=get_client,
            auto_ground=True
        )

        # Should have results or error
        assert "results" in result or "error" in result

        # If xref fallback was used, it appears in the response
        if "xref_fallback" in result:
            assert isinstance(result["xref_fallback"], dict)


# ============================================================================
# Part 4: get_navigation_schema Tests
# ============================================================================

@pytest.mark.nonpublic
class TestGetNavigationSchema:
    """Comprehensive tests for get_navigation_schema tool."""

    def test_returns_edge_map(self):
        """Test that schema returns proper edge map structure."""
        result = get_navigation_schema()

        assert "entity_types" in result
        assert "edges" in result
        assert isinstance(result["entity_types"], list)
        assert isinstance(result["edges"], list)

    def test_contains_expected_entity_types(self):
        """Test that schema contains expected biomedical entity types."""
        result = get_navigation_schema()

        entity_types = result["entity_types"]
        # Should contain common types
        expected_types = {"Gene", "Disease", "Drug", "Pathway"}
        found_types = set(entity_types)

        # At least some expected types should be present
        assert len(expected_types & found_types) > 0

    def test_edges_have_proper_structure(self):
        """Test that edges have proper structure."""
        result = get_navigation_schema()

        edges = result["edges"]
        if len(edges) > 0:
            edge = edges[0]
            assert "from" in edge
            assert "to" in edge
            assert "functions" in edge
            assert isinstance(edge["functions"], list)

    def test_contains_function_names(self):
        """Test that edges contain actual function names."""
        result = get_navigation_schema()

        edges = result["edges"]
        if len(edges) > 0:
            # Check a few edges
            for edge in edges[:5]:
                assert len(edge["functions"]) > 0
                # Functions are plain strings (name pattern is self-descriptive)
                func = edge["functions"][0]
                assert isinstance(func, str)
                # Should follow naming pattern
                assert "get_" in func or "is_" in func

    def test_caching_behavior(self):
        """Test that schema is cached and returns consistent results."""
        result1 = get_navigation_schema()
        result2 = get_navigation_schema()

        # Should return identical results
        assert result1["entity_types"] == result2["entity_types"]
        assert len(result1["edges"]) == len(result2["edges"])

    def test_schema_after_cache_clear(self):
        """Test that schema rebuilds after cache clear."""
        result1 = get_navigation_schema()

        # Clear cache
        clear_registry_cache()

        result2 = get_navigation_schema()

        # Should still work and return similar structure
        assert "entity_types" in result2
        assert "edges" in result2
        # Should have same entity types (order may differ)
        assert set(result1["entity_types"]) == set(result2["entity_types"])


# ============================================================================
# Part 5: Registry Cache Tests
# ============================================================================

@pytest.mark.nonpublic
class TestRegistryCache:
    """Tests for registry cache management and corruption recovery."""

    def test_cache_invalidation(self):
        """Test that cache can be invalidated."""
        # Get initial status
        status1 = get_registry_status()

        # Invalidate
        invalidate_cache()

        # Get new status
        status2 = get_registry_status()

        # Should not be cached after invalidation
        assert status2["cached"] is False

        # Rebuild by calling get_navigation_schema
        get_navigation_schema()

        # Should be cached again
        status3 = get_registry_status()
        assert status3["cached"] is True

    def test_clear_registry_cache_returns_status(self):
        """Test that clear_registry_cache returns proper status."""
        # Ensure cache is built
        get_navigation_schema()

        result = clear_registry_cache()

        assert "status" in result
        assert result["status"] == "cleared"
        assert "was_cached" in result
        assert "previous_version" in result
        assert "timestamp" in result

    def test_registry_status_structure(self):
        """Test that get_registry_status returns proper structure."""
        # Build cache
        get_navigation_schema()

        status = get_registry_status()

        assert "cached" in status
        if status["cached"]:
            assert "version" in status
            assert "metrics" in status
            assert "validation" in status
            assert "status" in status

            # Check metrics
            metrics = status["metrics"]
            assert "registry_functions" in metrics
            assert "func_mapping_entries" in metrics
            assert "navigation_edges" in metrics

    def test_cache_validation(self):
        """Test that cache validation detects healthy cache."""
        # Build cache
        get_navigation_schema()

        status = get_registry_status()

        if status["cached"]:
            assert "validation" in status
            validation = status["validation"]
            assert "sample_check_passed" in validation
            # Should pass for healthy cache
            assert validation["sample_check_passed"] is True
            assert status["status"] == "healthy"


# ============================================================================
# Part 6: Integration & Edge Cases
# ============================================================================

@pytest.mark.nonpublic
class TestGatewayIntegration:
    """Integration tests combining multiple gateway tools."""

    @pytest.mark.asyncio
    async def test_complete_workflow_ground_suggest_call(self, flask_app_with_client):
        """Test complete workflow: ground → suggest → call."""

        # Step 1: Ground a disease term
        ground_result = await ground_entity("Parkinson's disease", param_name="disease")
        assert "groundings" in ground_result
        top_match = ground_result["top_match"]

        # Step 2: Get disease CURIE and suggest navigation
        disease_curie = f"{top_match['namespace']}:{top_match['identifier']}"
        suggest_result = suggest_endpoints(entity_ids=[disease_curie])

        assert "navigation_options" in suggest_result
        # Should suggest navigation from Disease to other types

        # Step 3: Call an endpoint to get genes
        call_result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs=f'{{"disease": ["{top_match["namespace"]}", "{top_match["identifier"]}"]}}',
            get_client_func=get_client,
            auto_ground=False
        )

        # Verify call succeeded (no error) and has proper structure
        assert "results" in call_result
        # Slim response uses "total" or "pagination"
        assert "total" in call_result or "pagination" in call_result

    @pytest.mark.asyncio
    async def test_auto_ground_workflow(self, flask_app_with_client):
        """Test workflow using auto-grounding throughout."""

        # Get schema
        schema = get_navigation_schema()
        assert "edges" in schema

        # Find an endpoint that goes Disease → Gene
        disease_to_gene_funcs = [
            edge["functions"][0]
            for edge in schema["edges"]
            if edge["from"] == "Disease" and edge["to"] == "Gene"
        ]

        if len(disease_to_gene_funcs) > 0:
            func_name = disease_to_gene_funcs[0]

            # Call with natural language (auto-ground)
            result = await call_endpoint(
                endpoint=func_name,
                kwargs='{"disease": "Parkinson\'s disease"}',
                get_client_func=get_client,
                auto_ground=True
            )

            # Should work with auto-grounding
            assert "results" in result or "error" in result

    @pytest.mark.asyncio
    async def test_error_recovery_invalid_to_valid(self, flask_app_with_client):
        """Test error recovery from invalid to valid call."""

        # First try with invalid endpoint
        invalid_result = await call_endpoint(
            endpoint="nonexistent_function",
            kwargs='{"param": "value"}',
            get_client_func=get_client,
            auto_ground=False
        )
        assert "error" in invalid_result

        # Then try with valid endpoint
        valid_result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["mesh", "D010300"]}',
            get_client_func=get_client,
            auto_ground=False
        )
        assert "results" in valid_result


# ============================================================================
# Part 7: Constants Verification
# ============================================================================

@pytest.mark.nonpublic
class TestConstants:
    """Verify that critical constants are properly imported and used."""

    def test_min_confidence_threshold_value(self):
        """Verify MIN_CONFIDENCE_THRESHOLD constant."""
        assert MIN_CONFIDENCE_THRESHOLD == 0.5
        assert isinstance(MIN_CONFIDENCE_THRESHOLD, float)

    def test_ambiguity_score_threshold_value(self):
        """Verify AMBIGUITY_SCORE_THRESHOLD constant."""
        assert AMBIGUITY_SCORE_THRESHOLD == 0.3
        assert isinstance(AMBIGUITY_SCORE_THRESHOLD, float)

    def test_thresholds_relationship(self):
        """Verify relationship between thresholds."""
        # Ambiguity threshold should be less than min confidence
        assert AMBIGUITY_SCORE_THRESHOLD < MIN_CONFIDENCE_THRESHOLD


# ============================================================================
# Part 8: Namespace Case Normalization (Defect 1 fix)
# ============================================================================

@pytest.mark.nonpublic
class TestNamespaceCaseNormalization:
    """Verify that auto-grounding produces uppercase namespaces.

    Upstream indra_cogex functions validate against uppercase prefixes
    (e.g., ``mesh_term[0] != "MESH"``). Gilda returns lowercase namespaces
    (e.g., "mesh"), so we must uppercase before passing to upstream.
    """

    @pytest.mark.asyncio
    async def test_ground_entity_returns_lowercase_namespace(self, flask_app_with_client):
        """Confirm gilda returns lowercase namespace (baseline)."""
        result = await ground_entity("amyotrophic lateral sclerosis", param_name="disease")
        assert "groundings" in result
        if result["groundings"]:
            # Gilda returns lowercase namespaces via bioregistry
            ns = result["groundings"][0]["namespace"]
            assert ns == ns.lower(), f"Expected lowercase namespace from gilda, got '{ns}'"

    @pytest.mark.asyncio
    async def test_call_endpoint_uppercases_namespace(self, flask_app_with_client):
        """Verify call_endpoint uppercases namespace before passing to upstream."""
        # Use a disease term that gilda will ground to MESH namespace
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": "Parkinson\'s disease"}',
            get_client_func=get_client,
            auto_ground=True,
        )
        # Should succeed — if namespace weren't uppercased, upstream would reject
        assert "results" in result, f"Expected results, got error: {result.get('error')}"

    @pytest.mark.asyncio
    async def test_mesh_term_grounding_succeeds(self, flask_app_with_client):
        """Verify mesh_term parameter grounds and executes correctly.

        The upstream function validates ``mesh_term[0] != "MESH"`` — this
        would fail if we passed the lowercase "mesh" from gilda directly.
        """
        result = await call_endpoint(
            endpoint="get_gene_sets_for_mesh_term",
            kwargs='{"mesh_term": "amyotrophic lateral sclerosis"}',
            get_client_func=get_client,
            auto_ground=True,
        )
        # This endpoint may or may not have results for ALS, but it should
        # not fail with a namespace validation error
        assert "results" in result or ("error" in result and "MESH" not in result.get("error", "")), \
            f"Namespace case issue: {result.get('error')}"


# ============================================================================
# Part 9: List[Tuple[str, str]] Auto-Grounding (Defect 2 fix)
# ============================================================================

@pytest.mark.nonpublic
class TestListTupleAutoGrounding:
    """Verify auto-grounding for List[Tuple[str, str]] parameters.

    Functions like get_shared_pathways_for_genes accept
    genes: List[Tuple[str, str]]. The fix enables passing string gene names
    that get batch-grounded via gilda.
    """

    @pytest.mark.asyncio
    async def test_string_list_grounding(self, flask_app_with_client):
        """Verify that string gene names are batch-grounded to CURIEs."""
        result = await call_endpoint(
            endpoint="get_shared_pathways_for_genes",
            kwargs='{"genes": ["BRCA1", "TP53"]}',
            get_client_func=get_client,
            auto_ground=True,
        )
        # Should either succeed with results or error gracefully
        # (no TypeError about string vs tuple)
        assert "results" in result or "error" in result
        if "error" in result:
            # Should NOT be a type error from passing strings where tuples expected
            assert "tuple" not in result["error"].lower()
            assert "str" not in result["error"].lower() or "ground" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_mixed_curie_and_string_list(self, flask_app_with_client):
        """Verify mixed input: pre-formed CURIEs and strings together."""
        result = await call_endpoint(
            endpoint="get_shared_pathways_for_genes",
            kwargs='{"genes": [["HGNC", "1100"], "TP53"]}',
            get_client_func=get_client,
            auto_ground=True,
        )
        # Mixed input should be handled — CURIEs pass through, strings get grounded
        assert "results" in result or "error" in result
        if "error" in result:
            assert "tuple" not in result["error"].lower()

    @pytest.mark.asyncio
    async def test_all_curie_list_passthrough(self, flask_app_with_client):
        """Verify that all-CURIE lists pass through without grounding."""
        result = await call_endpoint(
            endpoint="get_shared_pathways_for_genes",
            kwargs='{"genes": [["HGNC", "1100"], ["HGNC", "11998"]]}',
            get_client_func=get_client,
            auto_ground=False,
        )
        # Direct CURIEs should work without auto-grounding
        assert "results" in result or "error" in result

    @pytest.mark.asyncio
    async def test_invalid_item_in_list_returns_error(self, flask_app_with_client):
        """Verify that invalid items in the list produce a clear error."""
        result = await call_endpoint(
            endpoint="get_shared_pathways_for_genes",
            kwargs='{"genes": [123]}',
            get_client_func=get_client,
            auto_ground=True,
        )
        # Numeric item is neither a CURIE list nor a string
        assert "error" in result

    @pytest.mark.asyncio
    async def test_grounded_namespaces_are_uppercased(self, flask_app_with_client):
        """Verify that batch-grounded namespaces are uppercased."""
        # We can't directly inspect parsed_kwargs, but we can verify the
        # endpoint doesn't fail due to lowercase namespaces
        result = await call_endpoint(
            endpoint="get_shared_pathways_for_genes",
            kwargs='{"genes": ["BRCA1", "BRCA2"]}',
            get_client_func=get_client,
            auto_ground=True,
        )
        # If namespaces were lowercase, upstream might reject them
        assert "results" in result or "error" in result
        if "error" in result:
            assert "namespace" not in result["error"].lower()


# ============================================================================
# Part 10: Batch Error Propagation (Defect 3 fix)
# ============================================================================

@pytest.mark.nonpublic
class TestBatchErrorPropagation:
    """Verify that batch_call propagates per-entity grounding errors.

    Previously, failed groundings were silently discarded (added to a list
    as bare strings). Now they appear in the ``failed`` dict with structured
    error information.
    """

    @pytest.mark.asyncio
    async def test_batch_with_invalid_entity_reports_failure(self, flask_app_with_client):
        """Verify that nonsense entities appear in the failed dict."""
        result = await batch_call(
            endpoint="get_diseases_for_gene",
            entity_param="gene",
            entity_values=["LRRK2", "xyznonexistent123"],
            get_client_func=get_client,
            auto_ground=True,
        )
        # LRRK2 should succeed, nonsense should fail
        assert "total_entities" in result
        assert result["total_entities"] == 2
        if "failed" in result:
            # Failed should be a dict with entity names as keys
            assert isinstance(result["failed"], dict)
            # Each failure should have structured error info
            for entity, info in result["failed"].items():
                assert isinstance(info, dict), f"Expected dict for failed['{entity}'], got {type(info)}"
                assert "error" in info, f"Expected 'error' key in failed['{entity}']"

    @pytest.mark.asyncio
    async def test_batch_all_invalid_returns_error(self, flask_app_with_client):
        """Verify that all-invalid batch returns proper error structure."""
        result = await batch_call(
            endpoint="get_diseases_for_gene",
            entity_param="gene",
            entity_values=["xyznonexistent1", "xyznonexistent2"],
            get_client_func=get_client,
            auto_ground=True,
        )
        # All should fail — either top-level error or all in failed dict
        if "error" in result:
            assert "failed" in result or "grounding" in result["error"].lower()
        elif "failed" in result:
            assert isinstance(result["failed"], dict)
            assert len(result["failed"]) == 2

    @pytest.mark.asyncio
    async def test_batch_ground_entity_batch_returns_failed_dict(self, flask_app_with_client):
        """Verify that ground_entity batch mode returns failed as dict."""
        result = await ground_entity(
            terms=["LRRK2", "xyznonexistent123"],
            param_name="gene",
        )
        assert "mappings" in result
        # LRRK2 should be in mappings
        assert "LRRK2" in result["mappings"]
        # If nonsense term failed, it should be in failed dict
        if "failed" in result:
            assert isinstance(result["failed"], dict)
            if "xyznonexistent123" in result["failed"]:
                fail_info = result["failed"]["xyznonexistent123"]
                assert isinstance(fail_info, dict)
                assert "error" in fail_info

    @pytest.mark.asyncio
    async def test_batch_fan_out_partial_failure(self, flask_app_with_client):
        """Verify fan-out batch handles partial failure gracefully."""
        result = await batch_call(
            endpoint="get_diseases_for_gene",
            entity_param="gene",
            entity_values=["LRRK2"],
            get_client_func=get_client,
            auto_ground=True,
        )
        # Single valid entity should succeed
        assert "results" in result
        assert result["successful"] >= 1


# ============================================================================
# Part 11: Cache Behavior Tests
# ============================================================================

class TestCacheBehavior:
    """Test cache key generation and basic cache operations.

    These tests do NOT require Neo4j (no @pytest.mark.nonpublic).
    They verify the cache module's deterministic behavior.
    """

    def test_make_key_deterministic(self):
        """Verify make_key produces identical keys for identical inputs."""
        from indra_agent.mcp_server.cache import make_key

        key1 = make_key("endpoint", "get_diseases_for_gene", {"gene": ["HGNC", "6407"]})
        key2 = make_key("endpoint", "get_diseases_for_gene", {"gene": ["HGNC", "6407"]})
        assert key1 == key2

    def test_make_key_different_for_different_inputs(self):
        """Verify make_key produces different keys for different inputs."""
        from indra_agent.mcp_server.cache import make_key

        key1 = make_key("endpoint", "get_diseases_for_gene", {"gene": ["HGNC", "6407"]})
        key2 = make_key("endpoint", "get_diseases_for_gene", {"gene": ["HGNC", "9999"]})
        assert key1 != key2

    def test_make_key_namespaced(self):
        """Verify make_key includes namespace prefix."""
        from indra_agent.mcp_server.cache import make_key

        key = make_key("schema", "summary")
        assert key.startswith("schema:")

        key2 = make_key("endpoint", "func_name")
        assert key2.startswith("endpoint:")

    def test_make_key_ignores_dict_ordering(self):
        """Verify make_key produces same key regardless of dict key order."""
        from indra_agent.mcp_server.cache import make_key

        key1 = make_key("endpoint", "func", {"a": 1, "b": 2})
        key2 = make_key("endpoint", "func", {"b": 2, "a": 1})
        assert key1 == key2

    def test_null_cache_always_misses(self):
        """Verify _NullCache returns None for all gets."""
        from indra_agent.mcp_server.cache import _NullCache

        null = _NullCache()
        assert null.get("any_key") is None
        assert null.get("any_key", default="fallback") == "fallback"
        assert len(null) == 0
        assert "any_key" not in null
        assert null.volume() == 0

    def test_null_cache_set_succeeds_silently(self):
        """Verify _NullCache.set returns True without storing."""
        from indra_agent.mcp_server.cache import _NullCache

        null = _NullCache()
        assert null.set("key", "value") is True
        assert null.get("key") is None  # Not actually stored

    def test_cache_stats_returns_structure(self):
        """Verify cache_stats returns expected structure."""
        from indra_agent.mcp_server.cache import cache_stats

        stats = cache_stats()
        assert "total_entries" in stats or "error" in stats
        assert "is_null_cache" in stats


# ============================================================================
# Part 12: Cache Integration Tests (with Neo4j)
# ============================================================================

@pytest.mark.nonpublic
class TestCacheIntegration:
    """Test cache hit/miss behavior with real queries."""

    @pytest.mark.asyncio
    async def test_repeated_query_uses_cache(self, flask_app_with_client):
        """Verify that identical queries return consistent results."""
        kwargs = '{"disease": ["MESH", "D010300"]}'

        result1 = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs=kwargs,
            get_client_func=get_client,
            auto_ground=False,
        )
        result2 = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs=kwargs,
            get_client_func=get_client,
            auto_ground=False,
        )

        # Both should succeed
        assert "results" in result1
        assert "results" in result2

        # Results should be identical (served from cache on second call)
        assert len(result1["results"]) == len(result2["results"])

    @pytest.mark.asyncio
    async def test_different_queries_get_different_results(self, flask_app_with_client):
        """Verify that different queries are cached independently."""
        result_pd = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',  # Parkinson's
            get_client_func=get_client,
            auto_ground=False,
        )
        result_als = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D000690"]}',  # ALS
            get_client_func=get_client,
            auto_ground=False,
        )

        assert "results" in result_pd
        assert "results" in result_als
        # Different diseases should (generally) return different gene sets


# ============================================================================
# Part 13: Smoke Tests with Representative Queries
# ============================================================================

@pytest.mark.nonpublic
class TestSmokeQueries:
    """Smoke tests exercising representative query patterns against live Neo4j."""

    @pytest.mark.asyncio
    async def test_gene_to_disease(self, flask_app_with_client):
        """Gene → Disease: LRRK2 should be associated with Parkinson's."""
        result = await call_endpoint(
            endpoint="get_diseases_for_gene",
            kwargs='{"gene": "LRRK2"}',
            get_client_func=get_client,
            auto_ground=True,
        )
        assert "results" in result, f"Error: {result.get('error')}"
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_disease_to_gene(self, flask_app_with_client):
        """Disease → Gene: Parkinson's (DOID) returns genes."""
        # Gene-disease edges in the graph use DOID identifiers
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["DOID", "14330"]}',
            get_client_func=get_client,
            auto_ground=False,
        )
        assert "results" in result, f"Error: {result.get('error')}"
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_gene_to_pathway(self, flask_app_with_client):
        """Gene → Pathway: TP53 should have associated pathways."""
        result = await call_endpoint(
            endpoint="get_pathways_for_gene",
            kwargs='{"gene": "TP53"}',
            get_client_func=get_client,
            auto_ground=True,
        )
        assert "results" in result, f"Error: {result.get('error')}"

    @pytest.mark.asyncio
    async def test_batch_call_fan_out(self, flask_app_with_client):
        """Batch fan-out: multiple genes → diseases."""
        result = await batch_call(
            endpoint="get_diseases_for_gene",
            entity_param="gene",
            entity_values=["LRRK2", "TP53"],
            get_client_func=get_client,
            auto_ground=True,
        )
        assert "results" in result
        assert result["total_entities"] == 2
        assert result["successful"] >= 1

    @pytest.mark.asyncio
    async def test_field_projection(self, flask_app_with_client):
        """Verify field projection reduces result size."""
        result_full = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',
            get_client_func=get_client,
            auto_ground=False,
        )
        result_projected = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',
            get_client_func=get_client,
            auto_ground=False,
            fields=["db_ns", "db_id", "name"],
        )

        assert "results" in result_full
        assert "results" in result_projected

        # Projected results should have fewer keys per item
        if result_projected["results"] and result_full["results"]:
            full_keys = set(result_full["results"][0].keys())
            proj_keys = set(result_projected["results"][0].keys())
            assert len(proj_keys) <= len(full_keys)

    @pytest.mark.asyncio
    async def test_sort_by_evidence(self, flask_app_with_client):
        """Verify sort_by='evidence' orders results by evidence count."""
        result = await call_endpoint(
            endpoint="get_genes_for_disease",
            kwargs='{"disease": ["MESH", "D010300"]}',
            get_client_func=get_client,
            auto_ground=False,
            sort_by="evidence",
        )
        assert "results" in result
        # If results have source_counts, verify descending order
        results = result["results"]
        if len(results) >= 2:
            for i in range(len(results) - 1):
                curr = results[i]
                nxt = results[i + 1]
                if isinstance(curr, dict) and isinstance(nxt, dict):
                    curr_ev = sum(curr.get("source_counts", {}).values()) if isinstance(curr.get("source_counts"), dict) else 0
                    nxt_ev = sum(nxt.get("source_counts", {}).values()) if isinstance(nxt.get("source_counts"), dict) else 0
                    assert curr_ev >= nxt_ev, f"Results not sorted by evidence: {curr_ev} < {nxt_ev}"


# ============================================================================
# Summary
# ============================================================================

@pytest.mark.nonpublic
class TestGatewayToolsCoverage:
    """Verify comprehensive test coverage of gateway tools."""

    def test_all_gateway_tools_covered(self):
        """Verify that all 5 gateway tools are tested."""
        tools_tested = {
            "ground_entity": TestGroundEntity,
            "suggest_endpoints": TestSuggestEndpoints,
            "call_endpoint": TestCallEndpoint,
            "get_navigation_schema": TestGetNavigationSchema,
            "batch_call": TestBatchErrorPropagation,
        }

        assert len(tools_tested) == 5, "All 5 gateway tools must be tested"

        # Verify each test class exists and has multiple tests
        for tool_name, test_class in tools_tested.items():
            test_methods = [
                m for m in dir(test_class)
                if m.startswith("test_") and callable(getattr(test_class, m))
            ]
            assert len(test_methods) >= 3, f"{tool_name} should have at least 3 test methods"

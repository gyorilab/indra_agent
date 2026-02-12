"""Entity type mappings and namespace filters for MCP gateway.

These constants define how entity types are recognized and filtered during
grounding and navigation operations in the autoclient gateway tools.
"""

# Entity type mappings from parameter names to canonical types
# (Aligned with schema_builder pattern from commit a6081263)
ENTITY_TYPE_MAPPINGS = {
    'disease': 'Disease',
    'gene': 'Gene',
    'genes': 'Gene',
    'drug': 'Drug',
    'drugs': 'Drug',
    'pathway': 'Pathway',
    'phenotype': 'Phenotype',
    'variant': 'Variant',
    'target': 'Protein',
    'targets': 'Protein',
    'protein': 'Protein',
    'tissue': 'Tissue',
    'cell_line': 'CellLine',
    'cell_type': 'CellType',
    'marker': 'Gene',  # Markers are typically genes
    'go_term': 'GOTerm',
    'trial': 'ClinicalTrial',
    'side_effect': 'SideEffect',
    'mesh_term': 'MeSHTerm',
    'paper_term': 'Publication',
    'pmid_term': 'Publication',
    'pmids': 'Publication',
    'project': 'ResearchProject',
    'publication': 'Publication',
    'journal': 'Journal',
    'publisher': 'Publisher',
    'domain': 'ProteinDomain',
    'enzyme': 'EnzymeActivity',
    'indication': 'Disease',
    'molecule': 'Drug',
}

# Map CURIE prefixes to entity types for NAVIGATION
# CRITICAL: Map to the types used in function parameters, not graph labels
# e.g., MESH diseases should map to Disease (param type), not MeSHTerm (graph label)
CURIE_PREFIX_TO_ENTITY = {
    # Genes
    'hgnc': 'Gene',
    'ncbigene': 'Gene',
    'ensembl': 'Gene',
    # Diseases - MESH/DOID are used for disease params
    'mesh': 'Disease',  # MESH diseases map to Disease for navigation
    'doid': 'Disease',
    'mondo': 'Disease',
    'efo': 'Disease',  # EFO can be disease or cell line - default to disease
    'hp': 'Phenotype',
    'orphanet': 'Disease',
    # Drugs
    'chebi': 'Drug',
    'chembl': 'Drug',
    'chembl.compound': 'Drug',
    'pubchem.compound': 'Drug',
    'drugbank': 'Drug',
    # Pathways
    'go': 'GOTerm',
    'reactome': 'Pathway',
    'wikipathways': 'Pathway',
    'kegg.pathway': 'Pathway',
    # Other
    'uberon': 'Tissue',
    'cl': 'CellType',
    'pubmed': 'Publication',
    'pmid': 'Publication',
    'clinicaltrials': 'ClinicalTrial',
    'nct': 'ClinicalTrial',
    'dbsnp': 'Variant',
    'interpro': 'ProteinDomain',
    'ec-code': 'EnzymeActivity',
    'uniprot': 'Protein',
}

# Parameter semantics: map parameter names to valid GILDA namespaces
# This filters grounding results to appropriate entity types
PARAM_NAMESPACE_FILTERS = {
    'disease': {'mesh', 'doid', 'efo', 'mondo', 'hp', 'orphanet', 'umls'},
    'gene': {'hgnc', 'ncbigene', 'ensembl', 'uniprot', 'fplx'},
    'genes': {'hgnc', 'ncbigene', 'ensembl', 'uniprot', 'fplx'},
    'drug': {'chebi', 'drugbank', 'pubchem.compound', 'chembl.compound', 'chembl'},
    'drugs': {'chebi', 'drugbank', 'pubchem.compound', 'chembl.compound', 'chembl'},
    'target': {'hgnc', 'uniprot', 'ncbigene'},  # Protein targets
    'targets': {'hgnc', 'uniprot', 'ncbigene'},
    'pathway': {'reactome', 'wikipathways', 'kegg.pathway', 'go'},
    'go_term': {'go'},
    'phenotype': {'hp', 'mesh', 'efo'},
    'tissue': {'uberon', 'bto'},
    'cell_type': {'cl'},
    'cell_line': {'efo', 'cellosaurus', 'ccle'},
    'variant': {'dbsnp', 'clinvar'},
    'mesh_term': {'mesh'},
    'side_effect': {'umls', 'mesh'},
    'indication': {'mesh', 'doid', 'efo', 'mondo'},  # Same as disease
    'molecule': {'chebi', 'drugbank', 'pubchem.compound', 'chembl'},  # Same as drug
}

def _build_organism_map() -> dict:
    """Build organism name → taxonomy ID mapping from gilda's canonical data.

    Layers (in order of construction):
    1. Latin names from gilda.resources.organism_labels (inverted)
    2. Common English names validated against NCBI Entrez Taxonomy API

    Anything not in this local map is resolved at runtime via NCBI Entrez
    (see _normalize_organism in autoclient_tools.py).
    """
    mapping = {}

    # Layer 1: Invert gilda's canonical taxonomy_id → Latin name mapping
    # These are authoritative — gilda uses them for organism filtering.
    try:
        from gilda.resources import organism_labels
        for tax_id, latin_name in organism_labels.items():
            mapping[latin_name.lower()] = tax_id
    except ImportError:
        pass

    # Layer 2: Common English names validated against NCBI Entrez (2026-02-08).
    # Only includes names where NCBI returns the exact expected taxonomy ID.
    # All other common names (fly, worm, yeast, etc.) are intentionally
    # omitted — they are ambiguous and NCBI cannot resolve them to the
    # specific model organism strain. The NCBI Entrez fallback in
    # _normalize_organism handles novel/formal names at runtime.
    _validated_common_names = {
        'human': '9606',       # NCBI-validated
        'mouse': '10090',      # NCBI-validated
        'rat': '10116',        # NCBI-validated
        'cow': '9913',         # NCBI-validated
        'zebrafish': '7955',   # NCBI-validated
        'thale cress': '3702', # NCBI-validated
    }
    mapping.update(_validated_common_names)

    return mapping


# Gilda requires taxonomy IDs (e.g., "9606"), not names (e.g., "human").
# Passing unrecognized strings silently drops all organism-specific results.
# Built from gilda.resources.organism_labels + NCBI-validated common names.
# Unknown names fall through to NCBI Entrez API at runtime.
ORGANISM_TO_TAXONOMY_ID = _build_organism_map()

# Ambiguity detection thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5  # Absolute: top score must be >= this
AMBIGUITY_SCORE_THRESHOLD = 0.3  # Relative: no result in top 5 within this of top

__all__ = [
    'ENTITY_TYPE_MAPPINGS',
    'CURIE_PREFIX_TO_ENTITY',
    'PARAM_NAMESPACE_FILTERS',
    'MIN_CONFIDENCE_THRESHOLD',
    'AMBIGUITY_SCORE_THRESHOLD',
    'ORGANISM_TO_TAXONOMY_ID',
]

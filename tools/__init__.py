"""
DERMA-Agent Tools Package
=========================

This package contains tools for:
- Knowledge Fabric (medical knowledge graph)
- Enhanced Data Client (TCGA data access)
- Enhanced Clinical Statistics (survival analysis, ML)
- Enhanced Pathology (image segmentation)
- Legacy tools (original GDC client, stats, pathology)
"""

# Enhanced components
from .knowledge_fabric import (
    KnowledgeFabric,
    Node,
    Edge,
    MedicalKnowledgeBuilder,
    create_default_knowledge_fabric
)

from .enhanced_data_client import (
    get_data_client,
    EnhancedGDCClient,
    MultiSourceDataClient,
    EXPANDED_CANCER_PROJECTS
)

from .enhanced_clinical_stats import (
    EnhancedStatsEngine,
    SurvivalAnalysisResult,
    quick_survival_summary
)

from .enhanced_pathology import (
    EnhancedWSIProcessor,
    NucleiSegmenter,
    TissueAnalyzer,
    PathologyFeatures,
    create_synthetic_pathology_image,
    analyze_wsi_path
)

# Legacy components (for backward compatibility)
try:
    from .gdc_client import fetch_tcga_clinical_data, CANCER_PROJECTS
    from .clinical_stats import execute_codeact
    from .pathology_utils import generate_saliency_heatmap, segment_nuclei
except ImportError:
    # Legacy components may not be available
    pass

__version__ = "2.0.0"
__author__ = "DERMA-Agent Team"

__all__ = [
    # Knowledge Fabric
    "KnowledgeFabric",
    "Node",
    "Edge",
    "MedicalKnowledgeBuilder",
    "create_default_knowledge_fabric",
    
    # Enhanced Data Client
    "get_data_client",
    "EnhancedGDCClient",
    "MultiSourceDataClient",
    "EXPANDED_CANCER_PROJECTS",
    
    # Enhanced Clinical Stats
    "EnhancedStatsEngine",
    "SurvivalAnalysisResult",
    "quick_survival_summary",
    
    # Enhanced Pathology
    "EnhancedWSIProcessor",
    "NucleiSegmenter",
    "TissueAnalyzer",
    "PathologyFeatures",
    "create_synthetic_pathology_image",
    "analyze_wsi_path",
]

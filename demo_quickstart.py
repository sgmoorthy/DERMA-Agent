#!/usr/bin/env python3
"""
DERMA-Agent Quick Start Demo
============================

This script demonstrates the enhanced DERMA-Agent capabilities:
1. Knowledge Fabric - Medical knowledge graph
2. Fast Discovery Engine - Parallel hypothesis testing
3. Enhanced Data Access - TCGA clinical data
4. Advanced Pathology - Image segmentation and analysis

Usage:
    python demo_quickstart.py

Requirements:
    - Python 3.10+
    - Dependencies in requirements.txt
    - OpenAI API key (optional, will use mock mode if not set)
"""

import os
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

# Set mock mode if no API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  No OPENAI_API_KEY found. Running in MOCK MODE (no LLM calls)")
    print("   Set OPENAI_API_KEY environment variable for full functionality\n")


def demo_knowledge_fabric():
    """Demo 1: Knowledge Fabric - Medical Knowledge Graph"""
    print("\n" + "="*60)
    print("🔬 DEMO 1: Knowledge Fabric - Medical Knowledge Graph")
    print("="*60)
    
    from tools.knowledge_fabric import create_default_knowledge_fabric
    
    # Create or load knowledge fabric
    kg_path = Path("data/knowledge_fabric.json")
    kg = create_default_knowledge_fabric(str(kg_path))
    
    # Show statistics
    stats = kg.get_statistics()
    print(f"\n📊 Knowledge Graph Statistics:")
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Graph Density: {stats['density']:.4f}")
    
    print(f"\n📦 Node Types:")
    for node_type, count in stats['node_types'].items():
        print(f"   • {node_type}: {count}")
    
    # Query example
    print(f"\n🔍 Query: Drugs treating Melanoma")
    drugs = []
    for edge in kg.edges:
        if edge.relation == "TREATS" and edge.target == "Melanoma":
            drug_node = kg.get_node(edge.source)
            if drug_node:
                drugs.append((drug_node.id, edge.properties))
    
    for drug, props in drugs:
        print(f"   • {drug}: response_rate={props.get('response_rate', 'N/A')}")
    
    print("\n✅ Knowledge Fabric demo complete!")
    return kg


def demo_data_client():
    """Demo 2: Enhanced Data Client - TCGA Data Access"""
    print("\n" + "="*60)
    print("📊 DEMO 2: Enhanced Data Client - TCGA Data Access")
    print("="*60)
    
    from tools.enhanced_data_client import get_data_client
    
    client = get_data_client()
    
    print("\n📥 Fetching Breast Cancer (TCGA-BRCA) data...")
    df = client.get_survival_analysis_ready_data("TCGA-BRCA")
    df = client.enrich_with_derived_features(df)
    
    print(f"\n📋 Data Summary:")
    print(f"   Samples: {len(df)}")
    
    if 'event' in df.columns:
        events = df['event'].sum()
        print(f"   Events (deaths): {int(events)} ({100*events/len(df):.1f}%)")
    
    if 'age_years' in df.columns:
        print(f"   Age range: {df['age_years'].min():.1f} - {df['age_years'].max():.1f} years")
    
    if 'stage_group' in df.columns:
        print(f"\n📊 Stage Distribution:")
        for stage, count in df['stage_group'].value_counts().items():
            print(f"   • {stage}: {count} ({100*count/len(df):.1f}%)")
    
    print("\n✅ Data Client demo complete!")
    return df


def demo_pathology():
    """Demo 3: Enhanced Pathology - Image Analysis"""
    print("\n" + "="*60)
    print("🔬 DEMO 3: Enhanced Pathology - Image Analysis")
    print("="*60)
    
    from tools.enhanced_pathology import (
        create_synthetic_pathology_image, 
        NucleiSegmenter, 
        TissueAnalyzer
    )
    import numpy as np
    
    print("\n🖼️  Generating synthetic pathology images...")
    
    # Create synthetic images
    patterns = ['mixed', 'high_cellularity']
    segmenter = NucleiSegmenter()
    analyzer = TissueAnalyzer()
    
    for pattern in patterns:
        print(f"\n   Analyzing {pattern} pattern:")
        image = create_synthetic_pathology_image((256, 256), pattern)
        
        # Analyze
        features = analyzer.analyze_tile(image)
        mask = segmenter.segment_nuclei(image)
        
        print(f"      Nuclei count: {features.nuclei_count}")
        print(f"      Cellularity: {features.cellularity:.3f}")
        print(f"      Tissue area ratio: {features.tissue_area_ratio:.3f}")
        
        if features.texture_features:
            texture = features.texture_features
            print(f"      Texture (contrast): {texture.get('contrast', 0):.3f}")
    
    print("\n✅ Pathology demo complete!")


def demo_discovery(kg, df):
    """Demo 4: Fast Discovery Engine - Hypothesis Testing"""
    print("\n" + "="*60)
    print("🚀 DEMO 4: Fast Discovery Engine - Hypothesis Testing")
    print("="*60)
    
    from agents.discovery_engine import FastDiscoveryEngine, DiscoveryConfig
    
    # Configure for quick demo (1 hypothesis to save time)
    config = DiscoveryConfig(
        max_iterations=1,
        parallel_workers=1,
        hypothesis_per_cohort=1,  # Just 1 for quick demo
        significance_threshold=0.05,
        use_knowledge_fabric=True,
        auto_correct_errors=True,
        timeout_seconds=60
    )
    
    print("\n⚙️  Configuration:")
    print(f"   Parallel workers: {config.parallel_workers}")
    print(f"   Hypotheses per cohort: {config.hypothesis_per_cohort}")
    print(f"   Using knowledge fabric: {config.use_knowledge_fabric}")
    
    print("\n🔬 Running discovery on Breast Cancer...")
    print("   (This may take 30-60 seconds...)")
    
    try:
        engine = FastDiscoveryEngine(config, kg)
        results = engine.discover_single_cohort("TCGA-BRCA", "Breast Cancer")
        
        print(f"\n📊 Results:")
        print(f"   Hypotheses tested: {len(results)}")
        
        significant = [r for r in results if r.significant]
        print(f"   Significant findings: {len(significant)}")
        
        for i, result in enumerate(results, 1):
            print(f"\n   Hypothesis {i}:")
            print(f"      Statement: {result.hypothesis[:80]}...")
            print(f"      P-value: {result.p_value if result.p_value else 'N/A'}")
            print(f"      Significant: {'Yes' if result.significant else 'No'}")
            print(f"      Execution time: {result.execution_time:.2f}s")
        
        # Save results
        engine.save_ledger("discoveries/demo_ledger.json")
        print(f"\n💾 Results saved to: discoveries/demo_ledger.json")
        
    except Exception as e:
        print(f"\n⚠️  Discovery demo encountered an error: {e}")
        print("   This is expected if running in mock mode without API key")
    
    print("\n✅ Discovery demo complete!")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("🚀 DERMA-Agent Quick Start Demo")
    print("="*60)
    print("\nThis demo showcases the enhanced DERMA-Agent capabilities:")
    print("   1. Knowledge Fabric - Medical knowledge graph")
    print("   2. Enhanced Data Client - TCGA data access")
    print("   3. Advanced Pathology - Image segmentation")
    print("   4. Fast Discovery Engine - Parallel hypothesis testing")
    print("\n" + "="*60)
    
    try:
        # Run demos
        kg = demo_knowledge_fabric()
        df = demo_data_client()
        demo_pathology()
        demo_discovery(kg, df)
        
        # Summary
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("   1. Launch dashboard: streamlit run app_enhanced.py")
        print("   2. Explore notebook: jupyter notebook notebooks/comprehensive_demo.ipynb")
        print("   3. Run full discovery: python -c \"from agents.discovery_engine import run_fast_discovery; run_fast_discovery(['Breast Cancer', 'Skin Cancer'])\"")
        print("\nFor more info, see README.md")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

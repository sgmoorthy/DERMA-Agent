"""
Knowledge Fabric: Medical Knowledge Graph for DERMA-Agent
A Neo4j-style graph database implementation for storing and reasoning over
medical knowledge, cancer pathways, drug interactions, and clinical relationships.
"""

import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    label: str  # e.g., "Gene", "Protein", "Disease", "Drug", "Pathway", "Clinical_Feature"
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "properties": self.properties,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        return cls(
            id=data["id"],
            label=data["label"],
            properties=data.get("properties", {}),
            embedding=embedding
        )


@dataclass
class Edge:
    """Represents a relationship between nodes."""
    source: str
    target: str
    relation: str  # e.g., "ASSOCIATED_WITH", "TREATS", "INHIBITS", "EXPRESSED_IN"
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "properties": self.properties,
            "weight": self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        return cls(
            source=data["source"],
            target=data["target"],
            relation=data["relation"],
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0)
        )


class KnowledgeFabric:
    """
    Medical Knowledge Graph for reasoning about cancer pathology.
    Uses NetworkX for graph operations with support for embeddings and semantic search.
    """
    
    def __init__(self, name: str = "DERMA-KG"):
        self.name = name
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self.version = "1.0"
        self.data_sources = {
            "Gene": "NCBI / HGNC / COSMIC",
            "Disease": "MeSH / TCGA",
            "Drug": "DrugBank / FDA",
            "Pathway": "Reactome / KEGG",
            "Clinical_Feature": "TCGA / GDC"
        }
        
    def add_node(self, node: Node) -> None:
        """Add a node to the knowledge graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.properties, label=node.label)
        if node.embedding is not None:
            self._embedding_cache[node.id] = node.embedding
            
    def add_edge(self, edge: Edge) -> None:
        """Add a relationship between nodes."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source, 
            edge.target, 
            relation=edge.relation, 
            weight=edge.weight,
            **edge.properties
        )
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self.nodes.get(node_id)
        
    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[Node, Edge]]:
        """Get all neighbors of a node with their connecting edges."""
        if node_id not in self.graph:
            return []
            
        neighbors = []
        for neighbor_id in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor_id)
            if relation is None or edge_data.get("relation") == relation:
                if neighbor_id in self.nodes:
                    edge = Edge(
                        source=node_id,
                        target=neighbor_id,
                        relation=edge_data.get("relation"),
                        properties={k: v for k, v in edge_data.items() 
                                  if k not in ["relation", "weight"]},
                        weight=edge_data.get("weight", 1.0)
                    )
                    neighbors.append((self.nodes[neighbor_id], edge))
        return neighbors
    
    def find_path(self, source: str, target: str, max_depth: int = 4) -> Optional[List[str]]:
        """Find a path between two nodes using shortest path algorithm."""
        try:
            path = nx.shortest_path(self.graph, source, target)
            if len(path) <= max_depth + 1:
                return path
            return None
        except nx.NetworkXNoPath:
            return None
            
    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 5, 
                        node_type: Optional[str] = None) -> List[Tuple[Node, float]]:
        """Search for similar nodes using cosine similarity on embeddings."""
        similarities = []
        
        for node_id, embedding in self._embedding_cache.items():
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if node_type and node.label != node_type:
                continue
                
            # Cosine similarity
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((node, float(sim)))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def query_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the graph for patterns matching given criteria."""
        results = []
        
        for node_id, node in self.nodes.items():
            match = True
            for key, value in pattern.items():
                if key == "label":
                    if node.label != value:
                        match = False
                        break
                elif key in node.properties:
                    if node.properties[key] != value:
                        match = False
                        break
                else:
                    match = False
                    break
                    
            if match:
                results.append({
                    "node": node,
                    "neighbors": self.get_neighbors(node_id)
                })
                
        return results
    
    def subgraph(self, node_ids: List[str]) -> 'KnowledgeFabric':
        """Extract a subgraph containing only specified nodes."""
        subgraph = KnowledgeFabric(name=f"{self.name}-subgraph")
        
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
                
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge)
                
        return subgraph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_type_counts = defaultdict(int)
        relation_counts = defaultdict(int)
        
        for node in self.nodes.values():
            node_type_counts[node.label] += 1
            
        for edge in self.edges:
            relation_counts[edge.relation] += 1
            
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_type_counts),
            "relation_types": dict(relation_counts),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph) if len(self.graph) > 0 else False
        }
    
    def save(self, filepath: str) -> None:
        """Save the knowledge graph to a JSON file."""
        data = {
            "name": self.name,
            "version": getattr(self, "version", "1.0"),
            "data_sources": getattr(self, "data_sources", {}),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeFabric':
        """Load a knowledge graph from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        fabric = cls(name=data.get("name", "loaded-kg"))
        fabric.version = data.get("version", "1.0")
        fabric.data_sources = data.get("data_sources", {})
        
        for node_data in data.get("nodes", []):
            fabric.add_node(Node.from_dict(node_data))
            
        for edge_data in data.get("edges", []):
            fabric.add_edge(Edge.from_dict(edge_data))
            
        return fabric
    
    def find_scientific_pathways(self, start: str, end: str, allowed_relations: List[str] = None) -> List[List[Dict[str, Any]]]:
        """Find scientific pathways between two nodes in the graph, optionally filtering by allowed relation types."""
        if start not in self.nodes or end not in self.nodes:
            return []
        
        # Standard BFS to find all paths up to length 4
        paths = []
        queue = [[(start, None)]]
        
        while queue:
            path = queue.pop(0)
            current = path[-1][0]
            
            if len(path) > 4:
                continue
                
            if current == end:
                paths.append(path)
                continue
                
            if current in self.graph:
                for successor in self.graph.successors(current):
                    edge_data = self.graph.get_edge_data(current, successor)
                    relation = edge_data.get("relation")
                    
                    if allowed_relations and relation not in allowed_relations:
                        continue
                        
                    # Avoid loops
                    visited = [step[0] for step in path]
                    if successor not in visited:
                        new_path = list(path)
                        new_path.append((successor, relation))
                        queue.append(new_path)
                     
        # Format paths nicely
        formatted_paths = []
        for p in paths:
            formatted_path = []
            for i in range(len(p) - 1):
                node_a = self.nodes.get(p[i][0])
                node_b = self.nodes.get(p[i+1][0])
                relation = p[i+1][1]
                formatted_path.append({
                    "source": node_a,
                    "relation": relation,
                    "target": node_b
                })
            formatted_paths.append(formatted_path)
            
        return formatted_paths

    def calculate_hypothesis_prior_score(self, hypothesis: str) -> Dict[str, Any]:
        """Statically analyze a hypothesis to find supporting links in the KG and calculate a confidence score."""
        hyp_lower = hypothesis.lower()
        
        # Identify genes in the hypothesis
        found_genes = []
        for nid, node in self.nodes.items():
            if node.label == "Gene" and nid.lower() in hyp_lower:
                found_genes.append(nid)
                 
        # Identify cancers/diseases in the hypothesis
        found_diseases = []
        for nid, node in self.nodes.items():
            if node.label == "Disease" and (nid.lower() in hyp_lower or node.properties.get("name", "").lower() in hyp_lower):
                found_diseases.append(nid)
                 
        if not found_genes or not found_diseases:
            return {
                "score": 0.3, # default baseline prior for novel hypothesis
                "evidence_path": "No matching gene-disease path found in knowledge fabric",
                "strategy": "Novel hypothesis (unsupported by graph prior)"
            }
             
        best_score = 0.3
        best_path_str = ""
        best_strategy = "Novel hypothesis"
         
        # Check all gene-disease pairs
        for gene in found_genes:
            for disease in found_diseases:
                # Check direct edge first
                if self.graph.has_edge(gene, disease):
                     edge_data = self.graph.get_edge_data(gene, disease)
                     rel = edge_data.get("relation", "ASSOCIATED_WITH")
                     score = 0.9 if rel in ["MUTATED_IN", "PROGNOSTIC_FACTOR_FOR"] else 0.7
                     if score > best_score:
                         best_score = score
                         best_path_str = f"{gene} -[{rel}]-> {disease}"
                         best_strategy = f"Direct clinical connection: {rel}"
                         
                # Check paths of length up to 3
                paths = self.find_scientific_pathways(gene, disease)
                for path in paths:
                     path_len = len(path)
                     score = 0.8 if path_len == 2 else (0.6 if path_len == 3 else 0.4)
                     if score > best_score:
                         best_score = score
                         parts = []
                         for step in path:
                             parts.append(f"{step['source'].id} -[{step['relation']}]-> {step['target'].id}")
                         best_path_str = " | ".join(parts)
                         best_strategy = f"Indirect connection of depth {path_len}"
                         
        return {
            "score": float(best_score),
            "evidence_path": best_path_str or "No matching path",
            "strategy": best_strategy
        }

    def constrained_walk(self, start_node: str, relation_sequence: List[str]) -> List[List[Dict[str, Any]]]:
        """Walk the graph starting from start_node following the specified relation sequence."""
        if start_node not in self.nodes:
            return []
            
        current_paths = [[(start_node, None)]]
        
        for rel in relation_sequence:
            next_paths = []
            for path in current_paths:
                curr_node_id = path[-1][0]
                if curr_node_id not in self.graph:
                    continue
                for successor_id in self.graph.successors(curr_node_id):
                    edge_data = self.graph.get_edge_data(curr_node_id, successor_id)
                    if edge_data.get("relation") == rel:
                        # Avoid cycles
                        visited = [step[0] for step in path]
                        if successor_id not in visited:
                            new_path = list(path)
                            new_path.append((successor_id, rel))
                            next_paths.append(new_path)
            current_paths = next_paths
            if not current_paths:
                break
                
        # Format the paths
        formatted_paths = []
        for path in current_paths:
            formatted_path = []
            for i in range(len(path) - 1):
                node_a = self.nodes.get(path[i][0])
                node_b = self.nodes.get(path[i+1][0])
                relation = path[i+1][1]
                formatted_path.append({
                    "source": node_a,
                    "relation": relation,
                    "target": node_b
                })
            formatted_paths.append(formatted_path)
            
        return formatted_paths

    def compute_confidence_score(self, path: List[Dict[str, Any]]) -> float:
        """Compute prior confidence score (0.0 to 1.0) based on cumulative edge weights and path length."""
        if not path:
            return 0.0
            
        score = 1.0
        for step in path:
            source_id = step["source"].id
            target_id = step["target"].id
            relation = step["relation"]
            
            edge_data = self.graph.get_edge_data(source_id, target_id)
            weight = 1.0
            if edge_data:
                weight = edge_data.get("weight", 1.0)
            
            # Additional discounts based on relation significance
            discount = 1.0
            if relation in ["MUTATED_IN", "PREDICTS_RESPONSE_TO", "PROGNOSTIC_FACTOR_FOR"]:
                discount = 1.0
            elif relation in ["TREATS", "TARGETS"]:
                discount = 0.9
            else:
                discount = 0.7
                
            score *= (weight * discount)
            
        # Discount by path length (longer paths have lower prior confidence)
        length_penalty = 1.0 / (len(path) ** 0.5)
        return float(max(0.0, min(1.0, score * length_penalty)))

    def get_kg_metadata(self) -> Dict[str, Any]:
        """Return metadata block for the knowledge graph."""
        from datetime import datetime
        return {
            "version": getattr(self, "version", "1.0"),
            "data_sources": getattr(self, "data_sources", {}),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "timestamp": datetime.now().isoformat()
        }
    
    def to_networkx(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph for advanced operations."""
        return self.graph


class MedicalKnowledgeBuilder:
    """Builder class for constructing medical knowledge graphs from various sources."""
    
    @staticmethod
    def build_oncology_knowledge_base() -> KnowledgeFabric:
        """Build a comprehensive oncology knowledge base."""
        kg = KnowledgeFabric(name="Oncology-KG")
        
        # Cancer Types
        cancers = [
            ("Melanoma", "Skin Cancer", {"stage_distribution": [0.15, 0.25, 0.35, 0.25]}),
            ("Breast_Carcinoma", "Breast Cancer", {"subtypes": ["Luminal A", "Luminal B", "HER2+", "Triple Negative"]}),
            ("Lung_Adenocarcinoma", "Lung Cancer", {"driver_mutations": ["EGFR", "KRAS", "ALK"]}),
            ("Glioblastoma", "Brain Cancer", {"grade": "IV", "median_survival_months": 15}),
            ("Colorectal_Cancer", "Colorectal Cancer", {"msi_status": "variable"}),
            ("Prostate_Cancer", "Prostate Cancer", {"psa_levels": "variable"}),
        ]
        
        for cancer_id, name, props in cancers:
            kg.add_node(Node(
                id=cancer_id,
                label="Disease",
                properties={"name": name, **props}
            ))
        
        # Key Genes/Proteins
        genes = [
            ("TP53", "Tumor Suppressor", {"chromosome": "17p13.1", "function": "DNA repair"}),
            ("BRCA1", "Tumor Suppressor", {"chromosome": "17q21", "function": "DNA repair"}),
            ("EGFR", "Receptor Tyrosine Kinase", {"chromosome": "7p12", "function": "Cell proliferation"}),
            ("KRAS", "GTPase", {"chromosome": "12p12.1", "function": "Signal transduction"}),
            ("BRAF", "Serine/Threonine Kinase", {"chromosome": "7q34", "function": "MAPK signaling"}),
            ("PIK3CA", "Lipid Kinase", {"chromosome": "3q26.3", "function": "PI3K/AKT pathway"}),
            ("CDKN2A", "Tumor Suppressor", {"chromosome": "9p21", "function": "Cell cycle"}),
            ("PTEN", "Tumor Suppressor", {"chromosome": "10q23.3", "function": "PI3K/AKT negative regulator"}),
            ("MYC", "Transcription Factor", {"chromosome": "8q24", "function": "Cell proliferation"}),
            ("CDH1", "Cell Adhesion", {"chromosome": "16q22.1", "function": "E-cadherin"}),
        ]
        
        for gene_id, gene_type, props in genes:
            kg.add_node(Node(
                id=gene_id,
                label="Gene",
                properties={"type": gene_type, **props}
            ))
        
        # Drugs/Therapies
        drugs = [
            (" Pembrolizumab", "Immunotherapy", {"target": "PD-1", "indications": ["Melanoma", "NSCLC"]}),
            ("Trastuzumab", "Targeted Therapy", {"target": "HER2", "indications": ["Breast_Carcinoma"]}),
            ("Osimertinib", "Targeted Therapy", {"target": "EGFR T790M", "indications": ["Lung_Adenocarcinoma"]}),
            ("Dabrafenib", "Targeted Therapy", {"target": "BRAF V600E", "indications": ["Melanoma"]}),
            ("Temozolomide", "Chemotherapy", {"target": "DNA alkylating", "indications": ["Glioblastoma"]}),
            ("Cisplatin", "Chemotherapy", {"target": "DNA crosslinking", "indications": ["Lung_Adenocarcinoma"]}),
            ("Carboplatin", "Chemotherapy", {"target": "DNA crosslinking", "indications": ["Lung_Adenocarcinoma", "Ovarian"]}),
            ("Bevacizumab", "Targeted Therapy", {"target": "VEGF", "indications": ["Colorectal_Cancer", "Glioblastoma"]}),
        ]
        
        for drug_id, drug_type, props in drugs:
            kg.add_node(Node(
                id=drug_id.strip(),
                label="Drug",
                properties={"type": drug_type, **props}
            ))
        
        # Clinical Features
        features = [
            ("Tumor_Stage", "Clinical_Feature", {"categories": ["I", "II", "III", "IV"]}),
            ("Lymph_Node_Status", "Clinical_Feature", {"categories": ["N0", "N1", "N2", "N3"]}),
            ("Metastasis_Status", "Clinical_Feature", {"categories": ["M0", "M1"]}),
            ("Tumor_Mutational_Burden", "Clinical_Feature", {"type": "continuous", "unit": "mutations/Mb"}),
            ("PD_L1_Expression", "Clinical_Feature", {"type": "continuous", "threshold": 1.0}),
            ("Microsatellite_Instability", "Clinical_Feature", {"categories": ["MSS", "MSI-L", "MSI-H"]}),
            ("Ki67_Index", "Clinical_Feature", {"type": "continuous", "unit": "percentage"}),
            ("Lymphocyte_Infiltration", "Clinical_Feature", {"type": "continuous", "description": "Density of tumor-infiltrating lymphocytes"}),
        ]
        
        for feature_id, feature_type, props in features:
            kg.add_node(Node(
                id=feature_id,
                label="Clinical_Feature",
                properties={"type": feature_type, **props}
            ))
        
        # Pathways
        pathways = [
            ("MAPK_Pathway", "Signaling_Pathway", {"genes": ["BRAF", "KRAS", "EGFR"]}),
            ("PI3K_AKT_mTOR", "Signaling_Pathway", {"genes": ["PIK3CA", "PTEN"]}),
            ("DNA_Repair_Pathway", "Signaling_Pathway", {"genes": ["BRCA1", "TP53"]}),
            ("Cell_Cycle_Control", "Signaling_Pathway", {"genes": ["CDKN2A", "MYC"]}),
            ("Immune_Checkpoint", "Signaling_Pathway", {"genes": ["PDCD1", "CD274"]}),
        ]
        
        for pathway_id, pathway_type, props in pathways:
            kg.add_node(Node(
                id=pathway_id,
                label="Pathway",
                properties={"type": pathway_type, **props}
            ))
        
        # Add relationships
        relationships = [
            # Gene-Cancer associations
            ("BRAF", "Melanoma", "MUTATED_IN", {"frequency": 0.45, "prognosis": "variable"}),
            ("BRCA1", "Breast_Carcinoma", "MUTATED_IN", {"frequency": 0.05, "risk_increase": "high"}),
            ("EGFR", "Lung_Adenocarcinoma", "MUTATED_IN", {"frequency": 0.15, "subtype": "adenocarcinoma"}),
            ("TP53", "Glioblastoma", "MUTATED_IN", {"frequency": 0.30, "prognosis": "poor"}),
            ("KRAS", "Lung_Adenocarcinoma", "MUTATED_IN", {"frequency": 0.25, "prognosis": "resistance_to_EGFR_TKI"}),
            ("PIK3CA", "Breast_Carcinoma", "MUTATED_IN", {"frequency": 0.30, "prognosis": "resistance_to_endocrine"}),
            
            # Drug-Target relationships
            ("Pembrolizumab", "Immune_Checkpoint", "TARGETS", {"mechanism": "PD-1 inhibition"}),
            ("Trastuzumab", "HER2", "TARGETS", {"mechanism": "antibody"}),
            ("Osimertinib", "EGFR", "TARGETS", {"mechanism": "TKI"}),
            ("Dabrafenib", "BRAF", "TARGETS", {"mechanism": "BRAF inhibitor"}),
            ("Temozolomide", "DNA_Repair_Pathway", "AFFECTS", {"mechanism": "DNA alkylation"}),
            
            # Drug-Cancer indications
            ("Pembrolizumab", "Melanoma", "TREATS", {"line": "first", "response_rate": 0.35}),
            ("Pembrolizumab", "Lung_Adenocarcinoma", "TREATS", {"line": "first", "response_rate": 0.45}),
            ("Trastuzumab", "Breast_Carcinoma", "TREATS", {"line": "first", "response_rate": 0.60}),
            ("Osimertinib", "Lung_Adenocarcinoma", "TREATS", {"line": "second", "response_rate": 0.70}),
            ("Dabrafenib", "Melanoma", "TREATS", {"line": "first", "response_rate": 0.55}),
            ("Temozolomide", "Glioblastoma", "TREATS", {"line": "first", "response_rate": 0.40}),
            
            # Gene-Pathway memberships
            ("BRAF", "MAPK_Pathway", "PART_OF", {}),
            ("KRAS", "MAPK_Pathway", "PART_OF", {}),
            ("EGFR", "MAPK_Pathway", "PART_OF", {}),
            ("PIK3CA", "PI3K_AKT_mTOR", "PART_OF", {}),
            ("PTEN", "PI3K_AKT_mTOR", "NEGATIVE_REGULATOR_OF", {}),
            ("BRCA1", "DNA_Repair_Pathway", "PART_OF", {}),
            ("TP53", "DNA_Repair_Pathway", "PART_OF", {}),
            ("CDKN2A", "Cell_Cycle_Control", "PART_OF", {}),
            
            # Clinical feature associations
            ("Tumor_Mutational_Burden", "Pembrolizumab", "PREDICTS_RESPONSE_TO", {"threshold": 10}),
            ("PD_L1_Expression", "Pembrolizumab", "PREDICTS_RESPONSE_TO", {"threshold": 1.0}),
            ("Microsatellite_Instability", "Pembrolizumab", "PREDICTS_RESPONSE_TO", {"type": "MSI-H"}),
            ("Lymphocyte_Infiltration", "Melanoma", "PROGNOSTIC_FACTOR_FOR", {"high": "better_survival"}),
            ("Ki67_Index", "Breast_Carcinoma", "PROGNOSTIC_FACTOR_FOR", {"high": "aggressive"}),
        ]
        
        for source, target, relation, props in relationships:
            kg.add_edge(Edge(
                source=source.strip() if isinstance(source, str) else source,
                target=target.strip() if isinstance(target, str) else target,
                relation=relation,
                properties=props
            ))
        
        return kg
    
    @staticmethod
    def enrich_with_tcga_data(kg: KnowledgeFabric, clinical_df: pd.DataFrame, 
                              cancer_type: str) -> KnowledgeFabric:
        """Enrich knowledge graph with TCGA clinical data patterns."""
        # Add derived clinical patterns
        if 'vital_status' in clinical_df.columns:
            survival_rate = (clinical_df['vital_status'] == 'Alive').mean()
            kg.add_node(Node(
                id=f"{cancer_type}_Survival_Pattern",
                label="Clinical_Pattern",
                properties={
                    "cancer_type": cancer_type,
                    "survival_rate": float(survival_rate),
                    "sample_size": len(clinical_df)
                }
            ))
            
            # Connect to cancer node
            kg.add_edge(Edge(
                source=f"{cancer_type}_Survival_Pattern",
                target=cancer_type.replace(" ", "_"),
                relation="OBSERVED_IN",
                properties={"data_source": "TCGA"}
            ))
        
        return kg


def create_default_knowledge_fabric(save_path: Optional[str] = None) -> KnowledgeFabric:
    """Create and optionally save the default oncology knowledge fabric."""
    kg = MedicalKnowledgeBuilder.build_oncology_knowledge_base()
    
    if save_path:
        kg.save(save_path)
        
    return kg


if __name__ == "__main__":
    # Test the knowledge fabric
    print("Building Oncology Knowledge Fabric...")
    kg = create_default_knowledge_fabric()
    
    stats = kg.get_statistics()
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")
    print(f"  Node Types: {stats['node_types']}")
    print(f"  Relation Types: {stats['relation_types']}")
    
    # Test queries
    print("\n--- Query: Genes mutated in Melanoma ---")
    melanoma_genes = kg.query_pattern({"label": "Gene"})
    for result in melanoma_genes[:3]:
        node = result["node"]
        neighbors = result["neighbors"]
        for neighbor, edge in neighbors:
            if edge.relation == "MUTATED_IN" and "Melanoma" in edge.target:
                print(f"  {node.id}: {edge.properties}")
    
    print("\n--- Find drugs treating Melanoma ---")
    for edge in kg.edges:
        if edge.relation == "TREATS" and edge.target == "Melanoma":
            drug_node = kg.get_node(edge.source)
            if drug_node:
                print(f"  {drug_node.id}: {edge.properties}")
    
    # Save to file
    kg.save("data/knowledge_fabric.json")
    print("\nKnowledge Fabric saved to data/knowledge_fabric.json")

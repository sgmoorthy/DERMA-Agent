import networkx as nx

class KnowledgeFabric:
    def __init__(self):
        self.graph = nx.Graph()
        # Seed default knowledge nodes
        self.add_relationship("BRAF", "MAPK_Pathway", "PART_OF")
        self.add_relationship("MAPK_Pathway", "Melanoma", "ASSOCIATED_WITH")
        self.add_relationship("Melanoma", "Dabrafenib", "TREATS")
        self.add_relationship("Dabrafenib", "BRAF", "TARGETS")
        
    def add_relationship(self, entity1, entity2, relation_type):
        self.graph.add_edge(entity1, entity2, relation=relation_type)
        
    def query_context(self, entity):
        if entity in self.graph:
            return list(self.graph.neighbors(entity))
        return []

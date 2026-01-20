
"""
Document chunking module with hierarchical parsing and summary generation.
"""
from typing import List, Dict
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import TextNode, NodeRelationship
from config import settings

class DocumentChunker:
    """Handles hierarchical document chunking with context summaries."""
    
    def __init__(self, llm):
        """
        Initialize document chunker.
        
        Args:
            llm: Language model for generating summaries
        """
        self.llm = llm
        self.chunk_sizes = settings.CHUNK_SIZES
        
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )
        
        print(f"✓ Document chunker initialized (chunk sizes: {self.chunk_sizes})")



    def create_nodes(self, documents: List[Document]) -> List:
        """
        Parse documents into hierarchical nodes.
        
        Args:
            documents: List of documents to parse
            
        Returns:
            List of all nodes (parent and leaf)
        """
        print(f"Creating hierarchical nodes...")
        nodes = self.parser.get_nodes_from_documents(documents)
        
        leaf_count = len(get_leaf_nodes(nodes))
        parent_count = len(nodes) - leaf_count
        
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Parent nodes: {parent_count}")
        print(f"  Leaf nodes: {leaf_count}")
        
        return nodes
    

    def generate_parent_summaries(self, nodes: List) -> Dict[str, str]:
        """
        Generate concise summaries for parent nodes.
        
        Args:
            nodes: List of all nodes
            
        Returns:
            Dictionary mapping node_id to summary
        """
        summaries = {}
        parent_nodes = [n for n in nodes if NodeRelationship.CHILD in n.relationships]
        
        if not parent_nodes:
            print("  No parent nodes to summarize")
            return summaries
        
        print(f"Generating summaries for {len(parent_nodes)} parent nodes...")
        
        for i, node in enumerate(parent_nodes):
            if i > 0 and i % 10 == 0:
                print(f"  Progress: {i}/{len(parent_nodes)}")
            
            prompt = f"""Provide a concise summary (2-3 sentences, max 100 tokens) of this text section: {node.get_content()[:3000]} Summary:"""
            
            try:
                response = self.llm.complete(prompt)
                summaries[node.node_id] = response.text.strip()
            except Exception as e:
                print(f"  Warning: Failed to generate summary for node {node.node_id}: {e}")
                summaries[node.node_id] = node.get_content()[:150] + "..."
        
        print(f"✓ Generated {len(summaries)} summaries")
        return summaries
    

    def enrich_leaf_nodes(self, nodes: List, parent_summaries: Dict[str, str]) -> List[TextNode]:
        """
        Add parent context to leaf nodes for better retrieval.
        
        Args:
            nodes: List of all nodes
            parent_summaries: Dictionary of parent summaries
            
        Returns:
            List of enriched leaf nodes
        """
        leaf_nodes = get_leaf_nodes(nodes)
        enriched_nodes = []
        
        print(f"Enriching {len(leaf_nodes)} leaf nodes with parent context...")
        
        for leaf in leaf_nodes:
            # Build hierarchy chain
            hierarchy_chain = []
            current = leaf
            
            # Traverse up the parent chain
            while NodeRelationship.PARENT in current.relationships:
                parent_id = current.relationships[NodeRelationship.PARENT].node_id
                parent_node = next((n for n in nodes if n.node_id == parent_id), None)
                
                if parent_node and parent_node.node_id in parent_summaries:
                    hierarchy_chain.insert(0, parent_summaries[parent_node.node_id])
                
                current = parent_node
                if current is None:
                    break
            
            # Create enriched content with context breadcrumbs
            if hierarchy_chain:
                context_str = " → ".join(hierarchy_chain)
                enriched_content = f"[CONTEXT: {context_str}]\n\n{leaf.get_content()}"
            else:
                enriched_content = leaf.get_content()
            
            # Create new enriched node
            enriched_node = TextNode(
                text=enriched_content,
                metadata={
                    **leaf.metadata,
                    "hierarchy_depth": len(hierarchy_chain),
                    "has_context": len(hierarchy_chain) > 0,
                    "original_node_id": leaf.node_id
                },
                relationships=leaf.relationships
            )
            enriched_node.node_id = leaf.node_id
            
            enriched_nodes.append(enriched_node)
        
        print(f"✓ Enriched {len(enriched_nodes)} nodes")
        return enriched_nodes

    def process_document(self, document: Document) -> tuple:
        """
        Complete processing pipeline for a document.
        
        Args:
            document: Document to process
            
        Returns:
            Tuple of (all_nodes, enriched_leaf_nodes)
        """
        # Create hierarchical nodes
        nodes = self.create_nodes([document])
        
        # Generate parent summaries
        parent_summaries = self.generate_parent_summaries(nodes)
        
        # Enrich leaf nodes
        enriched_leaf_nodes = self.enrich_leaf_nodes(nodes, parent_summaries)
        
        return nodes, enriched_leaf_nodes
    

if __name__ == "__main__":
    # Test chunker
    from src.embeddings import EmbeddingsManager
    
    manager = EmbeddingsManager()
    chunker = DocumentChunker(manager.get_llm())
    
    # Create a test document
    test_doc = Document(text="This is a test document. " * 1000)
    
    nodes, enriched = chunker.process_document(test_doc)
    print(f"\nProcessing complete:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Enriched leaf nodes: {len(enriched)}")
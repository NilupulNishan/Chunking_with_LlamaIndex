"""
Query engine module with support for persistent docstore and AutoMergingRetriever.
"""
import json
import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from config import settings
from src.embeddings import EmbeddingsManager


def get_docstore_path(collection_name):
    """Get path for docstore JSON file."""
    docstore_dir = settings.CHROMA_DB_PATH / "docstores"
    return docstore_dir / f"{collection_name}_docstore.json"


def load_docstore(collection_name):
    """Load docstore from disk."""
    docstore_path = get_docstore_path(collection_name)
    
    if not docstore_path.exists():
        return None
    
    docstore = SimpleDocumentStore()
    
    with open(docstore_path, 'r') as f:
        docs_dict = json.load(f)
    
    # Reconstruct nodes from dict
    for doc_id, doc_dict in docs_dict.items():
        node = TextNode.from_dict(doc_dict)
        docstore.add_documents([node])
    
    return docstore


class QueryEngine:
    """Handles querying and retrieval from ChromaDB collections."""
    
    def __init__(self, collection_name: str, verbose: bool = False, use_auto_merging: bool = True):
        """
        Initialize query engine for a specific collection.
        
        Args:
            collection_name: Name of the ChromaDB collection
            verbose: Whether to show verbose retrieval logs
            use_auto_merging: Whether to use AutoMergingRetriever (requires docstore)
        """
        self.collection_name = collection_name
        self.verbose = verbose
        self.use_auto_merging = use_auto_merging
        
        # Initialize embeddings
        self.embeddings_manager = EmbeddingsManager()
        
        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_PATH)
        )
        
        # Load collection
        try:
            self.chroma_collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            available = self.list_collections()
            raise ValueError(
                f"Collection '{collection_name}' not found.\n"
                f"Available collections: {', '.join(available)}"
            ) from e
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        
        # Load docstore if available and auto-merging is requested
        self.docstore = None
        if use_auto_merging:
            self.docstore = load_docstore(collection_name)
            if self.docstore is None:
                print(f"  Warning: Docstore not found for {collection_name}")
                print(f"  Auto-merging disabled. Using standard retrieval.")
                self.use_auto_merging = False
        
        # Create storage context
        if self.docstore:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.docstore
            )
        else:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        
        # Load index
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embeddings_manager.get_embed_model()
        )
        
        mode = "auto-merging" if self.use_auto_merging else "standard"
        print(f"✓ Query engine initialized ({mode}) for: {collection_name}")
    
    def query(self, query_text: str, similarity_top_k: int = None) -> str:
        """
        Query the collection and get a response.
        
        Args:
            query_text: The question or query
            similarity_top_k: Number of chunks to retrieve (default from settings)
            
        Returns:
            Response text
        """
        if similarity_top_k is None:
            similarity_top_k = settings.SIMILARITY_TOP_K
        
        if self.use_auto_merging and self.docstore:
            # Use AutoMergingRetriever
            base_retriever = self.index.as_retriever(
                similarity_top_k=similarity_top_k
            )
            
            retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=self.storage_context,
                verbose=self.verbose
            )
            
            query_engine = RetrieverQueryEngine.from_args(retriever)
        else:
            # Use standard query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                verbose=self.verbose
            )
        
        # Execute query
        response = query_engine.query(query_text)
        
        return str(response)
    
    def list_collections(self) -> list:
        """
        List all available collections in the database.
        
        Returns:
            List of collection names
        """
        collections = self.chroma_client.list_collections()
        return [col.name for col in collections]
    
    @staticmethod
    def get_available_collections() -> list:
        """
        Static method to get all available collections without initializing.
        
        Returns:
            List of collection names
        """
        chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_PATH)
        )
        collections = chroma_client.list_collections()
        return [col.name for col in collections]


class MultiCollectionQueryEngine:
    """Query engine that can search across multiple collections."""
    
    def __init__(self, collection_names: list = None, verbose: bool = False, use_auto_merging: bool = True):
        """
        Initialize multi-collection query engine.
        
        Args:
            collection_names: List of collections to search (None = all)
            verbose: Whether to show verbose logs
            use_auto_merging: Whether to use AutoMergingRetriever
        """
        if collection_names is None:
            collection_names = QueryEngine.get_available_collections()
        
        if not collection_names:
            raise ValueError("No collections available")
        
        self.engines = {
            name: QueryEngine(name, verbose=verbose, use_auto_merging=use_auto_merging)
            for name in collection_names
        }
        
        print(f"✓ Multi-collection engine initialized with {len(self.engines)} collections")
    
    def query(self, query_text: str, similarity_top_k: int = None) -> dict:
        """
        Query all collections and return results.
        
        Args:
            query_text: The question or query
            similarity_top_k: Number of chunks to retrieve per collection
            
        Returns:
            Dictionary mapping collection names to responses
        """
        results = {}
        
        for name, engine in self.engines.items():
            try:
                response = engine.query(query_text, similarity_top_k)
                results[name] = response
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        
        return results
    
    def query_best(self, query_text: str, similarity_top_k: int = None) -> tuple:
        """
        Query all collections and return the best result.
        
        Args:
            query_text: The question or query
            similarity_top_k: Number of chunks to retrieve per collection
            
        Returns:
            Tuple of (collection_name, response)
        """
        results = self.query(query_text, similarity_top_k)
        
        # Simple heuristic: longest response is usually best
        # You could implement more sophisticated ranking here
        best_collection = max(results.keys(), key=lambda k: len(results[k]))
        
        return best_collection, results[best_collection]


if __name__ == "__main__":
    # Test query engine
    try:
        collections = QueryEngine.get_available_collections()
        print(f"\nAvailable collections: {collections}")
        
        if collections:
            # Test with auto-merging
            engine = QueryEngine(collections[0], verbose=True, use_auto_merging=True)
            response = engine.query("What is this document about?")
            print(f"\nResponse:\n{response}")
        else:
            print("No collections available. Please process PDFs first.")
    except Exception as e:
        print(f"Error: {e}")
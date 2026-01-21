"""
Main script to process PDFs and create embeddings with persistent docstore.
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from config import settings
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingsManager
from src.chunker import DocumentChunker


def get_docstore_path(collection_name):
    """Get path for docstore JSON file."""
    docstore_dir = settings.CHROMA_DB_PATH / "docstores"
    docstore_dir.mkdir(exist_ok=True)
    return docstore_dir / f"{collection_name}_docstore.json"


def save_docstore(docstore, collection_name):
    """Save docstore to disk."""
    docstore_path = get_docstore_path(collection_name)
    
    # Convert docstore to JSON
    docs_dict = {}
    for doc_id, doc in docstore.docs.items():
        docs_dict[doc_id] = doc.to_dict()
    
    with open(docstore_path, 'w') as f:
        json.dump(docs_dict, f)
    
    print(f"  Saved docstore to: {docstore_path.name}")


def process_single_pdf(document, collection_name, pdf_path, embeddings_manager, chunker):
    """
    Process a single PDF document.
    
    Args:
        document: The loaded document
        collection_name: Name for the ChromaDB collection
        pdf_path: Path to the PDF file
        embeddings_manager: Embeddings manager instance
        chunker: Document chunker instance
    """
    print(f"\n{'='*80}")
    print(f"Processing: {pdf_path.name}")
    print(f"Collection: {collection_name}")
    print(f"{'='*80}")
    
    # Process document
    all_nodes, enriched_leaf_nodes = chunker.process_document(document)
    
    # Setup ChromaDB
    print(f"\nSetting up ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
    
    # Delete collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create document store with ALL nodes (for AutoMergingRetriever)
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    
    # Save docstore to disk
    save_docstore(docstore, collection_name)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )
    
    # Create index with progress bar
    print(f"\nCreating embeddings...")
    with tqdm(total=len(enriched_leaf_nodes), desc="Embedding nodes") as pbar:
        # Create index
        index = VectorStoreIndex(
            enriched_leaf_nodes,
            storage_context=storage_context,
            show_progress=False  # We're using our own progress bar
        )
        pbar.update(len(enriched_leaf_nodes))
    
    print(f"\n✓ Successfully processed {pdf_path.name}")
    print(f"  Collection: {collection_name}")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Indexed nodes: {len(enriched_leaf_nodes)}")


def main():
    """Main processing function."""
    print("\n" + "="*80)
    print("PDF EMBEDDINGS PROCESSING (With Persistent Docstore)")
    print("="*80)
    
    # Validate configuration
    try:
        settings.validate_config()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease set up your .env file with Azure OpenAI credentials.")
        return 1
    
    # Initialize components
    print("\nInitializing components...")
    pdf_processor = PDFProcessor()
    embeddings_manager = EmbeddingsManager()
    chunker = DocumentChunker(embeddings_manager.get_llm())
    
    # Load PDFs
    try:
        pdf_data = pdf_processor.load_all_pdfs(settings.PDF_DIRECTORY)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nPlease add PDF files to: {settings.PDF_DIRECTORY}")
        return 1
    
    if not pdf_data:
        print("\n❌ No PDFs were successfully loaded")
        return 1
    
    # Process each PDF
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(pdf_data)} PDF(S)")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    
    for i, (document, collection_name, pdf_path) in enumerate(pdf_data, 1):
        try:
            print(f"\n[{i}/{len(pdf_data)}] Processing {pdf_path.name}...")
            process_single_pdf(
                document,
                collection_name,
                pdf_path,
                embeddings_manager,
                chunker
            )
            successful += 1
        except Exception as e:
            print(f"\n❌ Error processing {pdf_path.name}: {e}")
            failed += 1
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful: {successful}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    
    # Show available collections
    print(f"\nAvailable collections:")
    chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
    collections = chroma_client.list_collections()
    for col in collections:
        docstore_path = get_docstore_path(col.name)
        docstore_exists = "✓" if docstore_path.exists() else "✗"
        print(f"  {docstore_exists} {col.name}")
    
    print(f"\nYou can now query these collections using scripts/query.py")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
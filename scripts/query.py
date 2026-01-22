"""
Interactive query script for searching PDF collections.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from colorama import init, Fore, Style
try:
    # Try to import v2 first (with docstore support)
    from src.query_engine import QueryEngine, MultiCollectionQueryEngine
except ImportError:
    # Fall back to v1 (simpler version)
    from src.query_engine import QueryEngine, MultiCollectionQueryEngine

# Initialize colorama for colored output
init()


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{Style.RESET_ALL}\n")


def print_response(text):
    """Print a formatted response."""
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}\n")


def select_collection(collections):
    """
    Let user select a collection to query.
    
    Args:
        collections: List of available collection names
        
    Returns:
        Selected collection name or None for all
    """
    print("Available collections:")
    print(f"  0. Search ALL collections")
    for i, name in enumerate(collections, 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input(f"\nSelect a collection (0-{len(collections)}): ").strip()
            
            if not choice:
                continue
            
            choice_num = int(choice)
            
            if choice_num == 0:
                return None  # Search all
            elif 1 <= choice_num <= len(collections):
                return collections[choice_num - 1]
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
            sys.exit(0)


def interactive_query():
    """Run interactive query session."""
    print_header("PDF QUERY SYSTEM")
    
    # Get available collections
    try:
        collections = QueryEngine.get_available_collections()
    except Exception as e:
        print(f"{Fore.RED}Error accessing database: {e}{Style.RESET_ALL}")
        print(f"\nPlease run 'python scripts/process_pdfs.py' first.")
        return 1
    
    if not collections:
        print(f"{Fore.RED}No collections found in the database.{Style.RESET_ALL}")
        print(f"\nPlease run 'python scripts/process_pdfs.py' first.")
        return 1
    
    # Select collection
    selected = select_collection(collections)
    
    # Initialize query engine
    try:
        if selected:
            engine = QueryEngine(selected, verbose=False)
            print(f"\n{Fore.GREEN}✓ Connected to collection: {selected}{Style.RESET_ALL}")
        else:
            engine = MultiCollectionQueryEngine(verbose=False)
            print(f"\n{Fore.GREEN}✓ Connected to all collections{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error initializing query engine: {e}{Style.RESET_ALL}")
        return 1
    
    # Query loop
    print(f"\n{Fore.YELLOW}Enter your questions (or 'quit' to exit){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Commands: 'quit', 'exit', 'change' (to change collection){Style.RESET_ALL}\n")
    
    while True:
        try:
            # Get query
            query = input(f"{Fore.CYAN}Query: {Style.RESET_ALL}").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
            
            if query.lower() == 'change':
                # Change collection
                selected = select_collection(collections)
                if selected:
                    engine = QueryEngine(selected, verbose=False)
                    print(f"\n{Fore.GREEN}✓ Switched to collection: {selected}{Style.RESET_ALL}")
                else:
                    engine = MultiCollectionQueryEngine(verbose=False)
                    print(f"\n{Fore.GREEN}✓ Switched to all collections{Style.RESET_ALL}")
                continue
            
            # Execute query
            print(f"\n{Fore.YELLOW}Searching...{Style.RESET_ALL}")
            
            if isinstance(engine, MultiCollectionQueryEngine):
                # Multi-collection search
                collection_name, response = engine.query_best(query)
                print(f"\n{Fore.YELLOW}Best match from: {collection_name}{Style.RESET_ALL}")
                print_response(response)
            else:
                # Single collection search
                response = engine.query(query)
                print_response(response)
        
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}\n")
            continue
    
    return 0


def single_query(collection_name, query_text):
    """
    Execute a single query (for programmatic use).
    
    Args:
        collection_name: Name of collection to query (None for all)
        query_text: Query string
    """
    try:
        if collection_name:
            engine = QueryEngine(collection_name)
            response = engine.query(query_text)
            print(response)
        else:
            engine = MultiCollectionQueryEngine()
            collection_name, response = engine.query_best(query_text)
            print(f"Best match from: {collection_name}")
            print(response)
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        # Interactive mode
        return interactive_query()
    elif len(sys.argv) == 2:
        # Single query, all collections
        return single_query(None, sys.argv[1])
    elif len(sys.argv) == 3:
        # Single query, specific collection
        return single_query(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Interactive mode:     python scripts/query.py")
        print("  Single query (all):   python scripts/query.py 'your question'")
        print("  Single query (one):   python scripts/query.py collection_name 'your question'")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
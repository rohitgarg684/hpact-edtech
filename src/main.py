"""
Main entry point for the LangChain web content processing pipeline.
Processes URLs, generates LangChain Documents, and runs them through the complete pipeline.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from web_crawler import create_web_crawler, WebCrawler
from pipeline import create_pipeline, LangChainPipeline


def setup_environment():
    """Load environment variables from .env file if present."""
    load_dotenv()
    
    # Set processing timestamp
    os.environ['PROCESSING_TIMESTAMP'] = datetime.now().isoformat()
    
    # Validate required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)


def process_url(url: str, crawler: WebCrawler, pipeline: LangChainPipeline) -> Dict[str, Any]:
    """
    Process a single URL through the complete pipeline.
    
    Args:
        url: URL to process
        crawler: WebCrawler instance
        pipeline: LangChainPipeline instance
        
    Returns:
        Dictionary containing processing results
    """
    try:
        print(f"Processing URL: {url}")
        
        # Step 1: Crawl the URL and create LangChain Document
        print("  → Crawling web content...")
        document = crawler.crawl_url(url)
        print(f"  → Extracted {len(document.page_content)} characters")
        print(f"  → Title: {document.metadata.get('title', 'No title')}")
        
        # Step 2: Process document through LangChain pipeline
        print("  → Processing through LangChain pipeline...")
        result = pipeline.process_document(document)
        
        if result['status'] == 'success':
            print(f"  ✓ Successfully processed document (ID: {result['doc_id']})")
            print(f"  → Generated {len(result['tags'].get('primary_tags', []))} primary tags")
            print(f"  → Embedding dimension: {result['embedding_dimension']}")
        else:
            print(f"  ✗ Processing failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        error_result = {
            "url": url,
            "status": "error",
            "error": str(e),
            "doc_id": None
        }
        print(f"  ✗ Failed to process URL: {str(e)}")
        return error_result


def process_urls_batch(urls: list, crawler: WebCrawler, pipeline: LangChainPipeline) -> Dict[str, Any]:
    """
    Process multiple URLs in batch.
    
    Args:
        urls: List of URLs to process
        crawler: WebCrawler instance
        pipeline: LangChainPipeline instance
        
    Returns:
        Dictionary containing batch processing results
    """
    results = []
    successful = 0
    failed = 0
    
    print(f"\nProcessing {len(urls)} URLs...")
    print("=" * 50)
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]", end=" ")
        result = process_url(url, crawler, pipeline)
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    batch_result = {
        "total_urls": len(urls),
        "successful": successful,
        "failed": failed,
        "results": results,
        "processing_timestamp": os.getenv('PROCESSING_TIMESTAMP')
    }
    
    print("\n" + "=" * 50)
    print(f"Batch processing completed: {successful} successful, {failed} failed")
    
    return batch_result


def search_knowledge_graph(query: str, pipeline: LangChainPipeline, limit: int = 5) -> Dict[str, Any]:
    """
    Search the knowledge graph for similar documents.
    
    Args:
        query: Search query
        pipeline: LangChainPipeline instance
        limit: Maximum number of results
        
    Returns:
        Search results
    """
    try:
        print(f"Searching knowledge graph for: '{query}'")
        results = pipeline.search_similar_documents(query, limit)
        
        print(f"Found {len(results)} similar documents:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (Score: {result['score']:.3f})")
            print(f"     URL: {result['url']}")
            print(f"     Preview: {result['content_preview']}...")
            print()
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return {
            "query": query,
            "results": [],
            "error": str(e)
        }


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='LangChain Web Content Processing Pipeline')
    parser.add_argument('--url', '-u', type=str, help='Single URL to process')
    parser.add_argument('--urls-file', '-f', type=str, help='File containing URLs (one per line)')
    parser.add_argument('--search', '-s', type=str, help='Search query for knowledge graph')
    parser.add_argument('--limit', '-l', type=int, default=5, help='Limit for search results')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Initialize components
    try:
        crawler = create_web_crawler()
        pipeline = create_pipeline()
        print("✓ Initialized web crawler and LangChain pipeline")
    except Exception as e:
        print(f"✗ Failed to initialize components: {str(e)}")
        sys.exit(1)
    
    results = None
    
    # Handle search functionality
    if args.search:
        results = search_knowledge_graph(args.search, pipeline, args.limit)
    
    # Handle URL processing
    elif args.url:
        # Process single URL
        results = process_url(args.url, crawler, pipeline)
    
    elif args.urls_file:
        # Process URLs from file
        try:
            with open(args.urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            if not urls:
                print(f"No URLs found in file: {args.urls_file}")
                sys.exit(1)
            
            results = process_urls_batch(urls, crawler, pipeline)
            
        except FileNotFoundError:
            print(f"File not found: {args.urls_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading URLs file: {str(e)}")
            sys.exit(1)
    
    else:
        # Interactive mode
        print("\nLangChain Web Content Processing Pipeline")
        print("=" * 45)
        print("Enter a URL to process, 'search <query>' to search, or 'quit' to exit:")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        search_knowledge_graph(query, pipeline, args.limit)
                    else:
                        print("Please provide a search query")
                
                elif user_input.startswith('http'):
                    process_url(user_input, crawler, pipeline)
                
                else:
                    print("Please enter a valid URL (starting with http) or 'search <query>'")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    # Save results to file if requested
    if results and args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    main()
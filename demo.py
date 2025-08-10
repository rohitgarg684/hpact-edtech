#!/usr/bin/env python3
"""
Example script demonstrating the LangChain Web Content Processing Pipeline.
This example shows how to use the pipeline programmatically.
"""

import os
from unittest.mock import patch, Mock
import json

# Set up mock environment for demo
os.environ['OPENAI_API_KEY'] = 'demo-api-key'

from src.web_crawler import create_web_crawler
from src.pipeline import create_pipeline


def demo_web_crawler():
    """Demonstrate web crawler functionality."""
    print("🌐 Web Crawler Demo")
    print("-" * 30)
    
    # Create web crawler
    crawler = create_web_crawler()
    print(f"Created web crawler with timeout: {crawler.timeout}s")
    
    # Sample HTML content (simulating a fetched webpage)
    sample_html = """
    <html>
    <head>
        <title>The Future of Artificial Intelligence</title>
        <meta name="description" content="Exploring the latest trends and developments in AI technology">
        <meta name="keywords" content="artificial intelligence, AI, machine learning, deep learning">
        <meta property="og:title" content="AI Trends 2024">
    </head>
    <body>
        <header>Site Header</header>
        <main>
            <article>
                <h1>The Future of Artificial Intelligence</h1>
                <p>Artificial Intelligence continues to evolve at an unprecedented pace, transforming industries and reshaping how we work and live.</p>
                <p>From natural language processing to computer vision, AI technologies are becoming more sophisticated and accessible.</p>
                <h2>Key Developments</h2>
                <ul>
                    <li>Large Language Models (LLMs)</li>
                    <li>Computer Vision Advances</li>
                    <li>Autonomous Systems</li>
                    <li>AI Ethics and Governance</li>
                </ul>
                <p>These developments represent just the beginning of what's possible with AI technology.</p>
            </article>
        </main>
        <footer>Site Footer</footer>
    </body>
    </html>
    """
    
    # Extract content using BeautifulSoup
    content = crawler.extract_with_beautifulsoup(sample_html)
    print(f"✓ Extracted content: {len(content)} characters")
    print(f"Content preview: {content[:150]}...")
    
    # Extract metadata
    metadata = crawler.extract_metadata(sample_html, "https://example.com/ai-future")
    print(f"✓ Extracted metadata: {metadata}")
    
    # Create LangChain Document
    from langchain_core.documents import Document
    document = Document(
        page_content=content,
        metadata=metadata
    )
    print(f"✓ Created LangChain Document with {len(document.page_content)} chars")
    
    return document


def demo_pipeline_processing(document):
    """Demonstrate pipeline processing with mocked APIs."""
    print("\n🔄 Pipeline Processing Demo")
    print("-" * 35)
    
    # Mock external API calls
    with patch('src.pipeline.ChatOpenAI') as mock_chat, \
         patch('src.pipeline.OpenAIEmbeddings') as mock_embeddings, \
         patch('src.pipeline.QdrantClient') as mock_qdrant:
        
        # Configure mocks
        mock_chat_instance = Mock()
        mock_chat.return_value = mock_chat_instance
        
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.return_value = [0.1, 0.2, 0.3] * 1024  # 3072 dims
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client_mock = Mock()
        mock_qdrant_instance.upsert.return_value = None
        mock_qdrant.return_value = mock_qdrant_instance
        
        # Create pipeline
        pipeline = create_pipeline()
        print("✓ Pipeline initialized with LangChain components")
        
        # Mock tagging response
        sample_tags = {
            "primary_tags": ["artificial intelligence", "AI technology", "future trends"],
            "categories": ["technology", "innovation"],
            "topics": ["machine learning", "computer vision", "automation"],
            "content_type": "article",
            "sentiment": "positive",
            "complexity": "intermediate",
            "key_concepts": ["large language models", "autonomous systems", "AI ethics"],
            "summary": "Article discussing the future developments and trends in artificial intelligence technology"
        }
        
        # Mock the tagging chain
        with patch.object(pipeline, 'tagging_chain') as mock_chain:
            mock_chain.invoke.return_value = json.dumps(sample_tags)
            
            # Process the document
            result = pipeline.process_document(document)
            
            print(f"✓ Document processing result: {result['status']}")
            print(f"✓ Generated tags: {result['tags']['primary_tags']}")
            print(f"✓ Content categorized as: {result['tags']['content_type']}")
            print(f"✓ Complexity level: {result['tags']['complexity']}")
            print(f"✓ Embedding dimension: {result['embedding_dimension']}")
            print(f"✓ Document ID: {result['doc_id']}")
            
            return result


def demo_search_functionality():
    """Demonstrate search functionality with mocked results."""
    print("\n🔍 Search Functionality Demo")
    print("-" * 35)
    
    with patch('src.pipeline.ChatOpenAI') as mock_chat, \
         patch('src.pipeline.OpenAIEmbeddings') as mock_embeddings, \
         patch('src.pipeline.QdrantClient') as mock_qdrant:
        
        # Configure mocks
        mock_chat.return_value = Mock()
        
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.return_value = [0.1] * 3072
        mock_embeddings.return_value = mock_embedding_instance
        
        # Mock search results
        mock_search_result1 = Mock()
        mock_search_result1.id = 'doc-ai-123'
        mock_search_result1.score = 0.92
        mock_search_result1.payload = {
            'url': 'https://example.com/ai-future',
            'title': 'The Future of Artificial Intelligence',
            'tags': {'primary_tags': ['AI', 'technology', 'innovation']},
            'content': 'Artificial Intelligence continues to evolve at an unprecedented pace...'
        }
        
        mock_search_result2 = Mock()
        mock_search_result2.id = 'doc-ml-456'
        mock_search_result2.score = 0.87
        mock_search_result2.payload = {
            'url': 'https://example.com/machine-learning',
            'title': 'Introduction to Machine Learning',
            'tags': {'primary_tags': ['machine learning', 'algorithms', 'data science']},
            'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms...'
        }
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_instance.search.return_value = [mock_search_result1, mock_search_result2]
        mock_qdrant.return_value = mock_qdrant_instance
        
        # Create pipeline and search
        pipeline = create_pipeline()
        
        query = "artificial intelligence and machine learning"
        results = pipeline.search_similar_documents(query, limit=5)
        
        print(f"✓ Search query: '{query}'")
        print(f"✓ Found {len(results)} similar documents:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (Score: {result['score']:.3f})")
            print(f"     URL: {result['url']}")
            print(f"     Preview: {result['content_preview'][:80]}...")
            print()


def main():
    """Run the complete demo."""
    print("🚀 LangChain Web Content Processing Pipeline Demo")
    print("=" * 55)
    
    try:
        # Demo 1: Web Crawler
        document = demo_web_crawler()
        
        # Demo 2: Pipeline Processing
        processing_result = demo_pipeline_processing(document)
        
        # Demo 3: Search Functionality
        demo_search_functionality()
        
        print("✅ Demo completed successfully!")
        print("\nTo use this pipeline in production:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Start Qdrant server (docker run -p 6333:6333 qdrant/qdrant)")
        print("3. Run: python src/main.py --url https://example.com/article")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
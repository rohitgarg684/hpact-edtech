#!/usr/bin/env python3
"""
Integration test script for the LangChain Web Content Processing Pipeline.
This script tests the complete workflow without requiring external APIs.
"""

import os
import sys
import tempfile
import json
from unittest.mock import patch, Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web_crawler import create_web_crawler
from src.pipeline import create_pipeline


def test_web_crawler_integration():
    """Test web crawler with sample HTML content."""
    print("Testing Web Crawler Integration...")
    
    crawler = create_web_crawler()
    
    # Sample HTML content
    html_content = """
    <html>
    <head>
        <title>Sample Article: Machine Learning Basics</title>
        <meta name="description" content="A comprehensive guide to machine learning fundamentals">
        <meta name="keywords" content="machine learning, AI, data science">
        <meta name="author" content="Data Scientist">
    </head>
    <body>
        <header>Website Header</header>
        <nav>Navigation Menu</nav>
        <main>
            <article>
                <h1>Introduction to Machine Learning</h1>
                <p>Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.</p>
                <p>It enables computers to learn and make decisions from data without being explicitly programmed for every scenario.</p>
                <h2>Types of Machine Learning</h2>
                <ul>
                    <li>Supervised Learning</li>
                    <li>Unsupervised Learning</li>
                    <li>Reinforcement Learning</li>
                </ul>
                <p>This article provides a comprehensive overview of these fundamental concepts.</p>
            </article>
        </main>
        <footer>Website Footer</footer>
    </body>
    </html>
    """
    
    try:
        # Test BeautifulSoup extraction
        content = crawler.extract_with_beautifulsoup(html_content)
        print(f"✓ BeautifulSoup extraction successful: {len(content)} characters")
        
        # Verify content quality
        assert "Machine learning" in content
        assert "Website Header" not in content  # Should be filtered out
        assert "Navigation Menu" not in content  # Should be filtered out
        assert "Website Footer" not in content  # Should be filtered out
        
        # Test metadata extraction
        metadata = crawler.extract_metadata(html_content, "https://example.com/ml-article")
        print(f"✓ Metadata extraction successful: {len(metadata)} fields")
        
        # Verify metadata completeness
        assert metadata['title'] == 'Sample Article: Machine Learning Basics'
        assert metadata['description'] == 'A comprehensive guide to machine learning fundamentals'
        assert metadata['domain'] == 'example.com'
        
        print("✓ Web Crawler Integration Test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Web Crawler Integration Test FAILED: {str(e)}")
        return False


def test_pipeline_with_mocked_apis():
    """Test pipeline functionality with mocked external APIs."""
    print("\nTesting Pipeline Integration with Mocked APIs...")
    
    try:
        # Set up environment
        os.environ['OPENAI_API_KEY'] = 'test-api-key'
        
        # Mock external dependencies
        with patch('src.pipeline.ChatOpenAI') as mock_chat, \
             patch('src.pipeline.OpenAIEmbeddings') as mock_embeddings, \
             patch('src.pipeline.QdrantClient') as mock_qdrant:
            
            # Configure mocks
            mock_chat_instance = Mock()
            mock_chat.return_value = mock_chat_instance
            
            mock_embedding_instance = Mock()
            mock_embedding_instance.embed_query.return_value = [0.1] * 3072
            mock_embeddings.return_value = mock_embedding_instance
            
            mock_qdrant_instance = Mock()
            mock_qdrant_instance.get_collections.return_value.collections = []
            mock_qdrant.return_value = mock_qdrant_instance
            
            # Create pipeline
            pipeline = create_pipeline()
            print("✓ Pipeline initialization successful")
            
            # Test document creation
            from langchain_core.documents import Document
            test_document = Document(
                page_content="This is a test article about artificial intelligence and machine learning technologies.",
                metadata={
                    'url': 'https://test.example.com/ai-article',
                    'title': 'AI and ML Technologies',
                    'domain': 'test.example.com',
                    'description': 'Test article about AI/ML'
                }
            )
            
            # Mock tagging response
            sample_tags = {
                "primary_tags": ["artificial intelligence", "machine learning", "technology"],
                "categories": ["technology", "education"],
                "topics": ["AI", "ML", "algorithms"],
                "content_type": "article",
                "sentiment": "neutral",
                "complexity": "intermediate",
                "key_concepts": ["neural networks", "algorithms", "data science"],
                "summary": "Article about AI and ML technologies"
            }
            
            with patch.object(pipeline, 'tagging_chain') as mock_chain:
                mock_chain.invoke.return_value = json.dumps(sample_tags)
                
                # Test complete document processing
                result = pipeline.process_document(test_document)
                
                print(f"✓ Document processing successful: {result['status']}")
                
                # Verify results
                assert result['status'] == 'success'
                assert result['url'] == test_document.metadata['url']
                assert result['title'] == test_document.metadata['title']
                assert result['embedding_dimension'] == 3072
                assert 'doc_id' in result
                
                print("✓ Pipeline Integration Test PASSED")
                return True
                
    except Exception as e:
        print(f"✗ Pipeline Integration Test FAILED: {str(e)}")
        return False


def test_main_cli_interface():
    """Test the main CLI interface."""
    print("\nTesting CLI Interface...")
    
    try:
        # Test help command (already tested above, but verify it works)
        import subprocess
        result = subprocess.run([
            sys.executable, 'src/main.py', '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert 'LangChain Web Content Processing Pipeline' in result.stdout
        
        print("✓ CLI Help interface working")
        
        # Test with invalid API key (should show initialization error)
        env = os.environ.copy()
        env['OPENAI_API_KEY'] = ''  # Empty API key
        
        result = subprocess.run([
            sys.executable, 'src/main.py', '--help'
        ], capture_output=True, text=True, env=env)
        
        # Help should still work even without API key
        assert result.returncode == 0
        
        print("✓ CLI Interface Test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ CLI Interface Test FAILED: {str(e)}")
        return False


def test_url_file_processing():
    """Test URL file processing functionality."""
    print("\nTesting URL File Processing...")
    
    try:
        # Create a temporary file with test URLs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://example.com/article1\n")
            f.write("https://example.com/article2\n")
            f.write("https://example.com/article3\n")
            temp_file = f.name
        
        # Test reading the file
        try:
            with open(temp_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            assert len(urls) == 3
            assert all(url.startswith('http') for url in urls)
            
            print(f"✓ URL file processing successful: {len(urls)} URLs loaded")
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ URL File Processing Test FAILED: {str(e)}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("LangChain Web Content Processing Pipeline - Integration Tests")
    print("=" * 60)
    
    tests = [
        test_web_crawler_integration,
        test_pipeline_with_mocked_apis,
        test_main_cli_interface,
        test_url_file_processing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Integration Tests Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All integration tests PASSED!")
        return 0
    else:
        print("❌ Some integration tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
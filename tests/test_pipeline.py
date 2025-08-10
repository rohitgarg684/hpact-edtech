"""
Unit tests for the LangChain pipeline module.
Tests pipeline functionality with mocked LangChain components.
"""

import os
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.pipeline import LangChainPipeline, create_pipeline


class TestLangChainPipeline:
    """Test class for LangChainPipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock environment variables
        self.env_vars = {
            'OPENAI_API_KEY': 'test-api-key',
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'OPENAI_TEMPERATURE': '0.1',
            'OPENAI_EMBEDDING_MODEL': 'text-embedding-3-large',
            'QDRANT_HOST': 'localhost',
            'QDRANT_PORT': '6333',
            'QDRANT_COLLECTION_NAME': 'test_collection',
            'EMBEDDING_SIZE': '3072'
        }
        
        # Sample document for testing
        self.sample_document = Document(
            page_content="This is a test article about machine learning and artificial intelligence. It covers basic concepts and applications in modern technology.",
            metadata={
                'url': 'https://example.com/ml-article',
                'title': 'Introduction to Machine Learning',
                'domain': 'example.com',
                'description': 'A beginner guide to ML concepts'
            }
        )
        
        # Sample tags response
        self.sample_tags = {
            "primary_tags": ["machine learning", "artificial intelligence", "technology"],
            "categories": ["education", "technology"],
            "topics": ["ML", "AI", "algorithms"],
            "content_type": "article",
            "sentiment": "positive",
            "complexity": "beginner",
            "key_concepts": ["neural networks", "supervised learning", "data science"],
            "summary": "An introductory article about machine learning concepts and applications"
        }
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_pipeline_initialization(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test successful pipeline initialization."""
        # Mock Qdrant client
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        assert pipeline.openai_api_key == 'test-key'
        mock_chat.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_qdrant_client.assert_called_once()
    
    def test_pipeline_initialization_missing_api_key(self):
        """Test pipeline initialization with missing API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                LangChainPipeline()
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_qdrant_collection_creation(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test Qdrant collection creation."""
        os.environ.update(self.env_vars)
        
        # Mock Qdrant client - collection doesn't exist
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        # Should create collection
        mock_qdrant_instance.create_collection.assert_called_once()
        
        # Verify collection creation parameters
        call_args = mock_qdrant_instance.create_collection.call_args
        assert call_args[1]['collection_name'] == 'test_collection'
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings') 
    @patch('src.pipeline.QdrantClient')
    def test_qdrant_collection_exists(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test when Qdrant collection already exists."""
        os.environ.update(self.env_vars)
        
        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = 'test_collection'
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = [mock_collection]
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        # Should not create collection
        mock_qdrant_instance.create_collection.assert_not_called()
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_generate_tags_success(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test successful tag generation."""
        os.environ.update(self.env_vars)
        
        # Mock successful tagging chain response
        mock_chain = Mock()
        mock_chain.invoke.return_value = json.dumps(self.sample_tags)
        
        # Mock pipeline setup
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        with patch.object(LangChainPipeline, 'tagging_chain', mock_chain):
            pipeline = LangChainPipeline()
            result = pipeline.generate_tags(self.sample_document)
        
        assert result == self.sample_tags
        mock_chain.invoke.assert_called_once()
        
        # Verify input structure
        call_args = mock_chain.invoke.call_args[0][0]
        assert 'title' in call_args
        assert 'url' in call_args
        assert 'content' in call_args
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_generate_tags_invalid_json(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test tag generation with invalid JSON response."""
        os.environ.update(self.env_vars)
        
        # Mock chain with invalid JSON response
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Not valid JSON response with some tags"
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        with patch.object(LangChainPipeline, 'tagging_chain', mock_chain):
            pipeline = LangChainPipeline()
            result = pipeline.generate_tags(self.sample_document)
        
        # Should fallback to basic structure
        assert 'primary_tags' in result
        assert 'categories' in result
        assert isinstance(result['primary_tags'], list)
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_generate_tags_exception(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test tag generation with exception."""
        os.environ.update(self.env_vars)
        
        # Mock chain that raises exception
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("OpenAI API error")
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        with patch.object(LangChainPipeline, 'tagging_chain', mock_chain):
            pipeline = LangChainPipeline()
            result = pipeline.generate_tags(self.sample_document)
        
        # Should return error structure
        assert result['primary_tags'] == ['web-content']
        assert 'tagging_error' in result
        assert 'OpenAI API error' in result['tagging_error']
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_generate_embeddings_success(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test successful embedding generation."""
        os.environ.update(self.env_vars)
        
        # Mock embeddings
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.return_value = [0.1, 0.2, 0.3] * 1024  # 3072 dimensions
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        result = pipeline.generate_embeddings(self.sample_document)
        
        assert len(result) == 3072
        assert all(isinstance(x, float) for x in result)
        mock_embedding_instance.embed_query.assert_called_once()
        
        # Verify embedding input includes title and content
        call_args = mock_embedding_instance.embed_query.call_args[0][0]
        assert self.sample_document.metadata['title'] in call_args
        assert self.sample_document.page_content in call_args
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_generate_embeddings_failure(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test embedding generation failure."""
        os.environ.update(self.env_vars)
        
        # Mock embeddings failure
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.side_effect = Exception("Embedding API error")
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        with pytest.raises(Exception, match="Failed to generate embeddings"):
            pipeline.generate_embeddings(self.sample_document)
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    @patch('uuid.uuid4')
    def test_store_in_knowledge_graph_success(self, mock_uuid, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test successful knowledge graph storage."""
        os.environ.update(self.env_vars)
        
        # Mock UUID
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='test-doc-id-123')
        
        # Mock Qdrant client
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        embedding = [0.1, 0.2, 0.3] * 1024
        doc_id = pipeline.store_in_knowledge_graph(
            self.sample_document, 
            self.sample_tags, 
            embedding
        )
        
        assert doc_id == 'test-doc-id-123'
        mock_qdrant_instance.upsert.assert_called_once()
        
        # Verify upsert call structure
        call_args = mock_qdrant_instance.upsert.call_args
        assert call_args[1]['collection_name'] == 'test_collection'
        assert len(call_args[1]['points']) == 1
        
        point = call_args[1]['points'][0]
        assert point.id == 'test-doc-id-123'
        assert point.vector == embedding
        assert 'url' in point.payload
        assert 'tags' in point.payload
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_store_in_knowledge_graph_failure(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test knowledge graph storage failure."""
        os.environ.update(self.env_vars)
        
        # Mock Qdrant client failure
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_instance.upsert.side_effect = Exception("Qdrant connection error")
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        embedding = [0.1, 0.2, 0.3] * 1024
        
        with pytest.raises(Exception, match="Failed to store in knowledge graph"):
            pipeline.store_in_knowledge_graph(
                self.sample_document,
                self.sample_tags,
                embedding
            )
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_process_document_success(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test successful complete document processing."""
        os.environ.update(self.env_vars)
        
        # Mock all components
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        # Mock individual methods
        with patch.object(pipeline, 'generate_tags', return_value=self.sample_tags), \
             patch.object(pipeline, 'generate_embeddings', return_value=[0.1] * 3072), \
             patch.object(pipeline, 'store_in_knowledge_graph', return_value='test-doc-id'):
            
            result = pipeline.process_document(self.sample_document)
        
        assert result['status'] == 'success'
        assert result['doc_id'] == 'test-doc-id'
        assert result['url'] == self.sample_document.metadata['url']
        assert result['title'] == self.sample_document.metadata['title']
        assert result['tags'] == self.sample_tags
        assert result['embedding_dimension'] == 3072
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_process_document_failure(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test document processing failure."""
        os.environ.update(self.env_vars)
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = LangChainPipeline()
        
        # Mock method failure
        with patch.object(pipeline, 'generate_tags', side_effect=Exception("Tag generation failed")):
            result = pipeline.process_document(self.sample_document)
        
        assert result['status'] == 'error'
        assert result['doc_id'] is None
        assert 'Tag generation failed' in result['error']
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_search_similar_documents_success(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test successful similar document search."""
        os.environ.update(self.env_vars)
        
        # Mock search results
        mock_search_result = Mock()
        mock_search_result.id = 'doc-123'
        mock_search_result.score = 0.95
        mock_search_result.payload = {
            'url': 'https://example.com/similar',
            'title': 'Similar Document',
            'tags': {'primary_tags': ['AI', 'ML']},
            'content': 'This is similar content about AI and machine learning.'
        }
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_instance.search.return_value = [mock_search_result]
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.return_value = [0.1] * 3072
        mock_embeddings.return_value = mock_embedding_instance
        
        pipeline = LangChainPipeline()
        
        results = pipeline.search_similar_documents("machine learning", limit=5)
        
        assert len(results) == 1
        assert results[0]['doc_id'] == 'doc-123'
        assert results[0]['score'] == 0.95
        assert results[0]['title'] == 'Similar Document'
        
        # Verify search was called correctly
        mock_qdrant_instance.search.assert_called_once()
        search_args = mock_qdrant_instance.search.call_args[1]
        assert search_args['collection_name'] == 'test_collection'
        assert search_args['limit'] == 5
    
    @patch.dict('os.environ')
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_search_similar_documents_failure(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test similar document search failure."""
        os.environ.update(self.env_vars)
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_instance.search.side_effect = Exception("Search failed")
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.return_value = [0.1] * 3072
        mock_embeddings.return_value = mock_embedding_instance
        
        pipeline = LangChainPipeline()
        
        with pytest.raises(Exception, match="Failed to search similar documents"):
            pipeline.search_similar_documents("test query")
    
    def test_extract_simple_tags(self):
        """Test simple tag extraction from unstructured text."""
        with patch.dict('os.environ', self.env_vars):
            # Mock the necessary components to create pipeline
            with patch('src.pipeline.ChatOpenAI'), \
                 patch('src.pipeline.OpenAIEmbeddings'), \
                 patch('src.pipeline.QdrantClient') as mock_qdrant:
                
                mock_qdrant_instance = Mock()
                mock_qdrant_instance.get_collections.return_value.collections = []
                mock_qdrant.return_value = mock_qdrant_instance
                
                pipeline = LangChainPipeline()
                
                text = "This article discusses machine learning algorithms and data science methodologies for artificial intelligence"
                tags = pipeline._extract_simple_tags(text)
                
                assert isinstance(tags, list)
                assert len(tags) <= 5
                assert all(len(tag) > 3 for tag in tags)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_create_pipeline_factory(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test the factory function for creating pipeline."""
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        pipeline = create_pipeline()
        
        assert isinstance(pipeline, LangChainPipeline)
        mock_chat.assert_called_once()
        mock_embeddings.assert_called_once()


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'QDRANT_HOST': 'localhost',
        'QDRANT_PORT': '6333'
    })
    @patch('src.pipeline.ChatOpenAI')
    @patch('src.pipeline.OpenAIEmbeddings')
    @patch('src.pipeline.QdrantClient')
    def test_full_pipeline_workflow(self, mock_qdrant_client, mock_embeddings, mock_chat):
        """Test complete pipeline workflow from document to storage."""
        # Mock all external dependencies
        mock_chat_instance = Mock()
        mock_chat.return_value = mock_chat_instance
        
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_query.return_value = [0.1] * 3072
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_qdrant_instance
        
        # Create mock tagging chain
        mock_tagging_chain = Mock()
        sample_tags_json = json.dumps({
            "primary_tags": ["integration", "test"],
            "categories": ["testing"],
            "content_type": "article",
            "sentiment": "neutral",
            "summary": "Integration test document"
        })
        mock_tagging_chain.invoke.return_value = sample_tags_json
        
        pipeline = LangChainPipeline()
        pipeline.tagging_chain = mock_tagging_chain
        
        # Test document
        document = Document(
            page_content="This is an integration test document for the pipeline.",
            metadata={
                'url': 'https://test.com/integration',
                'title': 'Integration Test',
                'domain': 'test.com'
            }
        )
        
        # Process document
        result = pipeline.process_document(document)
        
        # Verify successful processing
        assert result['status'] == 'success'
        assert result['url'] == 'https://test.com/integration'
        assert result['title'] == 'Integration Test'
        assert 'doc_id' in result
        assert result['embedding_dimension'] == 3072
        
        # Verify all steps were called
        mock_tagging_chain.invoke.assert_called_once()
        mock_embedding_instance.embed_query.assert_called_once()
        mock_qdrant_instance.upsert.assert_called_once()
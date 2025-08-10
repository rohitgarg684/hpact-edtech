import os
import boto3
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_community.graphs import NeptuneGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
import logging
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class LangChainNeptuneService:
    """
    LangChain-based AWS Neptune service for knowledge graph storage and retrieval.
    Manages entities, relationships, and semantic connections from documents.
    """
    
    def __init__(self):
        """Initialize Neptune service through LangChain with environment configuration."""
        # Read Neptune configuration from environment variables
        self.neptune_endpoint = os.getenv("NEPTUNE_ENDPOINT")
        self.neptune_port = int(os.getenv("NEPTUNE_PORT", "8182"))
        self.neptune_region = os.getenv("AWS_REGION", "us-east-1")
        self.neptune_iam_role = os.getenv("NEPTUNE_IAM_ROLE")
        
        # AWS credentials
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")  # Optional for STS
        
        if not self.neptune_endpoint:
            logger.warning("NEPTUNE_ENDPOINT not set, Neptune service will be disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize OpenAI for graph extraction
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Initialize Neptune components
        self._init_neptune_connection()
        self._init_graph_transformer()
        
        logger.info(f"Initialized LangChain Neptune service at {self.neptune_endpoint}")

    def _init_neptune_connection(self):
        """Initialize connection to Neptune database."""
        try:
            # Initialize AWS session
            if self.aws_access_key and self.aws_secret_key:
                session = boto3.Session(
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    aws_session_token=self.aws_session_token,
                    region_name=self.neptune_region
                )
            else:
                # Use default AWS credentials (IAM role, etc.)
                session = boto3.Session(region_name=self.neptune_region)
            
            # Initialize Neptune graph through LangChain
            self.graph = NeptuneGraph(
                host=self.neptune_endpoint,
                port=self.neptune_port,
                use_https=True,
                region=self.neptune_region,
                sign=True,  # Enable IAM signing
                credentials=session.get_credentials() if hasattr(session.get_credentials(), 'access_key') else None
            )
            
            logger.info("Successfully connected to Neptune database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neptune: {str(e)}")
            # Disable Neptune functionality if connection fails
            self.enabled = False

    def _init_graph_transformer(self):
        """Initialize LLM-based graph transformer for extracting entities and relationships."""
        try:
            self.graph_transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=["Person", "Organization", "Location", "Concept", "Technology", "Topic", "Document"],
                allowed_relationships=["RELATES_TO", "MENTIONS", "LOCATED_IN", "WORKS_FOR", "PART_OF", "DISCUSSES", "REFERENCES"]
            )
            
            logger.info("Successfully initialized graph transformer")
            
        except Exception as e:
            logger.error(f"Failed to initialize graph transformer: {str(e)}")
            self.graph_transformer = None

    async def extract_and_store_graph(
        self, 
        documents: List[Document], 
        tags_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from documents and store in Neptune.
        
        Args:
            documents: List of Document objects to process
            tags_data: List of dictionaries containing tags and metadata
            
        Returns:
            Dictionary with extraction and storage results
        """
        if not self.enabled:
            logger.warning("Neptune service is disabled")
            return {"status": "disabled", "message": "Neptune service not available"}
        
        if not self.graph_transformer:
            logger.error("Graph transformer not initialized")
            return {"status": "error", "message": "Graph transformer not available"}
        
        try:
            results = {
                "processed_documents": 0,
                "extracted_nodes": 0,
                "extracted_relationships": 0,
                "stored_entities": 0,
                "errors": []
            }
            
            for i, (doc, tags) in enumerate(zip(documents, tags_data)):
                try:
                    # Extract graph elements from document
                    graph_documents = await self._extract_graph_from_document(doc, tags)
                    
                    if graph_documents:
                        # Store in Neptune
                        await self._store_graph_elements(graph_documents, doc, tags)
                        
                        # Update results
                        for graph_doc in graph_documents:
                            results["extracted_nodes"] += len(graph_doc.nodes)
                            results["extracted_relationships"] += len(graph_doc.relationships)
                    
                    results["processed_documents"] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process document {i}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            logger.info(f"Graph extraction completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Graph extraction and storage failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _extract_graph_from_document(
        self, 
        document: Document, 
        tags_data: Dict[str, Any]
    ):
        """Extract graph elements from a single document."""
        try:
            # Use LangChain graph transformer to extract entities and relationships
            graph_documents = await self.graph_transformer.atransform_documents([document])
            
            # Enhance with metadata from tags
            for graph_doc in graph_documents:
                # Add document-level nodes
                doc_node = {
                    "id": f"doc_{uuid.uuid4().hex[:8]}",
                    "type": "Document",
                    "properties": {
                        "title": tags_data.get('tagging_result', {}).get('summary', 'Unknown Document')[:100],
                        "source_url": tags_data.get('source_url', 'unknown'),
                        "content_type": tags_data.get('tagging_result', {}).get('content_type', 'document'),
                        "created_at": datetime.utcnow().isoformat(),
                        "tags": json.dumps(tags_data.get('tagging_result', {}).get('tags', []))
                    }
                }
                graph_doc.nodes.append(doc_node)
                
                # Add relationships from extracted entities to document
                for node in graph_doc.nodes[:-1]:  # Exclude the document node we just added
                    relationship = {
                        "source": node["id"],
                        "target": doc_node["id"],
                        "type": "MENTIONED_IN",
                        "properties": {
                            "confidence": 0.8,
                            "context": "document_extraction"
                        }
                    }
                    graph_doc.relationships.append(relationship)
            
            return graph_documents
            
        except Exception as e:
            logger.error(f"Failed to extract graph from document: {str(e)}")
            return []

    async def _store_graph_elements(
        self, 
        graph_documents: List, 
        original_doc: Document, 
        tags_data: Dict[str, Any]
    ):
        """Store extracted graph elements in Neptune."""
        try:
            for graph_doc in graph_documents:
                # Store nodes (entities)
                for node in graph_doc.nodes:
                    await self._store_node(node)
                
                # Store relationships
                for relationship in graph_doc.relationships:
                    await self._store_relationship(relationship)
            
            logger.info(f"Successfully stored graph elements for document")
            
        except Exception as e:
            logger.error(f"Failed to store graph elements: {str(e)}")
            raise

    async def _store_node(self, node: Dict[str, Any]):
        """Store a single node in Neptune."""
        try:
            # Create Gremlin query for node creation
            properties_str = ", ".join([
                f"property('{k}', '{v}')" for k, v in node.get('properties', {}).items()
            ])
            
            query = f"""
            g.V().hasLabel('{node['type']}').has('id', '{node['id']}').fold().
            coalesce(
                unfold(),
                addV('{node['type']}').property('id', '{node['id']}').{properties_str}
            )
            """
            
            # Execute query using Neptune graph
            self.graph.query(query)
            
        except Exception as e:
            logger.error(f"Failed to store node {node.get('id', 'unknown')}: {str(e)}")

    async def _store_relationship(self, relationship: Dict[str, Any]):
        """Store a single relationship in Neptune."""
        try:
            # Create Gremlin query for relationship creation
            source_id = relationship['source']
            target_id = relationship['target']
            rel_type = relationship['type']
            
            properties_str = ", ".join([
                f"property('{k}', '{v}')" for k, v in relationship.get('properties', {}).items()
            ])
            
            query = f"""
            g.V().has('id', '{source_id}').
            coalesce(
                out('{rel_type}').has('id', '{target_id}'),
                addE('{rel_type}').to(g.V().has('id', '{target_id}')).{properties_str}
            )
            """
            
            # Execute query using Neptune graph
            self.graph.query(query)
            
        except Exception as e:
            logger.error(f"Failed to store relationship {relationship.get('type', 'unknown')}: {str(e)}")

    def search_entities(
        self, 
        entity_type: Optional[str] = None, 
        properties: Optional[Dict[str, str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for entities in the knowledge graph.
        
        Args:
            entity_type: Optional entity type to filter by
            properties: Optional properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        if not self.enabled:
            return []
        
        try:
            # Build Gremlin query
            query = "g.V()"
            
            if entity_type:
                query += f".hasLabel('{entity_type}')"
            
            if properties:
                for key, value in properties.items():
                    query += f".has('{key}', '{value}')"
            
            query += f".limit({limit}).valueMap(true)"
            
            # Execute query
            results = self.graph.query(query)
            
            logger.info(f"Found {len(results)} entities")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search entities: {str(e)}")
            return []

    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_type: Optional[str] = None,
        depth: int = 1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_id: ID of the source entity
            relationship_type: Optional relationship type to filter by
            depth: Traversal depth (1 or 2)
            limit: Maximum number of results
            
        Returns:
            List of related entities with relationship information
        """
        if not self.enabled:
            return []
        
        try:
            # Build Gremlin traversal query
            query = f"g.V().has('id', '{entity_id}')"
            
            if relationship_type:
                if depth == 1:
                    query += f".out('{relationship_type}').limit({limit})"
                else:
                    query += f".out('{relationship_type}').out().limit({limit})"
            else:
                if depth == 1:
                    query += f".out().limit({limit})"
                else:
                    query += f".out().out().limit({limit})"
            
            query += ".valueMap(true)"
            
            # Execute query
            results = self.graph.query(query)
            
            logger.info(f"Found {len(results)} related entities for {entity_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find related entities: {str(e)}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.enabled:
            return {"status": "disabled"}
        
        try:
            # Get node count by type
            node_count_query = "g.V().groupCount().by(label)"
            node_counts = self.graph.query(node_count_query)
            
            # Get total edge count
            edge_count_query = "g.E().count()"
            edge_count = self.graph.query(edge_count_query)
            
            # Get relationship types
            edge_types_query = "g.E().groupCount().by(label)"
            edge_types = self.graph.query(edge_types_query)
            
            stats = {
                "node_counts_by_type": node_counts[0] if node_counts else {},
                "total_edges": edge_count[0] if edge_count else 0,
                "edge_counts_by_type": edge_types[0] if edge_types else {},
                "total_nodes": sum(node_counts[0].values()) if node_counts and node_counts[0] else 0
            }
            
            logger.info(f"Graph statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {str(e)}")
            return {"status": "error", "message": str(e)}

    def clear_graph(self) -> bool:
        """
        Clear all data from the knowledge graph (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Drop all vertices (this will also drop connected edges)
            query = "g.V().drop()"
            self.graph.query(query)
            
            logger.warning("Successfully cleared all data from knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear graph: {str(e)}")
            return False
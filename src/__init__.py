"""
LangChain Web Content Processing Pipeline

A modular pipeline for web content extraction, tagging, embedding, and knowledge graph storage.
"""

__version__ = "1.0.0"
__author__ = "HPACT EdTech Team"
__description__ = "LangChain-based web content processing pipeline"

from .web_crawler import WebCrawler, create_web_crawler
from .pipeline import LangChainPipeline, create_pipeline

__all__ = [
    'WebCrawler',
    'create_web_crawler', 
    'LangChainPipeline',
    'create_pipeline'
]
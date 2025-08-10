"""
EdTech Content Processing System

A modular LangChain-based pipeline for web content processing, tagging,
and vector storage using OpenAI, Qdrant, and advanced web crawling.
"""

from .pipeline import (
    ContentProcessingPipeline,
    ProcessingConfig,
    ProcessingResult,
    create_pipeline,
    create_config
)
from .web_crawler import WebCrawler, CrawlResult, create_web_crawler
from .main import app

__version__ = "1.0.0"
__author__ = "EdTech Team"
__description__ = "Modular LangChain pipeline for content processing"

__all__ = [
    "ContentProcessingPipeline",
    "ProcessingConfig", 
    "ProcessingResult",
    "WebCrawler",
    "CrawlResult",
    "create_pipeline",
    "create_config",
    "create_web_crawler",
    "app"
]
"""
main.py

Entry point for the LangChain pipeline application.
Processes a URL, extracts content, tags, embeds, and stores knowledge graph triples.
"""

from web_crawler import crawl_and_parse
from pipeline import tag_documents, embed_and_store, build_knowledge_graph

def process_url(url):
    """
    Orchestrates crawling, tagging, embedding, and graph storage for a given URL.

    Args:
        url (str): The target URL.
    """
    doc = crawl_and_parse(url)
    docs = [doc]

    tagged_docs = tag_documents(docs)
    embed_and_store(tagged_docs)
    build_knowledge_graph(tagged_docs)

    print("URL processed, embedded in Qdrant, and added to Neptune knowledge graph.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <url>")
        exit(1)
    url = sys.argv[1]
    process_url(url)

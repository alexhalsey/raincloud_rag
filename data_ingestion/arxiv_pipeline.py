from datetime import datetime, timedelta
import fire
import hashlib
from loguru import logger
from pathlib import Path
from pydantic_settings import BaseSettings
import sys
from typing import List, Dict, Any

import arxiv
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
EMBEDDING_DIM = 768

class Config(BaseSettings):
    """configuration management using pydantic."""
    
    # qdrant settings
    qdrant_url: str
    qdrant_api_key: str
    
    # logging settings
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

def setup_logging(logger_name="arxiv_ingest"):
    """Set up logging configuration."""
    
    # create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # configure loguru
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"
            },
            {
                "sink": log_dir / f"{logger_name}.log",
                "rotation": "10 MB",
                "retention": "5 days",
                "format": "{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"
            }
        ]
    }
    
    # remove default handler and configure with new settings
    logger.configure(**config)
    
    return logger

class ArxivIngestCLI:
    """CLI for ingesting ArXiv papers into a vector database."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging(logger_name="arxiv_ingest")

    def _fetch_arxiv_papers(self,
                            content_query: str = "(cybersecurity) OR (cyber security)",
                            days_back: int = 14,
                            end_date: datetime = datetime.today()
                           ) -> List[Dict[str, Any]]:

        self.logger.info(f"topic focus: {content_query}")
        self.logger.info("connecting to ArXiv client")

        # construct the default API client.
        client = arxiv.Client()

        # calculate the date range for the query
        start_date = end_date - timedelta(days=days_back)
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"

        self.logger.info(f"searching the ArXiv for date range {start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}")
        
        search = arxiv.Search(
            query = f"{content_query} AND {date_query}",
            sort_by = arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        for result in client.results(search):
            paper_info = {
                "paper_id": result.entry_id.split('/')[-1],  # e.g. '2301.12345v1'
                "title": result.title,
                "abstract": result.summary,
                "authors": [str(author.name) for author in result.authors],
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat()
            }
            papers.append(paper_info)

        self.logger.info(f"retrieved {len(papers)} papers from ArXiv")

        return papers
    
    def _create_text_nodes(self,
                          papers: List[Dict[str, Any]]) -> List[TextNode]:

        self.logger.info("initializing chunking")

        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=50,
        )

        all_nodes = []
        for i, record in enumerate(papers):
            
            if i % 100 == 0:
                self.logger.info(f"chunking record {i}")

            text = f"{record['title']}. {record['abstract']}".replace("\n", " ")

            # for checking if records already exist
            content_hash = hashlib.md5(text.encode()).hexdigest()

            doc = [Document(
                text=text,
                metadata={
                    "paper_id": record["paper_id"],
                    "pub_date": record["published"],
                    "title": record["title"],
                    "content_hash": content_hash
                }
            )]

            # probably not necessary to chunk, but in case there are very long abstracts
            nodes = splitter.get_nodes_from_documents(doc)

            all_nodes.append(nodes)

        flattened = [item for sublist in all_nodes for item in sublist]

        self.logger.info(f"produced {len(flattened)} text chunks with metadata")

        return flattened
    
    # https://github.com/run-llama/llama_index/issues/12786
    def _create_index(self,
                      nodes: List[TextNode],
                      collection_name: str = "cyber_arxiv"):
        
        self.logger.info("connecting to QDrant client...")

        # debug environment variables
        qdrant_url = self.config.qdrant_url
        qdrant_api_key = self.config.qdrant_api_key
        
        self.logger.info(f"QDRANT_URL: {qdrant_url}")  
        if qdrant_api_key is None:
            self.logger.warning("QDRANT_API_KEY is not set")

        # qdrant set-up on disk
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        # Check if collection exists, if not create it
        try:
            _ = qdrant_client.get_collection(collection_name)
            self.logger.info(f"found existing collection: {collection_name}")
        except (UnexpectedResponse, ValueError):
            self.logger.info(f"creating new collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )

        vector_store = QdrantVectorStore(
            client=qdrant_client, 
            collection_name=collection_name
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # process nodes and check against existing records
        self.logger.info("checking for duplicate records...")
        processed_nodes = []
        for node in nodes:
            # Check if this hash exists in QDrant
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="content_hash",
                        match=MatchValue(value=node.metadata["content_hash"])
                    )
                ]
            )
            
            # search for existing records with this hash
            search_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=1
            )
            
            # if no matching record found, add to processed nodes
            if len(search_result[0]) == 0:
                processed_nodes.append(node)
        
        self.logger.info(f"original nodes: {len(nodes)}")
        self.logger.info(f"new unique nodes to be added: {len(processed_nodes)}")

        if processed_nodes:
            self.logger.info("attempting to upload to cloud db")
            _ =  VectorStoreIndex(processed_nodes, 
                                storage_context=storage_context, 
                                include_embeddings=True)

    def ingest(self,
               content_query: str = "(cybersecurity) OR (cyber security)",
               days_back: int = 14,
               end_date: str = None,
               collection_name: str = "cyber_arxiv"):
        """
        Ingest papers from ArXiv within the specified date range.

        Args:
            content_query: search query for ArXiv papers (default: cybersecurity related)
            days_back: number of days to look back for papers (default: 7)
            end_date: end date in YYYY-MM-DD format (default: today)
            collection_name: name of the QDrant collection (default: cyber_arxiv)
        """
        try:
            # Parse end date if provided, otherwise use today
            if end_date:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_date_dt = datetime.today()

            self.logger.info(f"starting ingestion process")
            
            # Fetch papers
            papers = self._fetch_arxiv_papers(
                content_query=content_query,
                days_back=days_back,
                end_date=end_date_dt
            )
            
            if not papers:
                self.logger.info("no papers found matching criteria")
                return
            
            # Create text nodes
            nodes = self._create_text_nodes(papers)
            
            if not nodes:
                self.logger.info("no nodes created from papers")
                return
            
            # Create or update index
            self._create_index(
                nodes=nodes,
                collection_name=collection_name
            )
            
            self.logger.info("ingestion process completed successfully")
            return 0

        except Exception as e:
            self.logger.error(f"error during ingestion: {str(e)}", exc_info=True)
            return 1

def main():
    """
    main entry point for the CLI.
    """
    fire.Fire(ArxivIngestCLI)

if __name__ == "__main__":
    main()


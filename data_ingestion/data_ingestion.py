import os
from typing import List, Dict, Generator, Optional
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
import time
from datetime import datetime

from loguru import logger
from datasets import load_dataset
from qdrant_client import QdrantClient
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

import keywords

def setup_logger():
    # configure loguru logger with file and console outputs
    logger.remove()  # remove default handler
    logger.add(
        f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    logger.add(
        lambda msg: print(msg),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        colorize=True
    )

@dataclass
class DataSourceConfig:
    # configuration for data source and processing parameters
    name: str  # collection name in qdrant
    dataset_path: Optional[str] = None  # huggingface dataset path
    dataset_name: Optional[str] = None  # specific configuration of dataset
    keywords: List[str] = None  # keywords to filter documents
    min_keyword_matches: int = 3  # minimum number of keywords required
    chunk_size: int = 1024  # size of text chunks for processing
    chunk_overlap: int = 50  # overlap between chunks
    batch_size: int = 1000  # number of documents per batch
    processes: int = 4  # number of parallel processes


class DataIngestionPipeline:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embedding_model: str = "BAAI/bge-base-en-v1.5"
    ):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        Settings.embed_model = FastEmbedEmbedding(model_name=embedding_model)
        self.splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
        setup_logger()

    def _filter_by_keywords(self, dataset: Generator, keywords: List[str], min_matches: int) -> Generator:
        processed = 0
        matched = 0
        start_time = time.time()
        
        for item in dataset:
            processed += 1
            keyword_count = sum(1 for keyword in keywords
                              if keyword.lower() in item['text'].lower() or 
                              keyword.lower() in item.get('title', '').lower())
            
            if keyword_count >= min_matches:
                matched += 1
                if matched % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    logger.info(f"Processed {processed} records, matched {matched} (Rate: {rate:.2f} records/sec)")
                yield item

    def _create_nodes(self, texts: List[Dict]) -> List:
        nodes = []
        for record in texts:
            text = record["text"].replace("\n", " ")
            doc = [Document(text=text)]
            curr_nodes = self.splitter.get_nodes_from_documents(doc)
            nodes.extend(curr_nodes)
        return nodes

    def _create_index_chunk(self, nodes: List, collection_name: str) -> VectorStoreIndex:
        logger.info(f"Creating index for chunk of {len(nodes)} nodes in {collection_name}")
        start_time = time.time()
        
        vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, include_embeddings=True)
        
        elapsed = time.time() - start_time
        logger.info(f"Finished indexing chunk in {elapsed:.2f} seconds (Rate: {len(nodes)/elapsed:.2f} nodes/sec)")
        return index

    def process_dataset(self, config: DataSourceConfig):
        logger.info(f"Starting ingestion for dataset: {config.name}")
        start_time = time.time()
        
        dataset = load_dataset(f"{config.dataset_path}/{config.dataset_name}", streaming=True)
        data_stream = dataset["train"]
        
        if config.keywords:
            logger.info(f"Filtering by {len(config.keywords)} keywords, minimum matches: {config.min_keyword_matches}")
            data_stream = self._filter_by_keywords(data_stream, config.keywords, config.min_keyword_matches)

        nodes = []
        total_processed = 0
        
        for record in data_stream:
            curr_nodes = self._create_nodes([record])
            nodes.extend(curr_nodes)
            total_processed += 1

            if len(nodes) >= config.batch_size:
                self._process_nodes_batch(nodes, config)
                logger.info(f"Processed batch of {len(nodes)} nodes. Total records: {total_processed}")
                nodes = []

        if nodes:
            self._process_nodes_batch(nodes, config)
            logger.info(f"Processed final batch of {len(nodes)} nodes. Total records: {total_processed}")

        elapsed = time.time() - start_time
        logger.success(f"Completed ingestion of {config.name}. Total time: {elapsed:.2f} seconds")

    def _process_nodes_batch(self, nodes: List, config: DataSourceConfig):
        logger.info(f"Processing batch of {len(nodes)} nodes with {config.processes} processes")
        chunks = [nodes[i:i + config.batch_size] for i in range(0, len(nodes), config.batch_size)]
        
        with Pool(processes=config.processes) as pool:
            pool.map(
                partial(self._create_index_chunk, collection_name=config.name),
                chunks
            )

arxiv_config = DataSourceConfig(
    name="arxiv_security",
    dataset_path="arxiv-community",
    dataset_name="arxiv_dataset",
    keywords=keywords.cyber_keywords,
    min_keyword_matches=2
)

# Initialize and run pipeline
pipeline = DataIngestionPipeline(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY")
)

# Process different datasets
pipeline.process_dataset(arxiv_config)
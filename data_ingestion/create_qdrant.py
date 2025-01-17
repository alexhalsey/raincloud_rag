import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import uuid
import time

import qdrant_client
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from multiprocessing import Pool
from datasets import load_dataset

from nltk import sent_tokenize

# read in HF dataset
# streaming = True for generator
wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

cybersecurity_keywords = [
    " Authentication ",
    " Authorization ",
    " Encryption ",
    " Decryption ",
    " Firewall ",
    " Malware ",
    " Virus ",
    " Trojan Horse ",
    " Spyware ",
    " Ransomware ",
    " Phishing ",
    " Social Engineering ",
    " Vulnerability ",
    " Patch ",
    " Exploit ",
    " Zero-Day ",
    " Denial of Service ",
    " Distributed Denial of Service ",
    " Intrusion Detection System ",
    " Intrusion Prevention System ",
    " Penetration Testing ",
    " Endpoint Security ",
    " Network Security ",
    " Application Security ",
    " Access Control ",
    " Public Key Infrastructure ",
    " Cryptography ",
    " Data Breach ",
    " Cyber Threat Intelligence ",
    " Security Operations Center ",
    " Incident Response ",
    " Forensics ",
    " Hashing ",
    " Tokenization ",
    " Multi-Factor Authentication ",
    " Privileged Access Management ",
    " Threat Modeling ",
    " Security Information and Event Management ",
    " Vulnerability Assessment ",
    " Risk Assessment ",
    " Operating System ",
    " Artificial Intelligence ",
    " Cybersecurity ",
    " Linux ",
    " Windows "
]

def create_bisentences_with_context(sentences):
    """
    Generate bi-sentence chunks with context metadata.
    Args:
        sentences (List[str]): List of sentences from the article.
    Returns:
        List[dict]: List of dictionaries with bi-sentence pairs and context metadata.
    """
    chunks = []
    for i in range(len(sentences) - 1):
        # Form the bi-sentence
        bi_sentence = sentences[i] + " " + sentences[i + 1]
        
        # Context sentences: previous and next
        prev_sentence = sentences[i - 1] if i > 0 else None
        next_sentence = sentences[i + 2] if i + 2 < len(sentences) else None
        
        # Store each bi-sentence with its context
        chunks.append({
            "bi_sentence": bi_sentence,
            "prev_sentence": prev_sentence,
            "next_sentence": next_sentence
        })
    return chunks

# about 4400 wikipedia pages that have 3 of these words
def filter_articles(dataset, keywords):
    for article in dataset:
        # Count occurrences of keywords in title and initial text
        keyword_count = sum(
            1 for keyword in keywords 
            if keyword.lower() in article['text'].lower() or keyword.lower() in article['title'].lower()
        )
        
        # Yield the article if three or more keywords are present
        if keyword_count >= 3:
            yield article

# Apply the filter function with streaming
cyber_articles = filter_articles(wiki_dataset["train"], cybersecurity_keywords)

# set up embedding model
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
embedding_model = Settings.embed_model

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=50,
)

all_nodes = []
for i, record in enumerate(cyber_articles):
    if i % 100 == 0:
        print(i)
    text = record["text"].replace("\n", " ")
    title = record["title"]  # Example metadata: title, set default if missing

    doc = [Document(
        text=text
    )]

    nodes = splitter.get_nodes_from_documents(doc)

    all_nodes.append(nodes)

flattened = [item for sublist in all_nodes for item in sublist]

# https://github.com/run-llama/llama_index/issues/12786
def create_index(nodes):
    # qdrant set-up on disk
    qdrant_client = QdrantClient(
        url="https://b9eeafd3-acd8-4903-a2e0-6051ad225cc9.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection_name = "cyber_wiki"
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name=collection_name
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex(nodes, 
                            storage_context=storage_context, 
                            include_embeddings=True)

# Split nodes into manageable chunks
chunks = [flattened[i:i + 1000] for i in range(0, len(flattened), 1000)]

# Process chunks in parallel
with Pool(processes=4) as pool:
    indices = pool.map(create_index, chunks)

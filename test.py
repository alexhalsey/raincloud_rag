import os
os.environ["OPENAI_API_KEY"] = "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"


# from llama_index.llms.openai import OpenAI

# resp = OpenAI().complete("Paul Graham is ")

# print(resp)


import logging
import sys
import os
import uuid

import qdrant_client
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import StorageContext

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings

from llama_index.core.node_parser import SentenceSplitter

from datasets import load_dataset

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

    collection_name = "cyber_wiki"
    vector_store = BaseVectorStore(
        client=qdrant_client, 
        collection_name=collection_name
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex(nodes, 
                            storage_context=storage_context, 
                            include_embeddings=True)

retriever = keyword_index.as_retriever()

nodes = retriever.retrieve("what are the main security vulnerabilities of linux?")









































































ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

ds = iter(ds["train"])
print(next(ds))
print(next(ds))
print(next(ds))


from datasets import load_dataset

# Load the Wikipedia dataset in streaming mode
wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

# Define keywords related to computer science
keywords = ["computer science", "cybersecurity", "algorithms", "data science", 
            "machine learning", "artificial intelligence", "cryptography", 
            "software engineering"]

# Create a generator function to filter articles
def filter_cs_articles(dataset, keywords):
    for article in dataset:
        # Check if any keyword is in the title or initial text of the article
        if any(keyword.lower() in article['text'][:500].lower() or keyword.lower() in article['title'].lower() for keyword in keywords):
            yield article

# Apply the filter function with streaming
cs_articles = filter_cs_articles(wiki_dataset["train"], keywords)

# Example: Printing the first 5 articles that match
for i, article in enumerate(cs_articles):
    if i >= 5:  # Limit to 5 for preview purposes
        break
    print(article["text"][:100])



### try to create qdrant db
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
embedding_model = Settings.embed_model

qdrant_client = QdrantClient(
    path="/mnt/c/Users/alexa/OneDrive/Desktop/qdrant_dir/"
)

collection_name = "test_wiki"

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

single_record = PointStruct(
    id=str(uuid.uuid4()),
    vector=[0.1] * 768,  # Replace with an actual sample embedding
    payload={"text": "test text"}
)

qdrant_client.upsert(collection_name=collection_name, points=[single_record])

# Scroll through to verify insertion
scroll_result = qdrant_client.scroll(collection_name=collection_name, limit=10, with_vectors=True)
print(scroll_result)


# Initialize QdrantVectorStore with the Qdrant client and specify the collection name
vector_store = QdrantVectorStore(
    collection_name=collection_name,  # Replace with your chosen collection name
    client=qdrant_client,
)

# Configure the storage context with Qdrant as the vector store
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

batch_size = 20  # Adjust batch size based on memory and performance testing
batch = []

# Apply the filter function with streaming
cs_articles = filter_cs_articles(wiki_dataset["train"], keywords)

for i, record in enumerate(cs_articles):

    if i > 100:
        break
    text = record["text"]
    title = record["title"]  # Example metadata: title, set default if missing

    # Generate embeddings using the specified embed model in llama-index
    embedding = embedding_model.get_text_embedding(text)
    
    point_id = str(uuid.uuid4())

    # Prepare PointStruct object with metadata in Qdrant's format
    point = PointStruct(
        id=point_id,  # Ensure ID is a string
        vector=embedding,
        payload={
            "text": text,
            "title": title
        }
    )


    batch.append(point)

    if len(batch) >= batch_size:
        print("upsert happening")
        print(point)
        # Directly upsert batch into Qdrant collection
        qdrant_client.upsert(collection_name=collection_name, points=batch)
        batch.clear()  # Clear the batch after uploading

# Example: Scroll through all records in the collection
scroll_limit = 1  # Number of records to retrieve per scroll
offset = None

while True:
    scroll_result = qdrant_client.scroll(
        collection_name=collection_name,
        limit=scroll_limit,
        offset=offset
    )
    records = scroll_result[0]
    offset = scroll_result[1]

    # Print out records
    for record in records:
        print(f"ID: {record.vector}")
        #print(f"ID: {record.id}, Vector: {record.vector}, Payload: {record.payload}")
    break
    # Break when there are no more records
    if offset is None:
        break

qdrant_client = None



import pandas as pd
df = pd.DataFrame.from_records(cs_articles)

vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
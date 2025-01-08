import os

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import PromptTemplate, Settings, QueryBundle
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.query_engine import CustomQueryEngine

from copy import deepcopy

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
embedding_model = Settings.embed_model

def generate_completion_to_prompt(system_prompt):
    """
    Creates a function that formats a user completion with a system prompt
    for the OpenAI API in a standard format.

    Args:
        system_prompt (str): The initial system-level prompt to set up context.

    Returns:
        function: A function that takes a user completion string and returns 
                  it formatted for the OpenAI API.
    """
    def completion_to_prompt(completion):
        """
        Formats a user completion string for the OpenAI API.

        Args:
            completion (str): The user's input or query to be transformed.

        Returns:
            str: The formatted prompt string ready for the OpenAI API.
        """
        return f"{system_prompt}\nUser: {completion}\nAssistant: "

    return completion_to_prompt

import re
from llama_index.llms.openai import OpenAI
llm_kwargs = {"temperature": 0.5, "do_sample":True}
system_prompt = "You are a computer scientist with a specialization in assessing cybersecurity risks. You are extremely thorough, accurate, and are skilled at communicating clearly. You do not use information outside the provided text."
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    generate_kwargs=llm_kwargs,
    completion_to_prompt=generate_completion_to_prompt(system_prompt),
)

#llm = Settings.llm
#llm.complete("FastAPI is")

def execute_hyde(user_prompt):

    # generate prompt to rewrite the user prompt
    prompt_str_1 = "Please edit the following prompt to be a more precisely phrased, detailed question while preserving the original meaning: {user_prompt}"
    prompt_tmpl_1 = PromptTemplate(prompt_str_1)

    # generate prompt for hallucinating answer
    prompt_str2 = (
        "Please write a passage to answer the question\n"
        "Try to include as many key details as possible.\n"
        "\n"
        "\n"
        "{query_str}\n"
        "\n"
        "\n"
        'Passage:"""\n'
    )
    prompt_tmpl2 = PromptTemplate(prompt_str2)

    # first query pipeline is for HyDE
    # generates better question and sample answer for lookup
    hyde_pipeline = QueryPipeline(
        chain=[prompt_tmpl_1, Settings.llm, prompt_tmpl2, Settings.llm], verbose=True
    )

    hyde_results, hyde_intermediates = hyde_pipeline.run_with_intermediates(user_prompt)

    # # get the prompt rewrite and the hypothetical answer
    # hyde_prompt = hyde_intermediates[list(hyde_intermediates.keys())[1]].outputs["output"].text
    strip_prefix = lambda x: re.sub(r'^assistant:\s*', '', x, flags=re.IGNORECASE)
    hyde_answer = strip_prefix(str(hyde_results))
    hyde_prompt = str(hyde_intermediates[list(hyde_intermediates.keys())[1]].outputs['output'])
    hyde_prompt = strip_prefix(hyde_prompt)

    return {"hyde_prompt": hyde_prompt,
            "hyde_answer": hyde_answer}        


def get_retriever(retriever_kwargs):
    similarity_top_k = retriever_kwargs["similarity_top_k"]
    #metadata_dict = retriever_kwargs["metadata_dict"]

    # vector index setup
    collection_name = retriever_kwargs["collection"]

    qdrant_client = QdrantClient(
        url="https://b9eeafd3-acd8-4903-a2e0-6051ad225cc9.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="LtLPxnJSmqaXuHlSWU077IfOoq-iCjzMB6wogMKAJdl9Hzhl0JGOvQ"
    )

    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # retriever!
    # we can improve this using a custom retriever to combine multiple types
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/
    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k
    )

    return retriever

def get_reranker(reranker_kwargs):

    # reranker!
    # https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/SentenceTransformerRerank/
    top_n = reranker_kwargs["top_n"]
    reranker = CohereRerank(api_key=os.environ["COHERE_API_KEY"], 
                            top_n=top_n)

    # reranker = SentenceTransformerRerank(
    #     model=reranker_model_name, top_n=top_n
    # )

    return reranker


class RAGQueryEngine(CustomQueryEngine):
    # custom query engine!
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine/
    retriever: BaseRetriever
    rerank: bool
    reranker: CohereRerank
    llm: OpenAI
    qa_prompt: PromptTemplate
    retrieval_str: str


    def custom_query(self, query_str: str):
        # retrieve from db based on retrieval_str (hallucinated answer)
        source_nodes = self.retriever.retrieve(self.retrieval_str)
        retrieved_nodes = deepcopy(source_nodes)
        
        if self.rerank:
            # rerank based on similarity to prompt
            # either rewritten prompt (if doing HYDE)
            # or base user prompt
            source_nodes = self.reranker.postprocess_nodes(source_nodes, QueryBundle(query_str))

        context_str = "\n\n".join([n.node.get_content() for n in source_nodes])
        final_prompt =  self.qa_prompt.format(context_str=context_str, query_str=query_str)
        
        response = self.llm.complete(final_prompt)

        return response, retrieved_nodes, source_nodes, final_prompt

def execute_rag(pipeline_kwargs, pipeline_objs):

    # parse the pipeline args
    user_prompt = pipeline_kwargs["user_prompt"]
    rerank = pipeline_kwargs["rerank"]
    hyde = pipeline_kwargs["hyde"]
    #sentence_window = pipeline_kwargs["sentence_window"]

    # execute HyDE
    if hyde:
        hyde_results = execute_hyde(user_prompt=user_prompt)

        # if doing HyDE we use the hallucinated answer for retrieval
        # and ask the LLM to answer the rewritten question
        retrieval_str = hyde_results["hyde_answer"]
        query_str = hyde_results["hyde_prompt"]

    else:  
        retrieval_str = user_prompt
        query_str = user_prompt

    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    # initialize custom query engine
    query_engine = RAGQueryEngine(
        retriever=pipeline_objs["retriever"], 
        rerank=rerank,
        reranker=pipeline_objs["reranker"],
        llm=Settings.llm, 
        qa_prompt=qa_prompt, 
        retrieval_str=retrieval_str
    )

    response, retrieved_nodes, source_nodes, final_prompt = query_engine.query(query_str)

    return {"response": response,
            "retrieved_nodes": retrieved_nodes,
            "source_nodes": source_nodes,
            "final_prompt": final_prompt,
            "query_str": query_str,
            "retrieval_str": retrieval_str}


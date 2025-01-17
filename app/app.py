from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import streamlit as st
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai import OpenAI

from rag_dev import (
    Config,
    get_config,
    execute_rag,
    get_retriever,
    get_reranker,
    generate_completion_to_prompt
)

COLLECTIONS = ["cyber_wiki", "cyber_arxiv"]
DEFAULT_TEMPERATURE = 0.5
DEFAULT_K = 5
DEFAULT_N = 3
SYSTEM_PROMPT = (
        "You are a computer scientist with a specialization in assessing cybersecurity risks. "
        "You are extremely thorough, accurate, and skilled at communicating clearly. "
        "You do not use information outside the provided text."
    )

@dataclass
class PipelineConfig:
    temperature: float
    use_hyde: bool
    use_rerank: bool
    num_sources: int
    num_rerank: int
    collection: str

class PipelineState:
    def __init__(self):
        self.retriever = None
        self.reranker = None
        self.config = None
        self.env_config = get_config()

    def update_if_needed(self, new_config: PipelineConfig) -> None:
        if not self.config or self._config_changed(new_config):
            self._reinitialize(new_config)
            self.config = new_config

    def _config_changed(self, new_config: PipelineConfig) -> bool:
        if not self.config:
            return True
        return (
            self.config.temperature != new_config.temperature or
            self.config.num_sources != new_config.num_sources or
            self.config.num_rerank != new_config.num_rerank or
            self.config.collection != new_config.collection
        )

    def _reinitialize(self, config: PipelineConfig) -> None:
        os.environ["OPENAI_API_KEY"] = self.env_config.openai_api_key
        if not self.config or self.config.temperature != config.temperature:
            Settings.llm = OpenAI(
                model="gpt-4o-mini",
                generate_kwargs={"temperature": config.temperature, "do_sample": True},
                completion_to_prompt=generate_completion_to_prompt(SYSTEM_PROMPT),
            )

        if not self.config or (
            self.config.num_sources != config.num_sources or
            self.config.collection != config.collection
        ):
            self.retriever = get_retriever({
                "similarity_top_k": config.num_sources,
                "collection": config.collection
            })

        if not self.config or self.config.num_rerank != config.num_rerank:
            self.reranker = get_reranker({"top_n": config.num_rerank}, self.env_config)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = PipelineState()

def setup_pipeline():
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

def render_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for idx, source_info in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx + 1}** (Score: {source_info['score']}):")
                        st.write(source_info['content'])

def handle_chat_input(config: PipelineConfig):
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        pipeline_kwargs = {
            "rerank": config.use_rerank,
            "hyde": config.use_hyde,
            "user_prompt": prompt,
        }

        state = st.session_state.pipeline_state
        results = execute_rag(
            pipeline_kwargs=pipeline_kwargs,
            pipeline_objs={
                "retriever": state.retriever,
                "reranker": state.reranker
            }
        )

        sources_list = [
            {"content": node.node.get_content(), "score": node.score}
            for node in results["source_nodes"]
        ]

        st.session_state.messages.append({
            "role": "assistant",
            "content": results["response"],
            "sources": sources_list
        })

        st.rerun()

def main():
    init_session_state()
    st.set_page_config(page_title="RainCloudRAG", layout="wide")
    st.title("Welcome to RainCloud RAG!")

    # load configuration at startup
    env_config = get_config()
    os.environ["OPENAI_API_KEY"] = env_config.openai_api_key

    config = PipelineConfig(
        temperature=st.sidebar.slider("LLM Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1),
        use_hyde=st.sidebar.checkbox("Use HyDE", value=True),
        use_rerank=st.sidebar.checkbox("Use Re-ranker", value=True),
        num_sources=st.sidebar.number_input("Number of Sources to Retrieve", 3, 20, DEFAULT_K),
        num_rerank=st.sidebar.number_input("Number of Re-ranked Sources", 1, 20, DEFAULT_N),
        collection=st.sidebar.selectbox("Dataset", COLLECTIONS)
    )

    st.session_state.pipeline_state.update_if_needed(config)
    render_messages()
    handle_chat_input(config)

# Run setup + main
if __name__ == "__main__":
    setup_pipeline()
    main()

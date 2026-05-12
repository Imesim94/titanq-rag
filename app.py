import os
import sys
import json
import time
import yaml
import logging
from typing import Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager

from langchain_ollama import OllamaEmbeddings, ChatOllama # format my text into API calls that Ollama understands, 
#triggering the nomic-embed-text and mistral models natively
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure robust, cross-platform logging (ASCII safe for Windows)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("titanq_rag")

@contextmanager
def track_latency(metrics_dict: Dict[str, float], key: str):
    """Context manager for elegant, non-intrusive latency telemetry."""
    start = time.perf_counter()
    yield
    metrics_dict[key] = round((time.perf_counter() - start) * 1000, 2)

class InsightEngine:
    """Production-grade RAG Pipeline leveraging Ollama and ChromaDB."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._init_models()
        self.retriever = None
        self.chain = None

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing configuration: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _init_models(self) -> None:
        logger.info("Initializing local Ollama models...")
        self.embeddings = OllamaEmbeddings(model=self.config["retrieval"]["embedding_model"])
        self.llm = ChatOllama(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
        )

    def ingest_data(self, data_path: str) -> None:
        """Loads JSON documents and builds the vector index."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset missing: {data_path}")

        with open(data_path, "r") as f:
            data = json.load(f)

        texts = [item["content"] for item in data if "content" in item]
        if not texts:
            raise ValueError("No valid content found in dataset.")

        logger.info(f"Building vector index for {len(texts)} documents...")
        
        # Using timezone-aware datetimes to resolve Python 3.12+ deprecation warnings
        metadatas = [{"source": "TitanQIO Dataset", "ingested_at": datetime.now(timezone.utc).isoformat()} for _ in texts]

        vectorstore = Chroma.from_texts(texts, embedding=self.embeddings, metadatas=metadatas)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": self.config["retrieval"]["k_results"]})
        self._build_chain()

    def _build_chain(self) -> None:
        """Constructs the LCEL (LangChain Expression Language) pipeline."""
        template = """You are a precise enterprise AI assistant. Answer using ONLY the context below.
        If the answer is not in the context, state "I lack the required context to answer."

        Context: {context}
        Question: {question}
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, user_query: str) -> Dict[str, Any]:
        """Executes a query with full telemetry and graceful degradation."""
        if not self.chain:
            raise RuntimeError("Pipeline not initialized. Call ingest_data() first.")

        metrics = {}
        result = {
            "query": user_query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunks": [],
            "response": None,
            "error": None
        }

        try:
            # Elegant telemetry wrapping
            with track_latency(metrics, "retrieval_ms"):
                docs = self.retriever.invoke(user_query)
                result["chunks"] = [doc.page_content for doc in docs]

            with track_latency(metrics, "generation_ms"):
                result["response"] = self.chain.invoke(user_query)

            result["telemetry"] = metrics
            result["telemetry"]["total_ms"] = round(metrics["retrieval_ms"] + metrics["generation_ms"], 2)

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            result["error"] = str(e)
            result["response"] = "Service temporarily unavailable due to an internal error."

        return result

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "="*60)
    print(" TitanQIO RAG System — Initializing")
    print("="*60)

    try:
        pipeline = InsightEngine(os.path.join(current_dir, "config.yaml"))
        pipeline.ingest_data(os.path.join(current_dir, "data.json"))
    except Exception as e:
        logger.critical(f"Fatal initialization error: {e}")
        sys.exit(1)

    queries = [
        "What does TitanQIO specialise in?",
        "How does TitanQIO use RAG?",
        "What technologies are commonly used?",
        "What are the main use cases of TitanQIO systems?",
        "Describe a possible business scenario where this system is used."
    ]

    # Execution & Display
    success_count = 0
    total_latency = 0.0

    for q in queries:
        print(f"\n{'-'*60}")
        print(f"User Query: {q}")
        
        res = pipeline.query(q)
        
        if res["error"]:
            print(f" [!] Error: {res['error']}")
            continue

        success_count += 1
        total_latency += res["telemetry"]["total_ms"]

        print("\n[ Retrieved Context ]")
        for i, chunk in enumerate(res["chunks"], 1):
            print(f"  {i}. {chunk[:120]}...")
            
        print(f"\n[ Assistant ]\n  {res['response']}")
        print(f"\n  ⏱️ Latency: Retrieval {res['telemetry']['retrieval_ms']}ms | Generation {res['telemetry']['generation_ms']}ms")

    # Session Rollup
    if success_count > 0:
        print("\n" + "="*60)
        print(f" Session Rollup: {success_count}/{len(queries)} successful | Avg Latency: {round(total_latency/success_count, 2)}ms")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
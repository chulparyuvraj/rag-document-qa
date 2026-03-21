
import os
from typing import Optional, List
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch
from loguru import logger

from .prompt_templates import RAG_PROMPT
from ..ingestion.pdf_loader import PDFLoader
from ..ingestion.chunker import ResearchPaperChunker
from ..retrieval.embeddings import get_embedding_model
from ..retrieval.vector_store import FAISSVectorStore
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.hybrid_retriever import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = None,
        llm_model_name: str = None,
        adapter_path: Optional[str] = None,
        index_path: str = "indexes/faiss",
        device: str = "auto",
        top_k: int = 5,
    ):
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm_model_name = llm_model_name or os.getenv(
            "BASE_LLM", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.adapter_path = adapter_path or os.getenv("FINETUNED_ADAPTER_PATH")
        self.index_path = index_path
        self.device = device
        self.top_k = top_k
        self._embeddings = None
        self._faiss_store = None
        self._bm25 = None
        self._hybrid_retriever = None
        self._llm_pipeline = None
        self._chunks: List[Document] = []

    def build_index(self, pdf_dir="data/papers", save_index=True, force_rebuild=False):
        self._embeddings = get_embedding_model(self.embedding_model_name)
        self._faiss_store = FAISSVectorStore(self._embeddings)

        if not force_rebuild and Path(self.index_path).exists():
            logger.info("Loading existing FAISS index from disk...")
            self._faiss_store.load(self.index_path)
            self._chunks = list(self._faiss_store._index.docstore._dict.values())
        else:
            logger.info("Building new index from PDFs...")
            loader = PDFLoader(source_dir=pdf_dir)
            pages = loader.load_all()
            chunker = ResearchPaperChunker(chunk_size=512, chunk_overlap=64)
            self._chunks = chunker.chunk(pages)
            self._faiss_store.build(self._chunks)
            if save_index:
                self._faiss_store.save(self.index_path)

        self._bm25 = BM25Retriever.from_documents(self._chunks, k=self.top_k)
        self._hybrid_retriever = HybridRetriever(
            faiss_store=self._faiss_store,
            bm25_retriever=self._bm25,
            k=self.top_k,
        )
        self._llm_pipeline = self._load_llm()
        logger.info("RAG pipeline ready!")

    def query(self, question: str) -> dict:
        if self._llm_pipeline is None:
            raise RuntimeError("Call build_index() first.")
        logger.info(f"Query: {question}")

        # Retrieve relevant docs
        source_docs = self._hybrid_retriever.invoke(question)

        # Build context string from retrieved chunks
        context = "\n\n".join([doc.page_content for doc in source_docs])

        # Format prompt manually — no LangChain chain needed
        prompt_text = RAG_PROMPT.format(context=context, question=question)

        # Run through HuggingFace pipeline directly
        response = self._llm_pipeline.pipeline(
            prompt_text,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.15,
        )
        answer = response[0]["generated_text"][len(prompt_text):].strip()

        return {
            "result": answer,
            "source_documents": source_docs,
        }

    def get_context(self, question: str) -> List[Document]:
        return self._hybrid_retriever.invoke(question)

    def _load_llm(self) -> HuggingFacePipeline:
        logger.info(f"Loading LLM: {self.llm_model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
        )
        if self.adapter_path and Path(self.adapter_path).exists():
            logger.info(f"Loading QLoRA adapter from {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.15,
        )
        return HuggingFacePipeline(pipeline=pipe)

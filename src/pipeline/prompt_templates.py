"""
prompt_templates.py
───────────────────
Prompt templates for the RAG chain.
Tuned for research paper QA — instructs the model to cite sources
and acknowledge when the answer is not in the retrieved context.
"""

from langchain_core.prompts import PromptTemplate

# ── Main RAG Prompt ───────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an expert research assistant specializing in \
academic papers and scientific literature.

Use ONLY the following retrieved context to answer the question.
If the answer cannot be found in the context, say:
"I could not find a direct answer in the provided documents."
Do NOT use prior knowledge outside the context.

Always cite the source paper and page number when referencing specific claims.

Context:
{context}

Question: {question}

Answer (be concise and cite sources):"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

# ── Summarization Prompt (for long contexts) ──────────────────────────────────
SUMMARIZE_PROMPT_TEMPLATE = """Summarize the following research paper excerpt \
in 3-5 sentences, focusing on the key findings and methodology:

{text}

Summary:"""

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=SUMMARIZE_PROMPT_TEMPLATE,
)

# ── Query Expansion Prompt (improves recall) ──────────────────────────────────
QUERY_EXPANSION_TEMPLATE = """Given the following research question, generate \
3 alternative phrasings that capture the same information need. \
Return only the alternative questions, one per line.

Original question: {question}

Alternative phrasings:"""

QUERY_EXPANSION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=QUERY_EXPANSION_TEMPLATE,
)

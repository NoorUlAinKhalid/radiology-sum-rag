"""
RAG (Retrieval-Augmented Generation) module.
Retrieves similar clean findings from a knowledge base to assist
summarization of noisy input.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    """
    Simple semantic RAG system.
    Given a noisy query finding, retrieves top-k similar
    clean findings+impression pairs and prepends them as context.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.knowledge_base = []
        self.kb_embeddings = None

    def build_index(self, records: list):
        """
        Build a retrieval index from a list of clean records.

        Args:
            records: List of dicts with 'findings' and 'impression'
        """
        self.knowledge_base = records
        texts = [r["findings"] for r in records]
        print(f"Building RAG index from {len(texts)} records...")
        self.kb_embeddings = self.embedder.encode(texts, show_progress_bar=True)
        print("Index built.")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve top-k most similar records to the query.

        Args:
            query: Noisy findings text
            top_k: Number of records to retrieve

        Returns:
            List of retrieved record dicts
        """
        if self.kb_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_emb = self.embedder.encode([query])
        # Cosine similarity
        norms_kb = np.linalg.norm(self.kb_embeddings, axis=1, keepdims=True)
        norms_q = np.linalg.norm(query_emb, axis=1, keepdims=True)
        sims = (self.kb_embeddings @ query_emb.T) / (norms_kb * norms_q + 1e-8)
        top_indices = np.argsort(sims[:, 0])[::-1][:top_k]
        return [self.knowledge_base[i] for i in top_indices]

    def build_rag_prompt(self, noisy_findings: str, top_k: int = 3) -> str:
        """
        Build an augmented input string with retrieved examples prepended.

        Args:
            noisy_findings: The (possibly noisy) findings text
            top_k: Number of retrieved examples to include

        Returns:
            Augmented prompt string
        """
        retrieved = self.retrieve(noisy_findings, top_k=top_k)
        context_parts = []
        for i, r in enumerate(retrieved):
            context_parts.append(
                f"Example {i+1}:\nFindings: {r['findings']}\nImpression: {r['impression']}"
            )
        context = "\n\n".join(context_parts)
        augmented = f"{context}\n\nNow summarize:\nFindings: {noisy_findings}\nImpression:"
        return augmented


def augment_records_with_rag(rag: SimpleRAG, records: list, top_k: int = 3) -> list:
    """
    Augment a list of records with RAG context in their findings field.

    Args:
        rag: Initialized SimpleRAG with built index
        records: List of dicts with 'findings'
        top_k: Retrieved examples per record

    Returns:
        New list with augmented 'findings'
    """
    augmented = []
    for rec in records:
        rag_prompt = rag.build_rag_prompt(rec["findings"], top_k=top_k)
        new_rec = dict(rec)
        new_rec["findings"] = rag_prompt
        augmented.append(new_rec)
    return augmented

# 📄 Enterprise RAG Document Assistant

An end-to-end, privacy-focused Retrieval-Augmented Generation (RAG) system engineered to parse unstructured PDF documentation, build dense semantic vector indexes, and answer user queries with strictly grounded document evidence.

---

## 🎯 Problem Statement & Solution

* **The Problem:** Organizations manage extensive libraries of SOPs, compliance handbooks, and operational guides. Traditional keyword search misses semantic context, while standard cloud LLMs pose data privacy concerns and risk hallucinating answers.
* **The Solution:** A local vector search pipeline that segments multi-page PDFs, computes dense vector representations, stores embeddings in a persistent ChromaDB instance, and retrieves top-matching text excerpts ranked by Euclidean distance.

---

## 🏗️ System Architecture

```text
[PDF Upload]
     │
     ▼
[PyPDF Text Extraction]
     │
     ▼
[Recursive Character Text Splitting] ──► Chunk size: 600 | Overlap: 80
     │
     ▼
[Sentence Transformers (all-MiniLM-L6-v2)] ──► Dense Vector Embeddings
     │
     ▼
[ChromaDB Vector Store] ──► Local Euclidean L2 Distance Index
     │
     ▼
[Streamlit Interface] ──► Top-K Semantic Retrieval & Source Citations

![Enterprise RAG Assistant Dashboard](assets/dashboard.png)
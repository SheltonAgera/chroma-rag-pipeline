Contextual RAG Pipeline

Overview:
This project is a simple end-to-end Retrieval-Augmented Generation (RAG) pipeline built using Python and Chroma. The system extracts financial text, processes it into semantic chunks, stores embeddings in a vector database, and retrieves the most relevant context for user queries. The focus of this project is on clean data pipelines, reliable retrieval, and explainable system design, not on black-box abstractions.

Tech Stack:
Python, Chroma (Vector Database), Sentence-Transformers (Embeddings)

Why Chroma?
Chroma was chosen because it is lightweight, runs locally, and makes vector storage and retrieval easy to inspect and debug during development.

Evaluation:

A simple Precision@K metric is used to evaluate retrieval quality. This helps identify weak retrieval results and improve chunking or embedding strategies.

def precision_at_k(retrieved_docs, relevant_keywords, k):
    """
    retrieved_docs: list of retrieved text chunks
    relevant_keywords: list of keywords that indicate relevance
    k: number of retrieved documents
    """
    retrieved_docs = retrieved_docs[:k]
    relevant_count = 0

    for doc in retrieved_docs:
        doc_lower = doc.lower()
        if any(keyword.lower() in doc_lower for keyword in relevant_keywords):
            relevant_count += 1

    return relevant_count / k if k > 0 else 0.0

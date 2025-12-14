def semantic_search(collection, query_embedding, top_k=3):
    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

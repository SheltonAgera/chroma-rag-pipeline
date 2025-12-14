import logging

from src.load_data import load_txt
from src.chunking import chunk_text
from src.embeddings import embed_text, get_model
from src.vector_store import get_collection
from src.query import semantic_search
from src.evaluation import precision_at_k

# ---------------- CONFIG ----------------
DOC_PATH = "data/docs.txt"
TOP_K = 3
RELEVANT_KEYWORDS = [
    "portfolio",
    "risk",
    "diversification",
    "volatility",
    "investment",
    "returns"
]
# ----------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting RAG pipeline")

    # -------- Load --------
    logger.info("Loading document...")
    text = load_txt(DOC_PATH)

    if not text.strip():
        raise ValueError("Document is empty or unreadable")

    # -------- Chunk --------
    logger.info("Chunking text...")
    chunks = chunk_text(text)
    logger.info(f"Total chunks created: {len(chunks)}")

    # -------- Embed --------
    logger.info("Generating embeddings...")
    embeddings = embed_text(chunks)

    # -------- Vector Store --------
    logger.info("Initializing vector database...")
    collection = get_collection()

    logger.info("Storing vectors in Chroma...")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))]
    )

    logger.info("RAG system ready")

    model = get_model()

    # -------- Query Loop --------
    while True:
        query = input("\nAsk a finance question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        query_embedding = model.encode([query])

        results = semantic_search(
            collection,
            query_embedding,
            top_k=TOP_K
        )

        retrieved_docs = results["documents"][0]
        retrieved_dists = results["distances"][0]

        print("\n--- Retrieved Context ---")
        for doc, dist in zip(retrieved_docs, retrieved_dists):
            print(f"\nScore: {dist:.4f}\n{doc}")

        # -------- Evaluation --------
        precision = precision_at_k(
            retrieved_docs,
            RELEVANT_KEYWORDS,
            k=TOP_K
        )

        print(f"\nPrecision@{TOP_K}: {precision:.2f}")


if __name__ == "__main__":
    main()

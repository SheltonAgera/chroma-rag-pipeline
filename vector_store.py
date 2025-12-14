import chromadb

def get_collection(name="rag_collection"):
    client = chromadb.Client()
    return client.get_or_create_collection(name=name)

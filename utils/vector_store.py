import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

def initialize_vector_store(db_path, collection_name, embed_model):
    db_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = db_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import streamlit_authenticator as stauth
from pymongo import MongoClient
from pinecone_text.sparse import BM25Encoder

API_KEY = st.secrets["API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# MONGODB_KEY = st.secrets["MONGODB_KEY"]

# client = MongoClient(MONGODB_KEY)
# db = client["my_database"]

# collection = db["my_collection"]
# doc = collection.find_one()
# print(doc)

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "hybrid-index"
hybrid_index = pc.Index(index_name)
bm25 = BM25Encoder()
bm25.load(f"bm25_params.json")


def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    
    return [v * alpha for v in dense], hs

def encode_search_rerank(user_query, top_k=20, top_n=40, alpha=0.75):
    # Embed the query using OpenAI

    embedding_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=user_query
    )

    sparse_query_embedding = bm25.encode_documents(user_query)

    hdense, hsparse = hybrid_score_norm(embedding_response.data[0].embedding, sparse_query_embedding, alpha)


    query_response = hybrid_index.query(
        namespace="proposal-embeddings",
        top_k=top_k,
        vector=hdense,
        sparse_vector=hsparse,
        include_values=True,
        include_metadata=True
    )
    print(query_response.matches)

    documents_to_rerank = [
        {
            "id": match.id,  
            "text": match.metadata["chunk"],
            "metadata": match.metadata
        }
        for match in query_response.matches
        if "chunk" in match.metadata
    ]

    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=user_query,
        documents=documents_to_rerank,
        rank_fields=["text"],
        top_n=top_n,
        return_documents=True,
        parameters={"truncate": "END"}
    )

    return result
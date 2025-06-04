import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import streamlit_authenticator as stauth
from pymongo import MongoClient

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
dense_index_name = "test-many-prop"
dense_index = pc.Index(dense_index_name)

def encode_search_rerank(user_query, top_k=20):
    # Embed the query using OpenAI

    embedding_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=user_query
    )

    query_vector = embedding_response.data[0].embedding

    # Search Pinecone
    search_results = dense_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace='proposal-embeddings'
    )
    print(search_results.matches)

    documents_to_rerank = [
        {
            "id": match.id,  
            "text": match.metadata["chunk"],
            "metadata": match.metadata
        }
        for match in search_results.matches
        if "chunk" in match.metadata
    ]
    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=user_query,
        documents=documents_to_rerank,
        rank_fields=["text"],
        top_n=20,
        return_documents=True,
        parameters={"truncate": "END"}
    )

    return result
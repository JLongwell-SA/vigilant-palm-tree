import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit_authenticator as stauth
from pymongo import MongoClient
from pinecone_text.sparse import BM25Encoder
import os
import re
from docx import Document
import tiktoken
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
import json
from tqdm import tqdm
import time
from utils.prompts import SUM_PROMPT

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
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "hybrid-index"
hybrid_index = pc.Index(index_name)
bm25 = BM25Encoder()
bm25.load(f"bm25_params.json")

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=3072,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

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

# improve the pipeline with a re-ranker cutoff value/token limit to the amount of chunks we use for context in the final decoder
def encode_search_rerank(user_query, summary, name_space, top_k=20, top_n=40, alpha=0.75):
    # Embed the query using OpenAI
    # write some retry logic here incase we get rate limits

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

    # print(query_response.matches)

    documents_to_rerank = [
        {
            "id": match.id,  
            "text": match.metadata["chunk"],
            "metadata": match.metadata
        }
        for match in query_response.matches
        if "chunk" in match.metadata
    ]

    # setup microsoft azure account and change this to query cohere re-rank api
    # write some retry logic here incase we get rate limits
    result = pc.inference.rerank(
        model="bge-reranker-v2-m3", # hopefully change this to cohere re-rank, currently at 1024 input token max
        query=user_query,
        documents=documents_to_rerank,
        rank_fields=["text"],
        top_n=top_n,
        return_documents=True,
        parameters={"truncate": "END"}
    )


    if (summary != "") and (name_space != ""):

        rfp_response = hybrid_index.query(
            namespace=name_space,
            top_k=top_k,
            vector=hdense,
            sparse_vector=hsparse,
            include_values=True,
            include_metadata=True
        )

        rfp_chunks_to_rerank = [
                {
                    "id": match.id,  
                    "text": match.metadata["chunk"],
                    "metadata": match.metadata
                }
                for match in rfp_response.matches
                if "chunk" in match.metadata

            ]
        
        # setup microsoft azure account and change this to query cohere re-rank api
        # write some retry logic here incase we get rate limits
        rfp_result = pc.inference.rerank(
            model="bge-reranker-v2-m3", # hopefully change this to cohere re-rank, currently at 1024 input token max
            query=user_query,
            documents=rfp_chunks_to_rerank,
            rank_fields=["text"],
            top_n=top_n,
            return_documents=True,
            parameters={"truncate": "END"}
        )

        return [result, rfp_result]
    
    return [result]

def summarize_rfp():
    return ""

def count_tokens(text, model="text-embedding-3-large"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def new_line_regex(text):
    return re.sub(r'(?:\s*\n){4,}', '\n', text)

def split_chunk_by_subheadings(chunk_text, subheadings, parent_title=""):
    split_chunks = []
    current_title = None
    current_lines = []

    for para_text, style in subheadings:
        if style in ['Heading 3', 'Heading 4', 'Heading 5', 'Heading 6', 'Heading 7', 'Heading 8', 'Heading 9']:
            if current_title and current_lines:
                combined_title = f"{parent_title}\n{current_title}" if parent_title else current_title
                split_chunks.append(f"{combined_title}\n" + "\n".join(current_lines).strip())
            current_title = para_text.strip()
            current_lines = []
        else:
            if current_title:
                current_lines.append(para_text)

    if current_title and current_lines:
        combined_title = f"{parent_title}\n{current_title}" if parent_title else current_title
        split_chunks.append(f"{combined_title}\n" + "\n".join(current_lines).strip())

    return split_chunks if split_chunks else [chunk_text]

def extract_section_chunks(doc, doc_title):

    section_chunks = []
    all_section_paras = []
    current_section_title = None
    current_section_lines = []
    current_section_paras = []
    intro_lines = []
    intro_paras = []
    found_first_heading = False
    intro_chunk = None

    def table_to_text(table):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows)


    def iter_block_items(parent):
        for child in parent.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            style_name = block.style.name
            text = block.text.strip()

            if style_name in ['Heading 1', 'Heading 2']:
                if not found_first_heading and intro_lines:
                    intro_title = f"Intro [{doc_title}]"
                    full_intro = f"{intro_title}\n" + "\n".join(intro_lines).strip()
                    full_intro = new_line_regex(full_intro)
                    intro_chunk = full_intro

                found_first_heading = True

                if current_section_title and current_section_lines:
                    full_section = f"{current_section_title}\n" + "\n".join(current_section_lines).strip()
                    full_section = new_line_regex(full_section)
                    section_chunks.append(full_section)
                    all_section_paras.append(current_section_paras)

                current_section_title = text
                current_section_lines = []
                current_section_paras = []
            else:
                if not found_first_heading:
                    intro_lines.append(text)
                    intro_paras.append((text, style_name))
                elif current_section_title:
                    current_section_lines.append(text)
                    current_section_paras.append((text, style_name))

        elif isinstance(block, Table):
            table_text = table_to_text(block)
            if not found_first_heading:
                intro_lines.append(table_text)
                intro_paras.append((table_text, 'Table'))
            elif current_section_title:
                current_section_lines.append(table_text)
                current_section_paras.append((table_text, 'Table'))

    if current_section_title and current_section_lines:
        full_section = f"{current_section_title}\n" + "\n".join(current_section_lines).strip()
        full_section = new_line_regex(full_section)
        section_chunks.append(full_section)
        all_section_paras.append(current_section_paras)

    if intro_chunk:
        if section_chunks and count_tokens(intro_chunk) + count_tokens(section_chunks[0]) < 4096:
            section_chunks[0] = intro_chunk + "\n\n" + section_chunks[0]
        else:
            section_chunks.insert(0, intro_chunk)
            all_section_paras.insert(0, intro_paras)

    print("Num of chunks before splitting big chunks " + str(len(section_chunks)))

    # Reprocess large chunks by Heading 3–6 if token count > 4096
    new_chunks = []
    for chunk_text, chunk_paras in zip(section_chunks, all_section_paras):
        if count_tokens(chunk_text) > 4096:
            print(f"Splitting large chunk ({count_tokens(chunk_text)} tokens)")
            parent_title = chunk_text.split("\n", 1)[0].strip()
            split = split_chunk_by_subheadings(chunk_text, chunk_paras, parent_title=parent_title)
            new_chunks.extend(split)
        else:
            new_chunks.append(chunk_text)

    section_chunks = new_chunks

    print("Num of chunks after splitting big chunks " + str(len(section_chunks)))

    # Merge small chunks (<= 256 tokens) with prev/next if combined <= 8000
    if section_chunks:
        unchanged_iterations = 0
        while unchanged_iterations < 2:
            count_of_section_chunks = len(section_chunks)
            i = 1
            while i < len(section_chunks) - 1:
                current_chunk = section_chunks[i]
                current_tokens = count_tokens(current_chunk)

                if current_tokens <= 256:
                    merged = False
                    prev_tokens = count_tokens(section_chunks[i - 1])
                    if prev_tokens + current_tokens <= 8192:
                        section_chunks[i - 1] += "\n\n" + current_chunk
                        merged = True

                    next_tokens = count_tokens(section_chunks[i + 1])
                    if next_tokens + current_tokens <= 8192:
                        section_chunks[i + 1] = current_chunk + "\n\n" + section_chunks[i + 1]
                        merged = True

                    if merged:
                        del section_chunks[i]
                        continue

                i += 1

            if len(section_chunks) == count_of_section_chunks:
                unchanged_iterations += 1
            else:
                unchanged_iterations = 0

    print("Num of chunks after merging chunks " + str(len(section_chunks)))

    # Final pass: Split any chunks over 8000 tokens in half with overlap, repeat until all are under 8000
    overlap_lines = 5  # Number of lines to include in both halves as overlap

    while any(count_tokens(chunk) > 8192 for chunk in section_chunks):
        updated_chunks = []
        for chunk in section_chunks:
            if count_tokens(chunk) > 8192:
                lines = chunk.split("\n")
                midpoint = len(lines) // 2

                # Calculate bounds for overlap
                start_overlap = max(0, midpoint - overlap_lines)
                end_overlap = min(len(lines), midpoint + overlap_lines)

                part1 = "\n".join(lines[:end_overlap]).strip()
                part2 = "\n".join(lines[start_overlap:]).strip()

                updated_chunks.append(part1)
                updated_chunks.append(part2)
            else:
                updated_chunks.append(chunk)
        section_chunks = updated_chunks

    section_chunks = [new_line_regex(chunk) for chunk in section_chunks]

    # remove any duplicates that may exist in chunks
    def deduplicate_within_chunk(chunk):
        seen = set()
        deduped_lines = []
        for line in chunk.split('\n'):
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                deduped_lines.append(line_stripped)
        return '\n'.join(deduped_lines)

    # Apply to all chunks
    section_chunks = [deduplicate_within_chunk(chunk) for chunk in section_chunks]

    return section_chunks

def scrape_rfp(uploaded_file):
    doc = Document(uploaded_file)
    doc_title = os.path.splitext(uploaded_file.name)[0]

    chunks = extract_section_chunks(doc, doc_title)
    # # Print for testing
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"\n--- Section {i} ---\n{chunk}\n")

    # list_of_dicts = []
    # token_lengths = [count_tokens(chunk) for chunk in chunks]

    # if token_lengths:
    #     max_index = token_lengths.index(max(token_lengths))
    #     largest_chunk = chunks[max_index]
    #     largest_tokens = token_lengths[max_index]
    # else:
    #     largest_chunk = ""
    #     largest_tokens = 0

    # list_of_dicts.append({
    #     "filename": doc_title,
    #     "chunks": chunks,
    #     "token_lengths": token_lengths,
    #     "largest": {
    #         "tokens": largest_tokens,
    #         "text": largest_chunk
    #     }
    # })

    #save this for logging purposes to mongo once we get it
    #list_of_dicts


    ###
    ### write some retry logic here to avoid rate limit failures
    ###
    response = client.embeddings.create(
            input=chunks,
            model="text-embedding-3-large"
        )
    
    doc_sparse_vectors = bm25.encode_documents(chunks)

    embeddings = [
                {
                    "id": doc_title +"_"+str(j),
                    "values": e.embedding,
                    "sparse_values": {
                    "indices": doc_sparse_vectors[j]["indices"],
                    "values": doc_sparse_vectors[j]["values"]
                        },
                    "metadata": {"filename": doc_title, "chunk": chunks[j]}
                    
                }
                for j, e in enumerate(response.data)
            ]
    # index.delete(delete_all=True, namespace=doc_title+"-embeddings")

    # while True:
    #     stats = index.describe_index_stats()
    #     ns_stats = stats['namespaces'].get(doc_title+"-embeddings", {})
    #     vector_count = ns_stats.get('vector_count', 0)

    #     if vector_count == 0:
    #         print("Deletion complete.")
    #         break

    #     print(f"Waiting for deletion... {vector_count} vectors remain.")
    #     time.sleep(2)

    index.upsert(vectors=embeddings, namespace=doc_title+"-embeddings")

    return doc_title+"-embeddings"

def extract_paragraphs_and_tables(doc):
    elements = []

    for block in doc.element.body:
        if block.tag.endswith('}p'):
            paragraph = Paragraph(block, doc)
            text = paragraph.text.strip()
            if text:
                elements.append(text)
        elif block.tag.endswith('}tbl'):
            table = Table(block, doc)
            table_data = []

            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)

            elements.append(table_data)

    return elements

def format_table_as_markdown(table_data):
    if not table_data:
        return ""

    header = table_data[0]
    rows = table_data[1:]

    # Build the header line
    md = "| " + " | ".join(header) + " |\n"
    # Build the separator line
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    # Build the data rows
    for row in rows:
        md += "| " + " | ".join(row) + " |\n"

    return md

def summarize_rfp(uploaded_file):
    doc = Document(uploaded_file)
    
    # Extract content
    elements = extract_paragraphs_and_tables(doc)

    # Convert everything to Markdown-formatted text
    final_text = ""
    for el in elements:
        if isinstance(el, str):
            final_text += el + "\n\n"
        else:
            final_text += format_table_as_markdown(el) + "\n\n"


    # make an API call where we pass in the doc + a query requestion for a summary
    response = client.responses.create(
        model ="gpt-4.1-2025-04-14",
        input = [

            {
                "role": "system",
                "content": SUM_PROMPT
            },
            {
                "role": "user",
                "content": "Here is the RFP document " + final_text
            } 
        ],
        temperature=1,
        top_p=1
    )

    return response.output_text

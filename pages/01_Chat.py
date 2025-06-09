import streamlit as st
from utils.utils import encode_search_rerank, scrape_rfp, client, summarize_rfp
from utils.prompts import NO_DOC_PROMPT, DOC_PROMPT
# Streamlit Page Config
st.set_page_config(page_title="Proposal Chat", layout="wide")


st.title("📄💬 Proposal Chatbot")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "namespace" not in st.session_state:
    st.session_state.namespace = ""
if "doc_title" not in st.session_state:
    st.session_state.doc_title = ""

st.write("Please upload the RFP you need assistance with. Be sure to first convert the RFP from PDF to Word using Bluebeam.")
with st.expander("PDF to Word conversion instructions"):
        st.write("Step 1: Open your RFP using Bluebeam")
        st.write("Step 2: Click file > Export > Word Document > Entire Document")
        st.write("Step 3: Drag and drop the exported Word doc below")
uploaded_file = st.file_uploader(" 📄 Word Document Uploader (.docx)", type="docx")

if (uploaded_file is not None) and (st.session_state.summary == "") :
    with st.spinner("Procesing..."):
    # calls function from utils that scrapes RFP -> chunks it -> stores it in its own namespace, returning name of the uploaded doc to use as namespace.
        st.session_state.doc_title = scrape_rfp(uploaded_file)
        print(st.session_state.doc_title)
        st.session_state.namespace = st.session_state.doc_title + "-embeddings"
    with st.spinner("Summarizing..."):
        st.session_state.summary = summarize_rfp(uploaded_file)
        print(st.session_state.summary)

with st.sidebar:
    # New Chat button
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.namespace = ""
        st.session_state.doc_title = ""
        st.rerun()

# Display chat messages
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display context for assistant messages
        if msg["role"] == "assistant" and "context" in msg:
            # Use a unique key for each expander to prevent interference, havent used this yet
            expander_key = f"context_expander_{msg_idx}"
            
            with st.expander("📄 View Retrieved Context Chunks", expanded=False):
                
                context_container = st.container()
                
                with context_container:
                    for i, match in enumerate(msg["context"]):
                        doc = match['document']
                        doc_id = doc['id']
                        metadata = doc.get('metadata', {})
                        
                        chunk_no = doc_id.split("_")[-1]
                        og_document_title = "_".join(doc_id.split("_")[:-1])
                        chunk_text = metadata.get("chunk", "[No chunk text available]")
                        
                        # Use columns to better organize the content and reduce height
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.markdown(f"**📘 Result {i+1}**")
                            st.markdown(f"Score: `{match['score']:.4f}`")
                            st.markdown(f"Doc: `{og_document_title}`")
                            st.markdown(f"Chunk: `{chunk_no}`")
                        
                        with col2:
                            
                            with st.container():
                                st.markdown("**Content:**")
                                st.markdown(chunk_text)
                        
                        # Add a subtle divider
                        if i < len(msg["context"]) - 1:
                            st.divider()

# Chat input
user_input = st.chat_input("Ask something about your engineering proposals...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Searching relevant proposal content..."):

        result = encode_search_rerank(user_input, st.session_state.summary, st.session_state.namespace, top_k=5, top_n=5, alpha=0.75)
        print("This is the len of the result", len(result))
        st.session_state.len_res = len(result)
        if not result[0].data:
            full_response = "🤔 I couldn't find anything relevant."
            # Save to chat history without context
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
        else:
            final_prompt = f""
            
            
            proposals_prompt = f""
            for i, match in enumerate(result[0].data):
                proposals_prompt += f"### Result {i+1} 📘 \n"
                proposals_prompt += f"**Score:** `{match['score']:.4f}`\n"
                doc = match['document']
                doc_id = doc['id']
                metadata = doc.get('metadata', {})

                chunk_no = doc_id.split("_")[-1]
                og_document_title = "_".join(doc_id.split("_")[:-1])
                chunk_text = metadata.get("chunk", "[No chunk text available]")

                proposals_prompt += f"**Original Document Title:** `{og_document_title}`\n"
                proposals_prompt += f"**Chunk Number:** `{chunk_no}`\n"
                proposals_prompt += f"**Content:** {chunk_text}\n"
                proposals_prompt += f"-------------------------------------\n"
            
    
            if st.session_state.len_res == 1:
                final_prompt = NO_DOC_PROMPT + proposals_prompt + "## </CONTEXT>\n\n## <User Query>\n"

            else:
                rfp_prompt = f"**Original Document Title:** `{st.session_state.doc_title}`\n\n"
                for i, match in enumerate(result[1].data):
                    rfp_prompt += f"### Result {i+1} 📘 \n"
                    rfp_prompt += f"**Score:** `{match['score']:.4f}`\n"
                    metadata = doc.get('metadata', {})

                    chunk_no = doc_id.split("_")[-1]
                    og_document_title = "_".join(doc_id.split("_")[:-1])
                    chunk_text = metadata.get("chunk", "[No chunk text available]")

                    rfp_prompt += f"**Chunk Number:** `{chunk_no}`\n"
                    rfp_prompt += f"**Content:** {chunk_text}\n"
                    rfp_prompt += f"-------------------------------------\n"

                final_prompt = DOC_PROMPT + "## <SUMMARY>\n" + st.session_state.summary + "\n## </SUMMARY>\n\n## <RFP>\n" + rfp_prompt + "\n## </RFP>\n\n## <PROPOSAL>\n" + proposals_prompt +  "\n## </PROPOSAL>\n\n## </CONTEXT>\n\n## <User Query>\n" 


            print("This is the final prompt", final_prompt)

            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {"role": "system",
                     "content": final_prompt
                    },
                    {"role": "user",
                     "content": user_input + "\n</User Query>\n"
                    }    
                ],
                temperature=1,
                max_tokens=8192,
                top_p=1,
                stream=True
            )

            # Stream assistant reply
            full_response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    placeholder.markdown(f"{full_response}▌")
                placeholder.markdown(full_response)

                ## Look for $ post process and make them good
                
                # Change these to redirect to another page? or wipe whats there or something
                # Add collapsible context display for current response. 
                # make another of these for the rfp retreived context
                # with st.expander("📄 View Retrieved Context Chunks", expanded=False):
                #     context_container = st.container()
                    
                #     with context_container:
                #         for i, match in enumerate(result[0].data):
                #             doc = match['document']
                #             doc_id = doc['id']
                #             metadata = doc.get('metadata', {})

                #             chunk_no = doc_id.split("_")[-1]
                #             og_document_title = "_".join(doc_id.split("_")[:-1])
                #             chunk_text = metadata.get("chunk", "[No chunk text available]")

                #             # Use columns for better layout
                #             col1, col2 = st.columns([1, 3])
                            
                #             with col1:
                #                 st.markdown(f"**📘 Result {i+1}**")
                #                 st.markdown(f"Score: `{match['score']:.4f}`")
                #                 st.markdown(f"Doc: `{og_document_title}`")
                #                 st.markdown(f"Chunk: `{chunk_no}`")
                            
                #             with col2:
                #                 st.markdown("**Content:**")
                #                 st.markdown(chunk_text)
                            
                #             if i < len(result[0].data) - 1:
                #                 st.divider()

            # Save to chat history WITH context data
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response.strip(),
                "context": result[0].data  # Store the context data
            })
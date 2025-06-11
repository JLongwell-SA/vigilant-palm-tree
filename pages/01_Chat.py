import streamlit as st
from utils.utils import encode_search_rerank, process_rfp_vectors, client, process_rfp_text, trim_history
from utils.prompts import NO_DOC_PROMPT, DOC_PROMPT, SUM_PROMPT
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
if "rolling_history" not in st.session_state:
    st.session_state.rolling_history = []

with st.sidebar:
    # New Chat button
    if st.button("🆕 New Chat"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.write("Please upload the RFP you need assistance with. Be sure to first convert the RFP from PDF to Word using Bluebeam.")
with st.expander("PDF to Word conversion instructions"):
        st.write("Step 1: Open your RFP using Bluebeam")
        st.write("Step 2: Click file > Export > Word Document > Entire Document")
        st.write("Step 3: Drag and drop the exported Word doc below")

uploaded_file = st.file_uploader(" 📄 Word Document Uploader (.docx)", type="docx", key = "doc_uploader")

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
        # print("This is the len of the result", len(result))
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

            #Update the rolling history in the session state, then pass it to the api
            st.session_state.rolling_history.extend([{"role": "system", "content":f"{final_prompt}"},{"role": "user", "content": user_input + "\n</User Query>\n"}])
            print("This is the rolling history:", st.session_state.rolling_history)
            response = client.responses.create(
                model="gpt-4.1-2025-04-14",
                input = st.session_state.rolling_history,
                temperature=1,
                top_p=1,
                stream=True
            )

            # Stream assistant reply
            full_response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in response:
                    if hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
                        if hasattr(chunk, 'delta') and chunk.delta:
                            delta = chunk.delta
                            escaped_delta = delta.replace('$', '\\$')
                            full_response += escaped_delta
                            placeholder.markdown(f"{full_response}▌")
                placeholder.markdown(full_response)

                # Change these to redirect to another page? or wipe whats there or something
                # Add collapsible context display for current response. 
                # make another of these for the rfp retreived context
                with st.expander("📄 View Retrieved Context Chunks", expanded=False):
                    context_container = st.container()
                    
                    with context_container:
                        for i, match in enumerate(result[0].data):
                            doc = match['document']
                            doc_id = doc['id']
                            metadata = doc.get('metadata', {})

                            chunk_no = doc_id.split("_")[-1]
                            og_document_title = "_".join(doc_id.split("_")[:-1])
                            chunk_text = metadata.get("chunk", "[No chunk text available]")

                            # Use columns for better layout
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.markdown(f"**📘 Result {i+1}**")
                                st.markdown(f"Score: `{match['score']:.4f}`")
                                st.markdown(f"Doc: `{og_document_title}`")
                                st.markdown(f"Chunk: `{chunk_no}`")
                            
                            with col2:
                                st.markdown("**Content:**")
                                st.markdown(chunk_text)
                            
                            if i < len(result[0].data) - 1:
                                st.divider()

            st.session_state.rolling_history.append({"role": "assistant", "content": full_response})
            st.session_state.rolling_history = trim_history(st.session_state.rolling_history)

            # Save to chat history WITH context data
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response.strip(),
                "context": result[0].data  # Store the context data
            })



if (uploaded_file is not None) and (st.session_state.summary == ""):
    with st.spinner("Procesing..."):
    # calls function from utils that scrapes RFP -> chunks it -> stores it in its own namespace, returning name of the uploaded doc to use as namespace.
        st.session_state.namespace = process_rfp_vectors(uploaded_file)
        # print(st.session_state.namespace)
        st.session_state.doc_title = st.session_state.namespace.split("-embeddings")[0]
    with st.spinner("Summarizing..."):
        full_text = process_rfp_text(uploaded_file)
        # print(st.session_state.summary)
        summary_response = client.responses.create(
            model ="gpt-4.1-2025-04-14",
            input = [
                {"role": "system", "content":f"{SUM_PROMPT}"},
                {"role": "user", "content": "Here is the RFP document " + full_text}
            ],
            temperature=1,
            top_p=1,
            stream=True
        )
        # Stream assistant reply
        full_summary = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in summary_response:
                if hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
                    if hasattr(chunk, 'delta') and chunk.delta:
                        delta = chunk.delta
                        escaped_delta = delta.replace('$', '\\$')
                        full_summary += escaped_delta
                        placeholder.markdown(f"{full_summary}▌")    
            placeholder.markdown(full_summary)

        st.session_state.summary = full_summary
        st.session_state.messages.append({
                "role": "assistant",
                "content": full_summary.strip(),
                "key": "summary"
            })
        print("This is the length of messages",len(st.session_state.messages))

with st.sidebar:
    #check if st.session_state.summary is not empty string
    if st.session_state.summary != "":
        
        if st.button("✏️ Edit Summary"):
            st.session_state.show_edit_popup = True
            # Show the text area in a simulated popup (expander)
            print(st.session_state.get("show_edit_popup", False))
        
        if st.session_state.get("show_edit_popup", False):
            # with st.expander("📝 Edit Model Summary", expanded=True):
            edited_summary = st.text_area("Edit below:", value=st.session_state.summary, height=300, key="summary_edit")
            if st.button("✅ Save Changes"):
                st.session_state.summary = edited_summary
                for i, message in enumerate(st.session_state.messages):
                    if message.get("key") == "summary":
                        st.session_state.messages[i]["content"] = edited_summary.strip()
                        break

                #edit the st.session_state.messages that has "key": "summary"
                st.success("Summary updated!")
                st.session_state.show_edit_popup = False
                st.rerun()

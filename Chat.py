import streamlit as st
from utils.utils import encode_search_rerank, client

# Streamlit Page Config
st.set_page_config(page_title="Proposal Chat", layout="wide")
st.page_link("pages/01_Search.py", label="🔍Go to Proposal Search", icon="🔎")

st.title("📄💬 Proposal Chatbot")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# Chat input
user_input = st.chat_input("Ask something about your engineering proposals...")


if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Searching relevant proposal content..."):
            result = encode_search_rerank(user_input, top_k=5)

            if not result.data:
                full_response = "🤔 I couldn't find anything relevant."
            else:
                prompt = '''You are an advanced AI assistant specializing in retrieving and synthesizing information from historical engineering proposals. Your primary function is to act as a RAG (Retrieval-Augmented Generation) question-answering bot, specifically operating over a corpus of past engineering proposals that were submitted by Smith + Andersen Engineering Consultants in response to Requests for Proposals (RFPs) from various companys and entities.

### Core Mission:
Accurately and comprehensively answer user questions by first retrieving relevant sections from the provided historical engineering proposals, and then synthesizing that information into clear, concise, and direct answers.

### Constraints & Guidelines:

## Strictly adhere to the provided context:
Your answers must be grounded solely in the information found within the retrieved engineering proposal documents. Do not introduce external knowledge, make assumptions, or extrapolate beyond the text.
## RAG Process Emulation:
When a user asks a question, mentally simulate a retrieval step. If you were truly a RAG system, you would query a vector database or similar index to find the most semantically relevant chunks of text from the proposal corpus.
For this simulation, assume that the most relevant paragraphs/sections from the proposals will be provided to you as [CONTEXT] in each user query. Your task begins after this retrieval step.
Your output should clearly demonstrate that you are drawing directly from this context.

## Answer Format:
# Direct Answer First:
Provide the most direct and concise answer to the user's question upfront.
# Elaboration (if necessary):
If the direct answer requires further explanation or supporting details, provide them in subsequent sentences or a short paragraph.
# Quoting (selectively):
Use direct quotes sparingly and only when the exact phrasing is crucial for accuracy or clarity. When quoting, enclose the quoted text in quotation marks.
# Avoid conversational filler:
Do not use phrases like "As an AI language model...", "I can help you with that...", or other non-informative greetings/closings. Get straight to the point.
# Handling Ambiguity/Lack of Information:
If the answer to the question cannot be found within the provided [CONTEXT], state explicitly and politely: "I apologize, but the provided context does not contain the information needed to answer this question." Do not guess or fabricate information.
If a question is vague or requires clarification, respond by stating what information is missing (e.g., "Could you please specify which project or client you are referring to?").
# Focus on Engineering & Proposal Specifics:
Pay close attention to technical specifications, proposed methodologies, project timelines, deliverables, team structures, cost breakdowns (if available in context), unique selling points, and responses to specific RFP requirements.
Understand that these documents are often formal, detailed, and persuasive in nature.
#Maintain Professional Tone:
Your responses should be professional, objective, and authoritative.
# Identify Key Entities:
Be able to identify and differentiate between Smith + Andersen (the proposer) and the "Client Company" (the requesting entity/customer) within the context.
# Iterative Refinement (Implicit):
If a user asks a follow-up question that relies on the same context, assume the context is still relevant or that a refined context would be provided.

Use the following retrieved information only to answer the user's question:
## <CONTEXT>

'''
                for i, match in enumerate(result.data):
                    prompt += f"### Result {i+1} 📘 \n"
                    prompt += f"**Score:** `{match['score']:.4f}`\n"
                    doc = match['document']
                    doc_id = doc['id']
                    metadata = doc.get('metadata', {})

                    chunk_no = doc_id.split("_")[-1]
                    og_document_title = "_".join(doc_id.split("_")[:-1])
                    chunk_text = metadata.get("chunk", "[No chunk text available]")

                    prompt += f"**Original Document Title:** `{og_document_title}`\n"
                    prompt += f"**Chunk Number:** `{chunk_no}`\n"
                    prompt += f"**Content:** {chunk_text}\n"
                    prompt += f"-------------------------------------\n"
                
                prompt += f"## </CONTEXT>\n\n## <User Query>\n"

                
                response = client.chat.completions.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system",
                         "content": prompt
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

                

                # Add collapsible context display
                with st.expander("📄 View Retrieved Context Chunks"):
                    for i, match in enumerate(result.data):
                        doc = match['document']
                        doc_id = doc['id']
                        metadata = doc.get('metadata', {})

                        chunk_no = doc_id.split("_")[-1]
                        og_document_title = "_".join(doc_id.split("_")[:-1])
                        chunk_text = metadata.get("chunk", "[No chunk text available]")

                        st.markdown(f"#### 📘 Result {i+1}")
                        st.markdown(f"- **Score:** `{match['score']:.4f}`")
                        st.markdown(f"- **Document:** `{og_document_title}`")
                        st.markdown(f"- **Chunk #** `{chunk_no}`")
                        st.markdown(f"```markdown\n{chunk_text.strip()}\n```")

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response.strip()
                })


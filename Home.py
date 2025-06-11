#landing and login page

import streamlit as st
APP_PATH = st.secrets["APP_PATH"]

# Set page config
st.set_page_config(page_title="My Homepage", layout="centered")

# Sidebar
# with st.sidebar:
#     st.title("Navigation")

app_path = APP_PATH ## change this in the secrets file before deployment

# Main Page
st.title("🏠 Home")

st.markdown("---")


# Custom CSS for card styling
st.markdown("""
    <style>
        .tool-card {
            border: 1px solid #444;
            border-radius: 16px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: all 0.2s ease-in-out;
        }
        .tool-card:hover {
            box-shadow: 0 6px 14px rgba(0,0,0,0.3);
        }
        .tool-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .tool-description {
            font-size: 16px;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)


# Tool 1: Chat
st.markdown(f"""
    <div class="tool-card">
        <div class="tool-header">💬 Proposal Chat</div>
        <div class="tool-description">
            Chat with our assistant to analyze past engineering proposals or draft new responses to RFPs. This tool leverages advanced retrieval and generation to give you fast, contextual answers.
        </div>
        <br>
        <a href="{app_path}/Chat" target="_self">
            <button style="padding:10px 20px;border:none;border-radius:8px;background-color:#4CAF50;color:white;cursor:pointer;">
                Go to Chat
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)



# Tool 2: Search
st.markdown(f"""
    <div class="tool-card">
        <div class="tool-header">🔍 Proposal Search</div>
        <div class="tool-description">
            Search through a curated archive of previous proposals submitted by the company. This tool helps you locate specific sections, project references, or content relevant to your current work.
        </div>
        <br>
        <a href="{app_path}/Search" target="_self">
            <button style="padding:10px 20px;border:none;border-radius:8px;background-color:#2196F3;color:white;cursor:pointer;">
                Go to Search
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
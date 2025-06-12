#landing and login page

import streamlit as st

# Set page config
st.set_page_config(page_title="My Homepage", layout="centered")


APP_PATH = st.secrets["APP_PATH"] ## change this in the secrets file before deployment

# Make the bar the tells them to use the nav bar
st.markdown("""
    <div style="
        background-color: #2a2d33;
        color: white;
        border-left: 6px solid #00bcd4;
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 8px;
        font-size: 16px;">
        👈 Use the <strong>left navigation sidebar</strong> to switch between tools in the app!
    </div>
""", unsafe_allow_html=True)

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
        <div class="tool-header">💬 Chat</div>
        <div class="tool-description">
            Chat with our assistant to analyze past engineering proposals or draft new responses to RFPs. This tool leverages advanced retrieval and generation to give you fast, contextual answers.
        </div>
        <br>
        <a href="{APP_PATH}/Chat" target="_self">
            <button style="padding:10px 20px;border:none;border-radius:8px;background-color:#fd7e14;color:white;cursor:pointer;">
                Go to Chat
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)


# Tool 2: Search
st.markdown(f"""
    <div class="tool-card">
        <div class="tool-header">🔍 Search</div>
        <div class="tool-description">
            Search through a curated archive of previous proposals submitted by the company. This tool helps you locate specific sections or content relevant to your current work.
        </div>
        <br>
        <a href="{APP_PATH}/Search" target="_self">
            <button style="padding:10px 20px;border:none;border-radius:8px;background-color:#2196F3;color:white;cursor:pointer;">
                Go to Search
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)

# Tool 3: Reference Finder
st.markdown(f"""
    <div class="tool-card">
        <div class="tool-header">📄 Reference Finder</div>
        <div class="tool-description">
            Query a structured database of previous company projects, extracted from MarCom brochures. This tool helps you find high quality project references relevant to your current work. 
        </div>
        <br>
        <a href="{APP_PATH}/Reference_Finder" target="_self">
            <button style="padding:10px 20px;border:none;border-radius:8px;background-color:#dc3545;color:white;cursor:pointer;">
                Go to Reference Finder
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)

# Tool 4: Requirement Extractor
st.markdown(f"""
    <div class="tool-card">
        <div class="tool-header">✅ Requirement Extractor</div>
        <div class="tool-description">
            Extract a list of requirements from a new Request For Proposal (RFP). This tool helps you itemize the nessesary components to putting together a successful response proposal package. 
        </div>
        <br>
        <a href="{APP_PATH}/Requirement_Extractor" target="_self">
            <button style="padding:10px 20px;border:none;border-radius:8px;background-color:#4CAF50;color:white;cursor:pointer;">
                Go to Requirement Extractor
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
import os
import streamlit as st

st.set_page_config(
    page_title="Space Research Assistant",
    page_icon="🚀",
    layout="wide"
)

if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from bot import get_rag_chain

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.28), transparent 25%),
                    radial-gradient(circle at 85% 15%, rgba(56, 189, 248, 0.20), transparent 25%),
                    radial-gradient(circle at 50% 100%, rgba(168, 85, 247, 0.20), transparent 30%),
                    #030712 !important;
    }
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #c084fc, #60a5fa, #22d3ee, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .hero-sub {
        color: #a8b3cf;
        font-size: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 25px;
    }
    .badge {
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #dbeafe;
    }
    </style>
""", unsafe_allowed_html=True)

st.markdown('<div class="hero-title">🚀 SPACE RESEARCH ASSISTANT</div>', unsafe_allowed_html=True)
st.markdown('<div class="hero-sub">Explore the universe • Discover the unknown • Ask anything about space 🌌</div>', unsafe_allowed_html=True)

st.markdown("""
<div class="badge-container">
    <span class="badge">📚 NASA Knowledge Base</span>
    <span class="badge">🌐 Live NASA Updates</span>
    <span class="badge">🧠 RAG Powered</span>
    <span class="badge">🤖 AI Assistant</span>
</div>
""", unsafe_allowed_html=True)

@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("💡 Try Asking")
example_questions = [
    "What is a black hole?",
    "What has the Hubble Space Telescope discovered?",
    "How does the James Webb Space Telescope work?",
    "What are the latest developments in space exploration?",
    "How do rockets work?",
    "What are exoplanets?"
]

selected_example = None
for q in example_questions:
    if st.sidebar.button(q):
        selected_example = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt_input = st.chat_input("What do you want to learn about... ✨")
user_query = selected_example or prompt_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the cosmos... 🌌"):
            try:
                response = chain.invoke(user_query)
            except Exception as e:
                response = f"⚠️ Something went wrong:\n\n{str(e)}"
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
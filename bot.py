import os
import warnings
import numpy as np
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ==========================================
# 1. EMBEDDINGS & ENVIRONMENT
# ==========================================
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

embeddings = download_hugging_face_embeddings()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

# ==========================================
# 2. PINECONE VECTOR STORE SETUP
# ==========================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "space-research"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def load_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents)

def add_pdfs_to_vectorstore(pdf_paths):
    all_docs = []
    for pdf_path in pdf_paths:
        all_docs.extend(load_pdf_file(pdf_path))
    
    chunks = text_split(all_docs)
    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings,
    )

# ==========================================
# 3. LIVE NASA NEWS FETCHER
# ==========================================
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)

def get_space_news(question, max_results=5):
    feed_url = "https://science.nasa.gov/feed/"
    try:
        response = requests.get(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        response.raise_for_status()
    except Exception as e:
        return f"Could not connect to NASA: {e}"

    soup = BeautifulSoup(response.content, "html.parser")
    items = soup.find_all("item")

    if not items:
        return "NASA feed was reached, but no articles were found."

    articles = []
    for item in items:
        title_tag = item.find("title")
        description_tag = item.find("description")
        link_tag = item.find("link")
        date_tag = item.find("pubdate")

        title = title_tag.get_text(" ", strip=True) if title_tag else "Untitled"
        description = clean_text(description_tag.decode_contents()) if description_tag else ""
        link = link_tag.get_text(" ", strip=True) if link_tag else ""
        published = date_tag.get_text(" ", strip=True) if date_tag else "Date unavailable"

        articles.append({
            "title": title,
            "description": description,
            "link": link,
            "published": published
        })

    article_texts = [f"{article['title']} {article['description']}" for article in articles]
    question_embedding = embeddings.embed_query(question)
    article_embeddings = embeddings.embed_documents(article_texts)

    question_vector = np.array(question_embedding)
    scores = []

    for article, article_vector in zip(articles, article_embeddings):
        article_vector = np.array(article_vector)
        similarity = np.dot(question_vector, article_vector) / (
            np.linalg.norm(question_vector) * np.linalg.norm(article_vector)
        )
        scores.append((similarity, article))

    scores.sort(key=lambda x: x[0], reverse=True)
    selected = scores[:max_results]

    news_context = ""
    for similarity, article in selected:
        news_context += f"""
Title: {article['title']}
Published: {article['published']}
Summary: {article['description']}
Source: {article['link']}
Relevance Score: {similarity:.3f}

"""
    return news_context

# ==========================================
# 4. RAG CHAIN PIPELINE & STRICT GROUNDING
# ==========================================
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

system_prompt = """
You are a Space Research Assistant.

Your job is to answer questions about:
- astronomy
- the universe
- stars and galaxies
- planets and moons
- black holes
- space missions
- space telescopes
- satellites
- rockets and propulsion
- space technology
- historical space discoveries
- recent space discoveries and developments

For information from uploaded PDFs, rely ONLY on the provided context.
For recent space news, use the recent-news context when available.

Clearly distinguish established scientific knowledge from recent news.

Do not make up facts, discoveries, dates, or sources.
If the provided context does not contain enough information to answer the question, say:
"I am sorry, but I do not have enough information in my provided documents or recent news to answer reliably."
Do NOT use internal knowledge to answer questions outside the provided context.

Explain things in simple, beginner-friendly language unless the user asks for a technical explanation.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_combined_context(question):
    pdf_docs = retriever.invoke(question)
    pdf_context = format_docs(pdf_docs)
    news_context = get_space_news(question, max_results=5)

    combined_context = f"""
==============================
ESTABLISHED SPACE KNOWLEDGE
FROM UPLOADED PDFS
==============================

{pdf_context}


==============================
RECENT NASA SCIENCE NEWS
==============================

{news_context}
"""
    return combined_context

def get_rag_chain():
    return (
        {
            "context": lambda question: get_combined_context(question),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = get_rag_chain()
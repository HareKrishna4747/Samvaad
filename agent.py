from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
from xml.etree import ElementTree
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
# Set your OpenAI API Key
from dotenv import load_dotenv
import os
import openai
OPENAI_API_KEY = os.environ.get('openai')
openai.api_key = os.environ.get('openai')
load_dotenv()
# Initialize OpenAI embeddings and chat model
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('openai'))
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)
# Function to extract text from PDFs
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Function to create and store FAISS vector database
def get_vector_store(text_chunks):
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(index_path)

# Function to load FAISS and answer questions
def user_input(user_question):
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chat_history_str = "\n".join([f"{role}: {msg}" for role, msg in st.session_state['chat_history']])
    chain = get_conversational_chain()
    response = chain({"chat_history": chat_history_str,"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response

# Function to set up QA chain with OpenAI
def get_conversational_chain():
    prompt_template = """
    You are an assistant with who remebers chat history for question-answering tasks based on multiple documents, 
    teacher information, and the Computer Science timetable at NMIMS University. 
    Use the following pieces of retrieved context to answer questions accurately. 
    If you don't know the answer, say that you don't know. be friendly.
    Provide detailed yet concise responses.sem I and II come under first year, sem III and IV come under second year and sem V and  come under third year.
    chat history:
    {chat_history}
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history","context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# List of PDF file paths
pdf_files = [
    "AI_course_policy.pdf",
    "B Tech Computer Science Time Table (Term II 2024-25)_January 10 2025.pdf",
    "DAiOT Coursepolicy.pdf",
    "DC_course_policy.pdf",
    "Ecommerce course policy.pdf",
    "First Year B Tech Time Table 2024-25_Term_II_04-01-2025.pdf",
    "HCI course policy.pdf",
    "python_course_policy.pdf",
    "SRB 2024-2025..pdf",
    "TCS course policy.pdf",
    "teacher info - Sheet1.pdf"
]

# Process PDFs
index_path = "faiss_index"
    
if not os.path.exists(index_path):
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

# Function to fetch research papers from Arxiv API
def fetch_arxiv_papers(query, max_results=5):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query={query}&start=0&max_results={max_results}"
    url = base_url + search_query
    
    response = requests.get(url)
    
    if response.status_code == 200:
        xml_content = response.text
        tree = ElementTree.fromstring(xml_content)
        entries = tree.findall("{http://www.w3.org/2005/Atom}entry")
        
        results = []
        for entry in entries:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            authors = [author.find("{http://www.w3.org/2005/Atom}name").text.strip() for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            published = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()
            link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
            
            results.append({
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "link": link
            })
        
        return results
    else:
        return f"Error: {response.status_code}"

# Integrating response generation
def generate_response(user_question):
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    wants_research_papers = any(keyword in user_question.lower() for keyword in ['research paper', 'recent papers', 'academic papers'])
    
    if wants_research_papers:
        papers = fetch_arxiv_papers(user_question)
        if isinstance(papers, str):
            return papers
        
        formatted_content = "\n\n".join(
            f"### 📖 **{paper['title']}**\n"
            f"**👨‍🏫 Authors:** {', '.join(paper['authors'])}\n"
            f"**📅 Published:** {paper['published']}\n"
            f"**🔍 Summary:** {paper['summary']}\n"
            f"🔗 [Read More]({paper['link']})\n"
            for paper in papers
        )
        return formatted_content
    
    response = user_input(user_question)
    full_response = response['output_text']
    st.session_state['chat_history'].append(("You", user_question))
    st.session_state['chat_history'].append(("Bot", full_response))
    print(st.session_state['chat_history'])
    return full_response

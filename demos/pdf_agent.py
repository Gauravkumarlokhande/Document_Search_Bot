from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from dotenv import load_dotenv
import streamlit as st
from phi.agent import AgentKnowledge
from phi.document.chunking.recursive import RecursiveChunking
from phi.vectordb.chroma import ChromaDb
from dotenv import load_dotenv
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from io import BytesIO
import os
import tempfile

load_dotenv()


db_url=os.getenv('DB_URL')
api_key=os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')


st.header('Research Tool')
user_input=st.text_input("Enter your prompt")
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())  # Write the contents of the uploaded file to the temporary file
        temp_file_path = temp_file.name 

if st.button("Answer"):
    knowledge_base = PDFKnowledgeBase(
    path=temp_file_path,
    vector_db=ChromaDb(collection="Pdf_files",
        embedder=GeminiEmbedder(model="gemini-embedding-exp-03-07",api_key=api_key))    ,  reader=PDFReader(chunk=True),chunking_strategy=RecursiveChunking(chunk_size=1500,overlap=500))
    agent = Agent(model=Groq(temperature=0, groq_api_key=groq_api_key,id="llama-3.3-70b-versatile"),knowledge=knowledge_base,search_knowledge=True)
    agent.knowledge.load(recreate=False)
    response = agent.run(user_input, stream=False)
    st.write(response)

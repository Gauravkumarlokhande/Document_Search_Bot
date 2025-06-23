import streamlit as st
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import base64
import io
import pymupdf as fitz
from PIL import Image
from IPython.display import Image as IPImage, display
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
import tempfile
import os
from langchain_ollama import OllamaEmbeddings


import openai
from typing import List

  # Or use Groq/OpenAI-compatible model API
load_dotenv("/root/DSVA/config_loader/.env")
groq_api_key = os.getenv('GROQ_API_KEY')

chat_suggest = ChatGroq(
    api_key=groq_api_key,
    temperature=0,
    model_name="deepseek-r1-distill-llama-70b"
)

def generate_suggestion_questions(user_question: str) -> List[str]:
    prompt = f"""
    
    Based on the following user question, generate 3 concise and relevant follow-up chemistry questions.
    The followup questions should be totally related to the user question.
    if the user question is not related to any chemistry related or scientific concept then just say: No Sugegstions for this question.

    User question: "{user_question}"

    Follow-up questions:
    1.
    2.
    3.
    """

    response = chat.invoke([
        {"role": "system", "content": "You are a helpful scientific assistant."},
        {"role": "user", "content": prompt},
    ])

    output = response.content
    return [line[3:].strip() for line in output.strip().splitlines() if line.startswith("1.") or line.startswith("2.") or line.startswith("3.")]




embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
# from google.colab import userdata


# api_key = os.getenv("GEMINI_API_KEY")
# embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=api_key)
pc = Pinecone(api_key='pcsk_5AtWWw_5qWvpcvukTDgQ1R9u8j48o3DgNqXFkTFAvPxnz9oTuGjSWs55WtbmnN3pbPyTTH')

# Create Index
index_name = "rag"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
print(f"Successfully connected to index: {index_name}")
# Connect to Pinecone (assuming 'index' object exists)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "score_threshold": 0.8},
)

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful chatbot called "Enhanced Document Search Chatbot" designed to assist with queries related to chemistry research and projects.
    Provide accurate step-by-step explanations of CO₂-to-ethylene reaction mechanisms.
    Identify all reactants, intermediates, products, and rate-determining steps.
    Distinguish between mechanistic pathways (e.g. formate, CO, carbene, C–C coupling).
    Specify catalyst composition, active sites, structure, and support materials.
    Report experimental conditions: temperature, pressure, electrolyte, pH, applied voltage/current, and reactor setup.
    Present performance metrics such as faradaic efficiency, selectivity, current density, and catalyst stability.
    Support mechanistic claims with DFT or experimental evidence when available.
Your tasks are to:
Provide clear, concise, and accurate responses.
Refer to relevant components, descriptions, and workflows.
Guide users based on the appropriate project information.
Use parameters provided to search for a document or provide answer related to particular project.
If possible provide exact information as retrieved in special cases such as in case of readings or observations.
If information present in the {context} is relevant to the user query then provide answer which is formed from the information available. Do not come up with information or conclusions on your own. Provide a detailed answer according to the {context}.
Your responses should ensure that users get precise and reliable insights.
Key Guidelines:
Ensure your responses are always clear, concise, and directly relevant to the user’s queries.
Maintain a professional tone in all responses.
If you cannot find an answer in the available information or if the information is not related to the query, then only respond with "No Information Available" instead of speculating.
If the context provides relevant information, use it to answer questions in a clear manner.
Do not reference the context or source documents directly in your response.
For follow-up questions, refer to the previous discussion to provide a more comprehensive response.
Respond with plain text only. Do not apply any formatting such as Markdown, LaTeX, bold, italics, or headings. Return all characters exactly as typed, including symbols like #, _, –, and backslashes. Output must preserve raw input formatting.

    if the {context} is not relevant to the user query just say, No Information available.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Define chat model and chains
chat = ChatGroq(
    api_key=groq_api_key,
    temperature=0,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)
combine_docs_chain = create_stuff_documents_chain(chat, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit interface
st.set_page_config(page_title="Scientific Assitant", layout="wide")
header = st.container()
header.title("🔬Scientific Assistant Chatbot")

# Add a small div to target for CSS
header.write('<div class="fixed-header"></div>', unsafe_allow_html=True)

# ✅ Inject CSS for sticky behavior
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;         /* adjust to position just below Streamlit’s toolbar */
        background-color: white;
        z-index: 999;
        border-bottom: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)



with st.sidebar:
    st.subheader("Chatbot Menu")
    option = st.selectbox(
        "Select answering source",
        ("Get answer from existing database", "Get answer from uploaded PDF File"),
    )




if option == "Get answer from existing database": 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.input = ""

    for history in st.session_state.chat_history:
        with st.chat_message(history["role"]):
            st.markdown(history["content"])
            
    
    if user_input := st.chat_input("What is up?"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.sidebar:
            st.subheader("suggestion questions")
            suggestions = generate_suggestion_questions(user_input)
            for i in range(len(suggestions)):
                st.markdown(f"{i+1}. {suggestions[i]}")



        with st.chat_message("assistant"):
            def get_response():
                response_dict={}
                def answer_from_bot():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        asyncio.to_thread(retrieval_chain.invoke, {'input': user_input})
                    )
                    response_dict['value']=response
                    for chunk in response['answer']:
                        yield chunk
                return answer_from_bot,response_dict 
            answer_from_bot,response_dict = get_response()
            res = st.write_stream(answer_from_bot())
            
        # context_docs = response.get("context", []) 
        st.session_state.chat_history.append({"role": "assistant", "content": res, 'docs':response_dict['value']})

    
    if st.session_state.get("chat_history"):

        last_q, last_a, last_docs = st.session_state.chat_history[-1]['docs']

        last_a=st.session_state.chat_history[-1]['docs'][last_a]
        if last_a and not "no information" in st.session_state.chat_history[-1]['content'].lower():
            st.subheader("Sources")
            for i, doc in enumerate(last_a):
                src = doc.metadata.get("source", "Unknown")
                pg  = doc.metadata.get("page", "N/A")
                st.markdown(f"{src} , Page Number: {pg}")
        else :
            st.markdown("*_No source documents used for this answer._*")

else:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = None
    if "chat_history_pdf" not in st.session_state:
        st.session_state.pdf_uploaded = None
        st.session_state.chat_history_pdf = []
    
    
    if st.button("Clear Chat"):
        st.session_state.chat_history_pdf = []
        st.session_state.input = ""
        st.session_state.pdf_uploaded = None

    
    user_question = st.chat_input("Ask your question here")
    
    uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
    st.session_state.pdf_uploaded = True
    
    

    if uploaded_file and user_question:
        for history in st.session_state.chat_history_pdf:
            with st.chat_message(history["role"]):
                st.markdown(history["content"])
        
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history_pdf.append({"role": "user", "content": user_question})
        if st.session_state.vector_store is None:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$, Full processing")


            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                pdf_path = temp_file.name

            def pdf_to_base64_images(pdf_path):
                pdf_document = fitz.open(pdf_path)
                base64_images = []
                for page_number in range(len(pdf_document)):
                    page = pdf_document.load_page(page_number)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    base64_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
                return base64_images

            base64_images = pdf_to_base64_images(pdf_path)

            prompt_img = (
                "extract the text from the images as it is and do not anything extra from your own."
            )

            embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

            docs = []

            for i, base64_image in enumerate(base64_images):
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_img},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ]
                )
                response = chat.invoke([message])

                # Add model response as a document to the vector store
                doc = Document(page_content=response.content, metadata={"page": i + 1})
                docs.append(doc)


            vector_store = FAISS.from_documents(docs, embedding)
            docs = vector_store.similarity_search(user_question, k=3)
            st.session_state.vector_store = True
            
        else:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$, short processing")

            docs = vector_store.similarity_search(user_question, k=3)
            


        # Perform similarity search in the vector store
        
        context = "\n\n".join([f"Page {doc.metadata['page']}: {doc.page_content}" for doc in docs])

        # Create a new message with context and question
        final_prompt = (
            f"Use the following extracted context to answer the question clearly and concisely.\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}"
        )

        # Invoke the model
        result = chat.invoke([HumanMessage(content=final_prompt)])
        content = result.content if isinstance(result.content, str) else str(result.content)
        with st.chat_message("assistant"):
            st.write(content)
        st.session_state.chat_history_pdf.append({"role": "assistant", "content": content})

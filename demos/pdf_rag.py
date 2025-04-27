import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

st.header('Research Tool')
user_input=st.text_input("Enter your prompt")

if st.button("Summarize"):
    model = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192",)
    result= model.invoke(user_input)

    st.write(result.content)
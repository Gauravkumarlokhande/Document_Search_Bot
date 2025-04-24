# import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


from langchain_core.prompts import PromptTemplate

prompt = (
    PromptTemplate.from_template("generate 3 similar rephrased questions for the give question :  {question}")
   
)

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the model
model = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Create a chain and invoke
chain = prompt | model


response = chain.invoke({"question": "who is elon musk?"})
print(response.text())
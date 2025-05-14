import json
import os
import yaml
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import yaml 
# from azure.communication.email import EmailClient
# from logger.loggong import log_structured


try:
    load_dotenv("config_loader/.env")
except Exception as e:
    print("env loading error: ",e)

class Config():
    def __init__(self):
        # log_structured
    
        self.embedding_model = os.getenv("GEMINI_MODEL")
        self.embedding_model_key = os.getenv("GEMINI_API_KEY")
        self.chat_model = os.getenv("GROQ_MODEL")   
        self.chat_model_key = os.getenv("GROQ_API_KEY") 
        self.db_url = os.getenv("POSTGRES_URL")
        self.storage = None

        try:
            self.llm = ChatGroq(temperature=0, groq_api_key=self.chat_model_key, model_name=self.chat_model)
        except Exception as e:
            print(e)


        try:
            self.embed = GoogleGenerativeAIEmbeddings(model=self.embedding_model,google_api_key=self.embedding_model_key)
        except Exception as e:
            print(e)


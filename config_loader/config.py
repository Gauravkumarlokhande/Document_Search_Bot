import json
import os
import yaml
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import yaml 
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncAzureOpenAI
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
    




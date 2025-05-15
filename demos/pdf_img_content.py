import base64
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import httpx
from langchain.chat_models import init_chat_model
import yaml
import os

with open(r"/root/DSVA/demos/prompt.yaml") as file:
    prompt = yaml.safe_load(file)
    prompt_temp=prompt["multimodal_prompt"]

image_url=r"/root/DSVA/processed_data/Machine Learning with Python/Machine Learning with Python_page_1.png"
with open(image_url,"rb") as img:
    image_data = base64.b64encode(img.read()).decode("utf-8")

load_dotenv("config_loader/.env")

groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": prompt_temp,
        },
        {
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": "image/jpeg",
        },
    ],
}
response = model.invoke([message])
print(response.text())
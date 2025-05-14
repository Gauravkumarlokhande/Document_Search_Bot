from pymilvus import MilvusClient
from pymilvus import model
import os
from dotenv import load_dotenv

load_dotenv("config_loader/.env")

api=os.getenv("GEMINI_API_KEY")

gemini_ef = model.dense.GeminiEmbeddingFunction(
    model_name='gemini-embedding-exp-03-07', # Specify the model name
    api_key=api, # Provide your OpenAI API key
)
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

docs_embeddings = gemini_ef.encode_documents(docs)

# Print embeddings
print("Embeddings:", docs_embeddings)

client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)




# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": docs_embeddings[i], "text": docs[i], "subject": "history"}
    for i in range(len(docs_embeddings))
]

res = client.insert(collection_name="demo_collection", data=data)
print(res)
from fastapi import FastAPI
from pydantic import BaseModel

class health(BaseModel):
    name:str 
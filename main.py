from fastapi import 
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager # this is used when asynchronous setup and teardown is required , such as databse connections
from uuid import uuid4  # used to generate unique random user id or request id
from fastapi import Request #fastapi request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

origins=["*"]

@asynccontextmanager
async def lifespan(app:FastAPI):
    await create_tables()
    yield 


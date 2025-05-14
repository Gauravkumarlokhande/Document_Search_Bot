from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager # this is used when asynchronous setup and teardown is required , such as databse connections
from uuid import uuid4  # used to generate unique random user id or request id
from fastapi import Request #fastapi request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from router.health import router as health_router
from structures.table_structure import create_table

origins=["*"]

# async def say_something():
#     return "hello"

@asynccontextmanager  # this is for the lifespan of the application. anaything before yield is at the startup and after yeild is at the shut down of app. used for resources that require startup and shutdown at the lifespan of the application
async def lifespan(app:FastAPI):
    
    await create_table()

    # await say_something()
    yield 

app = FastAPI(lifespan=lifespan)
app.include_router(health_router)

app.add_middleware(CORSMiddleware,allow_origins=origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"])

@app.exception_handler(RequestValidationError) # this request validation executes the custom_validation_exception_handler whenever there is any error in the app
async def custom_validation_exception_handler(
    request:Request , exc: RequestValidationError):
    try:
        body = await request.json()
        request_id=body.get("request_id",str(uuid4()))  # tries to search for the request id of the request, if not then creates a new request id
    except Exception: # if reading the body itself is failed in the try block then it will generate new id
        request_id = str(uuid4())
    
    try:
        error_details=exc.errors()[0] # searching for error message
        error_message = f"{error_details.get('msg')}: {error_details.get('loc')[-1]}"
    except Exception:
        
        error_message='Field Validation Error'
    
    return JSONResponse(
        status_code=422,
        content={
            "isSuccess":False,
            "response":{"request_id":request_id,"response":"null"},
            "error":error_message
        }
    )
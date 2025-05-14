from datetime import datetime
from typing import AsyncGenerator
from langchain_community.chatmessage_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pyodbc
import traceback
import pytz
from config_loader.config import cfg
from retriever.wrapped_get_content import get_content 
from models.chat_prompt_template_model import prompt 
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_community.callbacks import get_openai_callback
from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError
from models.chain_history_model import build_qa_chain
from retriever.handle_recent_history import handle_query_with_recent_history
from email_notifier.automated_mail_sender import get_msg, send_mail
from api.dependencies.auth import get_api_key
# data insertion
#retriever
from models.schemas import ApiResponse, UnifiedRequest
# logging
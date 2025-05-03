from contextlib import asynccontextmanager
import numpy as numpy
from sqlalchemy import (
    BigInteger,CheckConstraint,Column,DateTime,Float, ForeignKey, Integer, LargeBinary, Sequence, String, Text, Boolean
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import os
from dotenv import load_dotenv
load_dotenv()

db_url=os.getenv('DB_URL')

db_engine=create_async_engine(db_url,pool_size=10, max_overflow=10, pool_timeout=30)

#session factory for session creation with async
AsyncSessionLocal = sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base() # base class for orm models and you can inherit this class

@asynccontextmanager
async def async_session_scope():
    """Asynchronous context manager for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
            # log_structured
        except Exception as e:
            await session.rollback()
            # log_structured
        finally:
            await session.close()
            # log_structured


class UnifiedRequest(Base):
    __tablename__ = "unified_request"
    request_id = Column(String(255),primary_key=True)
    timestamp = Column(DateTime)
    user_email = Column(String(255))
    request_type = Column(String(255))
    chat_id = Column(String(255))
    question = Column(Text,nullable=True)
    doc_identifier = Column(Text,nullable=True)
    source_docs = Column(Text,nullable=True)

class ApiResponse(Base):
    __tablename__ = "api_response"
    response_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(255), ForeignKey("unified_request.request_id"))
    response = Column(Text)
    total_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    request = relationship("UnifiedRequest")

class RequestStatus(Base):
    __tablename__="request_status"
    status_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id  = Column(String, ForeignKey("unified_request.request_id"))
    status = Column(String(255))
    error_message = Column(Text, nullable=True)
    timestamp = Column(Float, nullable=True)
    request = relationship("UnifiedRequest")

class RetrievedChunks(Base):
    __tablename__="retrieved_chunks"
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(255), ForeignKey("unified_request.request_id"), nullable=True)
    document_metadata = Column(Text, nullable=True)
    page_content = Column(Text, nullable=True)
    sequence_number = Column(Integer, nullable=True)
    request = relationship("UnifiedRequest")


class SourceDocuments(Base):
    __tablename__ = "source_documents"
    document_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(255), ForeignKey("unified_request.request_id"))
    answer = Column(Text)
    source_documents = Column(Text)
    request = relationship("UnifiedRequest")

class Cache(Base):
    __tablename__ = "cache"
    id = Column(BigInteger, Sequence("cache_id_seq"), primary_key=True, autoincrement=True)
    question = Column(String)
    response = Column(String)
    user_email = Column(String)
    source_documents = Column(Text, nullable=True)
    semantic_vector = Column(LargeBinary, nullable=True)
    source_docs = Column(Text,nullable=True)
    time_added = Column(DateTime)

    @classmethod
    async def load_from_db(cls, session:AsyncSession, user_input, user_email,threshold=0.65):
        pass
        # try:
        #     # log_structured
        #     user_vector = np.array(cfg.)

async def create_table():
    try:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # log_structured
    except Exception as e:
        print(e)
        # log_structured



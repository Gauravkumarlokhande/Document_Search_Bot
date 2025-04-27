from contextlib import asynccontextmanager
import numpy as numpy
from sqlalchemy import (
    BigInteger,CheckConstraint,Column,DateTime,Float, ForeignKey, Integer, LargeBinary, Sequence, String, Text, Boolean
)
 from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
 from sqlalchemy.future import select
 from sqlalchemy.orm import declarative_base, relationship, sessionmaker

 db_engine=create_async_engine()
from sqlalchemy import create_engine,text # this engine connects to database
from dotenv import load_dotenv
import os


load_dotenv()
db_url=os.getenv("DB_URL")


engine = create_engine(db_url,echo=True) # here we need a database url
conn=engine.connect()
conn.execute(text("CREATE TABLE IF NOT EXISTS people (name TEXT, age INT)"))
conn.commit()
# created a postgres database on railway

from sqlalchemy.orm import Session
session=Session(engine)
session.execute(text("INSERT INTO people (name,age) VALUES ('mike',30);"))
session.commit()
# basic table creation and data insertion
